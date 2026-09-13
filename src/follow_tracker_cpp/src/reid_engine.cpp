#include "follow_tracker_cpp/reid_engine.hpp"
#include <fstream>
#include <cmath>
#include <cstring>

namespace follow_tracker_cpp
{

void ReidTrtLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    RCLCPP_INFO(rclcpp::get_logger("reid_engine"), "[TRT] %s", msg);
  }
}

ReidEngine::ReidEngine() {}

ReidEngine::~ReidEngine()
{
  if (gpu_in_) {cudaFree(gpu_in_);}
  if (gpu_out_) {cudaFree(gpu_out_);}
  if (stream_) {cudaStreamDestroy(stream_);}
  if (context_) {delete context_;}
  if (engine_) {delete engine_;}
  if (runtime_) {delete runtime_;}
}

bool ReidEngine::load(const std::string & engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    RCLCPP_ERROR(rclcpp::get_logger("reid_engine"), "엔진 파일 열기 실패: %s", engine_path.c_str());
    return false;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    return false;
  }

  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {return false;}
  engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
  if (!engine_) {return false;}
  context_ = engine_->createExecutionContext();
  if (!context_) {return false;}

  // I/O 텐서 이름/차원 확인 (TensorRT 10 API)
  int nb = engine_->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char * name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    auto dims = engine_->getTensorShape(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      in_name_ = name;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      out_name_ = name;
      // 출력 마지막 차원을 feat_dim으로 (1x1280 -> 1280)
      feat_dim_ = dims.d[dims.nbDims - 1];
    }
  }

  cudaStreamCreate(&stream_);
  cudaMalloc(&gpu_in_, static_cast<size_t>(IN_C) * IN_H * IN_W * sizeof(float));
  cudaMalloc(&gpu_out_, static_cast<size_t>(feat_dim_) * sizeof(float));

  ready_ = true;
  RCLCPP_INFO(
    rclcpp::get_logger("reid_engine"),
    "ReID 엔진 로드 완료: %s (in=%s out=%s feat_dim=%d)",
    engine_path.c_str(), in_name_.c_str(), out_name_.c_str(), feat_dim_);
  return true;
}

// CLIP 전처리: BGR crop -> RGB, 128x256, /255, (x-0.5)/0.5, CHW float
void ReidEngine::preprocess(const cv::Mat & crop_bgr, std::vector<float> & out)
{
  cv::Mat resized;
  // OpenCV resize는 (width, height) = (IN_W, IN_H) = (128, 256)
  cv::resize(crop_bgr, resized, cv::Size(IN_W, IN_H), 0, 0, cv::INTER_LINEAR);

  out.resize(static_cast<size_t>(IN_C) * IN_H * IN_W);
  // CHW, RGB, (x/255 - 0.5) / 0.5
  const int plane = IN_H * IN_W;
  for (int y = 0; y < IN_H; ++y) {
    const cv::Vec3b * row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < IN_W; ++x) {
      const cv::Vec3b & px = row[x];  // BGR
      float b = px[0] / 255.0f;
      float g = px[1] / 255.0f;
      float r = px[2] / 255.0f;
      int idx = y * IN_W + x;
      // RGB 순서로 채널 배치, (v-0.5)/0.5
      out[0 * plane + idx] = (r - 0.5f) / 0.5f;
      out[1 * plane + idx] = (g - 0.5f) / 0.5f;
      out[2 * plane + idx] = (b - 0.5f) / 0.5f;
    }
  }
}

std::vector<float> ReidEngine::extract(const cv::Mat & crop_bgr)
{
  std::vector<float> feat;
  if (!ready_ || crop_bgr.empty()) {return feat;}

  std::vector<float> input;
  preprocess(crop_bgr, input);

  cudaMemcpyAsync(
    gpu_in_, input.data(), input.size() * sizeof(float),
    cudaMemcpyHostToDevice, stream_);

  context_->setTensorAddress(in_name_.c_str(), gpu_in_);
  context_->setTensorAddress(out_name_.c_str(), gpu_out_);
  context_->enqueueV3(stream_);

  feat.resize(feat_dim_);
  cudaMemcpyAsync(
    feat.data(), gpu_out_, feat_dim_ * sizeof(float),
    cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // L2 정규화
  float norm = 0.0f;
  for (float v : feat) {norm += v * v;}
  norm = std::sqrt(norm);
  if (norm > 1e-8f) {
    for (float & v : feat) {v /= norm;}
  }
  return feat;
}

std::vector<std::vector<float>> ReidEngine::extract_boxes(
  const std::vector<cv::Rect> & boxes, const cv::Mat & img)
{
  std::vector<std::vector<float>> feats;
  feats.reserve(boxes.size());
  const int W = img.cols, H = img.rows;
  for (const auto & b : boxes) {
    // clamp (파이썬: max(0,x1), min(w,x2) ...)
    int x1 = std::max(0, b.x);
    int y1 = std::max(0, b.y);
    int x2 = std::min(W, b.x + b.width);
    int y2 = std::min(H, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) {
      feats.emplace_back();  // 빈 임베딩
      continue;
    }
    cv::Mat crop = img(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    feats.emplace_back(extract(crop));
  }
  return feats;
}

}  // namespace follow_tracker_cpp
