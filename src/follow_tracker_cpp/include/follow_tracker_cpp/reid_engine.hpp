#ifndef FOLLOW_TRACKER_CPP__REID_ENGINE_HPP_
#define FOLLOW_TRACKER_CPP__REID_ENGINE_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

namespace follow_tracker_cpp
{

// TensorRT 로거 (ReID 전용)
class ReidTrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

// CLIP-ReID TensorRT 추론 엔진
// 입력: crop 이미지들(BGR, cv::Mat) -> 출력: L2 정규화된 512/1280차 임베딩
class ReidEngine
{
public:
  ReidEngine();
  ~ReidEngine();

  // 엔진 로드. 성공 시 true.
  bool load(const std::string & engine_path);

  // 단일 crop(BGR) -> L2 정규화 임베딩. 실패 시 빈 vector.
  std::vector<float> extract(const cv::Mat & crop_bgr);

  // 여러 bbox를 원본 이미지에서 잘라 각각 임베딩 추출 (배치 아님, 순차)
  // xyxy: {x1,y1,x2,y2}, img: 원본 BGR
  std::vector<std::vector<float>> extract_boxes(
    const std::vector<cv::Rect> & boxes, const cv::Mat & img);

  int feat_dim() const {return feat_dim_;}
  bool ready() const {return ready_;}

private:
  // CLIP 전처리: crop(BGR) -> CHW float [(x/255)-0.5]/0.5, RGB, 128x256
  void preprocess(const cv::Mat & crop_bgr, std::vector<float> & out);

  ReidTrtLogger logger_;
  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};
  cudaStream_t stream_{nullptr};

  void * gpu_in_{nullptr};
  void * gpu_out_{nullptr};

  // CLIP 입력 규격: 1x3x256x128 (N,C,H,W)
  static constexpr int IN_H = 256;
  static constexpr int IN_W = 128;
  static constexpr int IN_C = 3;
  int feat_dim_{1280};        // 출력 차원 (엔진에서 확정)
  std::string in_name_;
  std::string out_name_;
  bool ready_{false};
};

}  // namespace follow_tracker_cpp

#endif  // FOLLOW_TRACKER_CPP__REID_ENGINE_HPP_
