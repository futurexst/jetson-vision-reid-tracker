#include "follow_tracker_cpp/camera_infer_node.hpp"
#include <fstream>
#include <opencv2/dnn.hpp>
#include <cv_bridge/cv_bridge.h>

namespace follow_tracker_cpp
{

void TrtLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    RCLCPP_INFO(rclcpp::get_logger("camera_infer_node"), "[TRT] %s", msg);
  }
}

CameraInferNode::CameraInferNode(const rclcpp::NodeOptions & options)
: Node("camera_infer_node", options)
{
  det_pub_ = this->create_publisher<follow_tracker_msgs::msg::DetectionSet>("~/detections", 10);

  std::string engine_path = this->declare_parameter<std::string>(
    "engine_path",
    "models/yolo11n_cpp.engine");

  if (!load_engine(engine_path)) {
    RCLCPP_ERROR(this->get_logger(), "엔진 로드 실패: %s", engine_path.c_str());
  }

  cudaStreamCreate(&cuda_stream_);

  try {
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    rs2::pipeline_profile profile = rs_pipe_.start(cfg);
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR)
      .as<rs2::video_stream_profile>();
    color_intrin_ = color_stream.get_intrinsics();
    rs_started_ = true;
    RCLCPP_INFO(this->get_logger(), "RealSense 파이프라인 시작 완료");
  } catch (const rs2::error & e) {
    RCLCPP_ERROR(this->get_logger(), "RealSense 시작 실패: %s", e.what());
  }

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(33),
    std::bind(&CameraInferNode::timer_callback, this));
  RCLCPP_INFO(this->get_logger(), "camera_infer_node started");
}

CameraInferNode::~CameraInferNode()
{
  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
  }
  if (trt_context_) {
    delete trt_context_;
  }
  if (trt_engine_) {
    delete trt_engine_;
  }
  if (trt_runtime_) {
    delete trt_runtime_;
  }
}

bool CameraInferNode::load_engine(const std::string & engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    return false;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    return false;
  }

  trt_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
  if (!trt_runtime_) {
    return false;
  }
  trt_engine_ = trt_runtime_->deserializeCudaEngine(buffer.data(), size);
  if (!trt_engine_) {
    return false;
  }
  trt_context_ = trt_engine_->createExecutionContext();
  if (!trt_context_) {
    return false;
  }
  RCLCPP_INFO(this->get_logger(), "TensorRT 엔진 로드 완료: %s", engine_path.c_str());

  cudaMalloc(&gpu_input_, 1 * 3 * 640 * 640 * sizeof(float));
  cudaMalloc(&gpu_output_, 1 * 84 * 8400 * sizeof(float));

  return true;
}

void CameraInferNode::timer_callback()
{
  if (!rs_started_) {
    return;
  }

  rs2::frameset frames;
  try {
    frames = rs_pipe_.wait_for_frames(1000);
  } catch (const rs2::error & e) {
    RCLCPP_WARN(this->get_logger(), "프레임 대기 실패: %s", e.what());
    return;
  }

  frames = rs_align_.process(frames);
  rs2::video_frame color = frames.get_color_frame();
  rs2::depth_frame depth = frames.get_depth_frame();
  if (!color || !depth) {
    return;
  }

  cv::Mat img(cv::Size(color.get_width(), color.get_height()), CV_8UC3,
    const_cast<void *>(color.get_data()), cv::Mat::AUTO_STEP);

  // letterbox: 종횡비 유지 + 패딩
  const int net_w = 640, net_h = 640;
  float r = std::min(static_cast<float>(net_w) / img.cols,
                     static_cast<float>(net_h) / img.rows);
  int new_w = static_cast<int>(std::round(img.cols * r));
  int new_h = static_cast<int>(std::round(img.rows * r));
  int pad_x = (net_w - new_w) / 2;
  int pad_y = (net_h - new_h) / 2;
  cv::Mat resized_tmp;
  cv::resize(img, resized_tmp, cv::Size(new_w, new_h));
  cv::Mat resized(net_h, net_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized_tmp.copyTo(resized(cv::Rect(pad_x, pad_y, new_w, new_h)));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat float_img;
  rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> chw(3);
  cv::split(float_img, chw);
  std::vector<float> input_data(3 * 640 * 640);
  for (int c = 0; c < 3; ++c) {
    memcpy(input_data.data() + c * 640 * 640, chw[c].data, 640 * 640 * sizeof(float));
  }

  cudaMemcpyAsync(gpu_input_, input_data.data(), input_data.size() * sizeof(float),
    cudaMemcpyHostToDevice, cuda_stream_);

  trt_context_->setTensorAddress("images", gpu_input_);
  trt_context_->setTensorAddress("output0", gpu_output_);
  trt_context_->enqueueV3(cuda_stream_);

  std::vector<float> output_data(84 * 8400);
  cudaMemcpyAsync(output_data.data(), gpu_output_, output_data.size() * sizeof(float),
    cudaMemcpyDeviceToHost, cuda_stream_);
  cudaStreamSynchronize(cuda_stream_);

  // ---- output0 디코딩 (1x84x8400, transposed) ----
  const int num_anchors = 8400;
  const float conf_thresh = 0.35f;
  const float nms_thresh = 0.45f;
  const int person_cls = 0;

  // letterbox 역변환: (net좌표 - pad) / r
  const float inv_r = 1.0f / r;

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;

  for (int i = 0; i < num_anchors; ++i) {
    float cls_score = output_data[(4 + person_cls) * num_anchors + i];
    if (cls_score < conf_thresh) {
      continue;
    }
    float cx = output_data[0 * num_anchors + i];
    float cy = output_data[1 * num_anchors + i];
    float w  = output_data[2 * num_anchors + i];
    float h  = output_data[3 * num_anchors + i];

    float x0 = (cx - w * 0.5f - pad_x) * inv_r;
    float y0 = (cy - h * 0.5f - pad_y) * inv_r;
    float bw = w * inv_r;
    float bh = h * inv_r;

    boxes.emplace_back(cv::Rect(
      static_cast<int>(x0), static_cast<int>(y0),
      static_cast<int>(bw), static_cast<int>(bh)));
    scores.emplace_back(cls_score);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, conf_thresh, nms_thresh, keep);

  RCLCPP_INFO(this->get_logger(), "person detections: %zu", keep.size());


  // ===== DetectionSet 발행 (검출 전체 + 컬러/depth 이미지, 제로카피 전달) =====
  auto det_msg = std::make_unique<follow_tracker_msgs::msg::DetectionSet>();
  det_msg->header.stamp = this->get_clock()->now();
  det_msg->header.frame_id = "camera_link";

  const int dw = static_cast<int>(depth.get_width());
  const int dh = static_cast<int>(depth.get_height());

  for (int idx : keep) {
    const cv::Rect & br = boxes[idx];
    follow_tracker_msgs::msg::Detection d;
    d.x = br.x;
    d.y = br.y;
    d.w = br.width;
    d.h = br.height;
    d.conf = scores[idx];
    int pcx = br.x + br.width / 2;
    int pcy = br.y + br.height / 2;
    pcx = std::max(0, std::min(pcx, dw - 1));
    pcy = std::max(0, std::min(pcy, dh - 1));
    float dist = depth.get_distance(pcx, pcy);
    if (dist > 0.0f) {
      float pixel[2] = {static_cast<float>(pcx), static_cast<float>(pcy)};
      float point[3] = {0.0f, 0.0f, 0.0f};
      rs2_deproject_pixel_to_point(point, &color_intrin_, pixel, dist);
      d.px = point[0];
      d.py = point[1];
      d.pz = point[2];
    } else {
      d.px = 0.0f;
      d.py = 0.0f;
      d.pz = 0.0f;
    }
    det_msg->detections.push_back(d);
  }

  // 컬러 프레임 -> sensor_msgs/Image
  std_msgs::msg::Header img_header = det_msg->header;
  cv_bridge::CvImage(img_header, "bgr8", img).toImageMsg(det_msg->color);

  // depth 프레임 -> sensor_msgs/Image (16UC1, mm 단위 원본)
  cv::Mat depth_mat(cv::Size(dw, dh), CV_16UC1,
    const_cast<void *>(depth.get_data()), cv::Mat::AUTO_STEP);
  cv_bridge::CvImage(img_header, "16UC1", depth_mat).toImageMsg(det_msg->depth);

  det_pub_->publish(std::move(det_msg));
}

}  // namespace follow_tracker_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(follow_tracker_cpp::CameraInferNode)
