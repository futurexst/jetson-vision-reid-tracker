#ifndef FOLLOW_TRACKER_CPP__CAMERA_INFER_NODE_HPP_
#define FOLLOW_TRACKER_CPP__CAMERA_INFER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <follow_tracker_msgs/msg/detection_set.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <memory>
#include <string>

namespace follow_tracker_cpp
{

class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

class CameraInferNode : public rclcpp::Node
{
public:
  explicit CameraInferNode(const rclcpp::NodeOptions & options);
  ~CameraInferNode();

private:
  void timer_callback();
  bool load_engine(const std::string & engine_path);

  rclcpp::Publisher<follow_tracker_msgs::msg::DetectionSet>::SharedPtr det_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  TrtLogger trt_logger_;
  nvinfer1::IRuntime * trt_runtime_{nullptr};
  nvinfer1::ICudaEngine * trt_engine_{nullptr};
  nvinfer1::IExecutionContext * trt_context_{nullptr};
  cudaStream_t cuda_stream_{nullptr};
  void * gpu_input_{nullptr};
  void * gpu_output_{nullptr};

  rs2::pipeline rs_pipe_;
  rs2::align rs_align_{RS2_STREAM_COLOR};
  rs2_intrinsics color_intrin_{};
  bool rs_started_{false};
};

}  // namespace follow_tracker_cpp

#endif  // FOLLOW_TRACKER_CPP__CAMERA_INFER_NODE_HPP_
