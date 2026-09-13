cmake_minimum_required(VERSION 3.8)
project(follow_tracker_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(follow_tracker_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
set(REALSENSE2_INCLUDE_DIR /usr/local/include)
set(REALSENSE2_LIB /usr/local/lib/librealsense2.so)

add_library(camera_infer_node SHARED
  src/camera_infer_node.cpp)

target_include_directories(camera_infer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${REALSENSE2_INCLUDE_DIR})

target_link_libraries(camera_infer_node
  ${OpenCV_LIBS}
  ${REALSENSE2_LIB}
  nvinfer
  ${CUDA_LIBRARIES})

ament_target_dependencies(camera_infer_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  follow_tracker_msgs)

rclcpp_components_register_node(camera_infer_node
  PLUGIN "follow_tracker_cpp::CameraInferNode"
  EXECUTABLE camera_infer_node_exe)

# ===== ReID 엔진 라이브러리 + 검증 실행파일 =====
add_library(reid_engine SHARED
  src/reid_engine.cpp)
target_include_directories(reid_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS})
target_link_libraries(reid_engine
  ${OpenCV_LIBS}
  nvinfer
  ${CUDA_LIBRARIES})
ament_target_dependencies(reid_engine
  rclcpp)

add_executable(reid_check
  src/reid_check.cpp)
target_link_libraries(reid_check
  reid_engine
  ${OpenCV_LIBS}
  nvinfer
  ${CUDA_LIBRARIES})
ament_target_dependencies(reid_check
  rclcpp)

# ===== tracker_node (IoU 트래커 + CLIP-ReID 락/게이트) =====
add_library(tracker_node SHARED
  src/tracker_node.cpp
  src/iou_tracker.cpp)
target_include_directories(tracker_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS})
target_link_libraries(tracker_node
  reid_engine
  ${OpenCV_LIBS}
  nvinfer
  ${CUDA_LIBRARIES})
ament_target_dependencies(tracker_node
  rclcpp
  rclcpp_components
  geometry_msgs
  std_msgs
  sensor_msgs
  cv_bridge
  follow_tracker_msgs)
rclcpp_components_register_node(tracker_node
  PLUGIN "follow_tracker_cpp::TrackerNode"
  EXECUTABLE tracker_node_exe)

install(TARGETS
  camera_infer_node
  reid_engine
  reid_check
  tracker_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
