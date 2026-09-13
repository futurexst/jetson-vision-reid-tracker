# camera_infer -> tracker 제로카피 전달용
std_msgs/Header header
Detection[] detections       # 검출 사람 배열
sensor_msgs/Image color      # 컬러 프레임 (ReID crop용)
sensor_msgs/Image depth      # depth 프레임 (거리 재계산용)
