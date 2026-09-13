import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
print("pipeline started")

# 자동 노출 안정화를 위해 앞 프레임 버림
for _ in range(30):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color = frames.get_color_frame()
depth = frames.get_depth_frame()

color_img = np.asanyarray(color.get_data())
depth_img = np.asanyarray(depth.get_data())

print("color shape:", color_img.shape)
print("depth shape:", depth_img.shape)

# 중앙 픽셀 거리 측정
cx, cy = 320, 240
dist = depth.get_distance(cx, cy)
print(f"center distance: {dist:.3f} m")

cv2.imwrite("test_color.jpg", color_img)
depth_vis = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
cv2.imwrite("test_depth.jpg", depth_vis)
print("saved test_color.jpg, test_depth.jpg")

pipeline.stop()
