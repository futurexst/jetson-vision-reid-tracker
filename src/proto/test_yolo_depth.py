import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# depth를 color에 정렬
align = rs.align(rs.stream.color)

for _ in range(30):
    pipeline.wait_for_frames()

print("running 100 frames...")
for i in range(100):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        continue

    img = np.asanyarray(color.get_data())

    # class 0 = person 만 검출
    results = model(img, classes=[0], verbose=False)
    boxes = results[0].boxes

    annotated = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = depth.get_distance(cx, cy)
        conf = float(box.conf[0])
        label = f"{dist:.2f}m {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if i % 20 == 0:
            print(f"frame {i}: person at ({cx},{cy}) dist={dist:.2f}m conf={conf:.2f}")

    if i == 99:
        cv2.imwrite("yolo_depth_result.jpg", annotated)
        print("saved yolo_depth_result.jpg")

pipeline.stop()
