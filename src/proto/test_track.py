import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

for _ in range(30):
    pipeline.wait_for_frames()


def median_depth(depth_frame, cx, cy, k=5):
    """중심점 주변 k x k 영역의 median 거리 (0.00m 홀 방지)"""
    half = k // 2
    vals = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            d = depth_frame.get_distance(cx + dx, cy + dy)
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0


print("running 150 frames... (사람이 좌우로 움직여 보세요)")
for i in range(150):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        continue

    img = np.asanyarray(color.get_data())

    # persist=True: 프레임 간 트랙 유지 / 커스텀 ReID yaml 사용
    results = model.track(
        img, persist=True, classes=[0],
        tracker="botsort_reid.yaml", verbose=False)

    boxes = results[0].boxes
    annotated = img.copy()

    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = median_depth(depth, cx, cy)
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"ID:{tid} {dist:.2f}m",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            if i % 25 == 0:
                print(f"frame {i}: ID={tid} cx={cx} dist={dist:.2f}m conf={conf:.2f}")

    if i == 149:
        cv2.imwrite("track_result.jpg", annotated)
        print("saved track_result.jpg")

pipeline.stop()
