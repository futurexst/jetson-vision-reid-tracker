import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# ===== 설정 =====
TARGET_DIST = 1.5          # 유지 목표 거리 (m)
DIST_DEADZONE = 0.15       # 목표거리 ±이 범위 내면 전후진 정지 (m)
CENTER_X = 320             # 화면 가로 중앙 (640 기준)
CX_DEADZONE = 40           # 중앙 ±이 범위 내면 회전 정지 (px)

MAX_LINEAR = 0.25          # 최대 선속도 (m/s)
MAX_ANGULAR = 0.6          # 최대 각속도 (rad/s)
KP_LINEAR = 0.4            # 거리 오차 -> 선속도 게인
KP_ANGULAR = 0.003         # 픽셀 오차 -> 각속도 게인

LOCK_FRAMES = 15           # 시작 후 안정화 뒤 타겟 락
# ================

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
    half = k // 2
    vals = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            d = depth_frame.get_distance(cx + dx, cy + dy)
            if d > 0:
                vals.append(d)
    return float(np.median(vals)) if vals else 0.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


target_id = None
frame_count = 0

print("'q' 키로 종료 / 시작 시 가장 가까운 사람을 타겟으로 락")
while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        continue

    img = np.asanyarray(color.get_data())
    results = model.track(
        img, persist=True, classes=[0],
        tracker="botsort_reid.yaml", verbose=False)

    boxes = results[0].boxes
    annotated = img.copy()
    frame_count += 1

    people = {}
    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = median_depth(depth, cx, cy)
            people[tid] = (cx, cy, dist, (x1, y1, x2, y2))

    if target_id is None and frame_count >= LOCK_FRAMES and people:
        valid = {tid: v for tid, v in people.items() if v[2] > 0}
        if valid:
            target_id = min(valid, key=lambda t: valid[t][2])
            print(f"[LOCK] target_id={target_id} dist={valid[target_id][2]:.2f}m")

    linear, angular = 0.0, 0.0
    status = "SEARCHING"

    if target_id is not None and target_id in people:
        cx, cy, dist, (x1, y1, x2, y2) = people[target_id]
        if dist > 0:
            dist_err = dist - TARGET_DIST
            if abs(dist_err) > DIST_DEADZONE:
                linear = clamp(KP_LINEAR * dist_err, -MAX_LINEAR, MAX_LINEAR)
            cx_err = cx - CENTER_X
            if abs(cx_err) > CX_DEADZONE:
                angular = clamp(-KP_ANGULAR * cx_err, -MAX_ANGULAR, MAX_ANGULAR)
            status = "FOLLOWING"
        else:
            status = "TARGET_NO_DEPTH"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"TARGET ID:{target_id} {dist:.2f}m",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    elif target_id is not None:
        status = "TARGET_LOST"

    for tid, (cx, cy, dist, (x1, y1, x2, y2)) in people.items():
        if tid == target_id:
            continue
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.line(annotated, (CENTER_X, 0), (CENTER_X, 480), (255, 0, 0), 1)
    cv2.putText(annotated, f"{status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated, f"lin:{linear:+.3f} ang:{angular:+.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("follow", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
