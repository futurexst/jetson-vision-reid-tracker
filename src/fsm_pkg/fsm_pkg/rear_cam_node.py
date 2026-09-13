import os
import time
import threading
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PKG_DIR, "yolo11n.engine")

# ===== 파라미터 =====
CAM_DEV = "/dev/v4l/by-id/usb-Suyin_HD_Camera_200910120001-video-index0"
IMG_WIDTH = 640
IMG_HEIGHT = 480
INFER_HZ = 5.0           # 추론 주기 (사람 있다/없다 판단엔 충분)
CONF_THRES = 0.5         # 사람 검출 신뢰도 임계값
STREAM_PORT = 8081
# ====================


# ============================================================
# MJPEG 스트리밍 서버 (후방, 포트 8081)
# 브라우저: http://192.168.0.103:8081/stream
# ============================================================
_stream_lock = threading.Lock()
_stream_frame = None

class _MjpegHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _stream_lock:
                    frame = _stream_frame
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        data = buf.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                time.sleep(0.1)
        except Exception:
            pass

def _start_stream_server(port=STREAM_PORT):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _MjpegHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
# ============================================================


class RearCamNode(Node):
    def __init__(self):
        super().__init__("rear_cam_node")
        self.person_pub = self.create_publisher(Bool, "/rear_person", 10)

        self.get_logger().info("loading YOLO...")
        self.model = YOLO(MODEL_PATH, task="detect")

        self.cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
        if not self.cap.isOpened():
            self.get_logger().error("후방 카메라 열기 실패")

        _start_stream_server(STREAM_PORT)
        self.get_logger().info(f"MJPEG stream: http://192.168.0.103:{STREAM_PORT}/stream")
        self.get_logger().info("rear_cam_node started.")

        self.timer = self.create_timer(1.0 / INFER_HZ, self.loop)

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        results = self.model.predict(
            frame, classes=[0], conf=CONF_THRES, verbose=False)
        boxes = results[0].boxes
        annotated = frame.copy()

        person_found = False
        if boxes is not None and len(boxes) > 0:
            person_found = True
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # /rear_person 발행
        msg = Bool()
        msg.data = person_found
        self.person_pub.publish(msg)

        # 화면 표시
        status = "PERSON" if person_found else "NO PERSON"
        color = (0, 255, 0) if person_found else (0, 0, 255)
        cv2.putText(annotated, f"REAR: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        global _stream_frame
        with _stream_lock:
            _stream_frame = annotated.copy()

    def destroy_node(self):
        try:
            self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RearCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
