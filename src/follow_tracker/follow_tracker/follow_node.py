import os
import json
import time
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs
from ultralytics import YOLO
from pathlib import Path
from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config
from follow_tracker import boxmot_patch
from follow_tracker import reid_calib

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Int32, String
from geometry_msgs.msg import Point

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 라벨용 폰트 (Noto Sans CJK KR) 로딩 =====
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
def _load_font(size):
    """Noto CJK .ttc에서 한국어 서브폰트를 찾아 로드. 실패 시 None."""
    try:
        return ImageFont.truetype(_FONT_PATH, size, index=1)
    except Exception:
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            return None
LABEL_FONT = _load_font(20)
LABEL_FONT_SMALL = _load_font(13)
MODEL_PATH = os.path.join(PKG_DIR, "yolo11n.engine")
BLOCK_MODEL_PATH = os.path.expanduser("~/follow_ws/src/block_detector/models/block_best.engine")
TRACKER_PATH = os.path.join(PKG_DIR, "botsort_reid.yaml")

# ===== 파라미터 =====
CENTER_X = 320
IMG_WIDTH = 640
HFOV_DEG = 69.0
LOCK_FRAMES = 15          # 워밍업 프레임
SHOW_WINDOW = False
LOST_DELAY = 8            # 사람 사라진 후 이 프레임 지나야 탐색 시작 (과민 방지)
FIND_HOLD_SEC = 0.5       # find(재발견) 시 정지 유지 시간

# linear.z 모드 플래그
MODE_FOLLOW = 0.0
MODE_SEARCH_LEFT = 1.0
MODE_SEARCH_RIGHT = 2.0
MODE_FIND = 3.0
# ===== 블록 검출 파라미터 =====
BLOCK_CONF = 0.60          # 블록 검출 최소 신뢰도
ROI_HALF = 70            # 레이저 지점 기준 crop 반경(px), 블록 하나만 포함
BLOCK_MAX_Z = 1.6          # 블록 최대 거리(m), 실운용 뎁스~블록 0.5~1.6m 기준
LASER_MIN_AREA = 2         # 레이저 점 최소 면적
LASER_TOP_PAD = 12         # 레이저-박스 매칭 상단 마진
LASER_SIDE_PAD = 8         # 레이저-박스 매칭 좌우 마진
# 레이저(빨강) HSV 범위
LASER_LOWER1 = np.array([0, 80, 215])
LASER_UPPER1 = np.array([10, 255, 255])
LASER_LOWER2 = np.array([170, 80, 215])
LASER_UPPER2 = np.array([180, 255, 255])
# ====================


# ============================================================
# MJPEG 스트리밍 서버 (follow 로직과 무관, 독립 실행)
# 브라우저: http://192.168.0.103:8080/stream
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

def _start_stream_server(port=8080):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _MjpegHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
# ============================================================


def draw_fancy_box(img, x1, y1, x2, y2, label, color, thickness=2, filled_label=False):
    L = max(15, int((x2 - x1) * 0.18))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * L, py), color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, thickness, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ly = max(y1 - 8, th + 6)
    if filled_label:
        cv2.rectangle(img, (x1, ly - th - 6), (x1 + tw + 10, ly + 2), color, -1, cv2.LINE_AA)
        txt_color = (255, 255, 255)
    else:
        txt_color = color
    cv2.putText(img, label, (x1 + 5, ly - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)


# ===== 블록 박스 디자인 (반투명 노란 브래킷 + 검은배경 노란글씨) =====
BOX_COLOR = (255, 255, 0)      # 시안 (BGR)
BOX_ALPHA = 0.55               # 선 반투명도 (0=투명, 1=불투명)


LABEL_BG_ALPHA = 140   # 라벨 배경 검정 불투명도 (0=완전투명 ~ 255=불투명)
SEG_COLOR_NAME = (255, 255, 255)   # 클래스명 (흰색)
SEG_COLOR_CONF = (0, 230, 118)     # 확률 (초록)
SEG_COLOR_DIST = (0, 200, 255)     # 거리 (하늘색)
SEG_COLOR_SEP  = (140, 140, 140)   # 구분자 | (회색)
SEG_COLOR_STATUS = (102, 210, 130)   # FSM 상태 텍스트 (차분한 초록)
SEG_COLOR_KEY    = (150, 160, 175)   # dist/ang 라벨 (연회색)
SEG_COLOR_VAL_D  = (0, 200, 255)     # 거리 수치 (하늘색)
SEG_COLOR_VAL_A  = (255, 220, 90)    # 각도 수치 (연노랑)


def _put_label_pil(img, text, x, y, fill_color=(255, 255, 0)):
    """반투명 검은 배경 + 글씨 라벨을 PIL Noto 폰트로 그린다."""
    if LABEL_FONT is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    l, t, r, b = draw.textbbox((0, 0), text, font=LABEL_FONT)
    tw, th = r - l, b - t
    pad_x, pad_y = 8, 5
    bx1, by1 = x, y - th - pad_y * 2
    bx2, by2 = x + tw + pad_x * 2, y
    if by1 < 0:
        by1, by2 = y, y + th + pad_y * 2
    # 반투명 검은 배경 (알파 = LABEL_BG_ALPHA)
    draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, LABEL_BG_ALPHA))
    # 노란 글씨는 불투명
    draw.text((bx1 + pad_x, by1 + pad_y - t), text, font=LABEL_FONT,
              fill=(fill_color[0], fill_color[1], fill_color[2], 255))
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _put_label_seg_right_pil(img, segments, y, font=LABEL_FONT_SMALL, margin=10):
    """세그먼트 라벨을 화면 우측 정렬로 그린다 (작은 폰트). x는 우측 margin 기준 자동 계산."""
    if font is None:
        _put_label_seg_pil(img, segments, margin, y)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad_x, pad_y = 8, 5
    l0, t0, r0, b0 = draw.textbbox((0, 0), "Ay", font=font)
    th = b0 - t0
    tw = 0
    for txt, _c in segments:
        sl, st, sr, sb = draw.textbbox((0, 0), txt, font=font)
        tw += sr - sl
    W = img.shape[1]
    bx2 = W - margin
    bx1 = bx2 - (tw + pad_x * 2)
    by1 = y - th - pad_y * 2
    by2 = y
    if by1 < 0:
        by1, by2 = y, y + th + pad_y * 2
    draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, LABEL_BG_ALPHA))
    cx = bx1 + pad_x
    for txt, col in segments:
        draw.text((cx, by1 + pad_y - t0), txt, font=font,
                  fill=(col[0], col[1], col[2], 255))
        sl, st, sr, sb = draw.textbbox((0, 0), txt, font=font)
        cx += sr - sl
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _put_label_seg_pil(img, segments, x, y):
    """세그먼트별 색상 라벨. segments=[(text, (R,G,B)), ...] 순서대로 이어 그린다."""
    if LABEL_FONT is None:
        full = "".join(s[0] for s in segments)
        (tw, th), _ = cv2.getTextSize(full, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img, full, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad_x, pad_y = 8, 5
    l0, t0, r0, b0 = draw.textbbox((0, 0), "Ay", font=LABEL_FONT)
    th = b0 - t0
    tw = 0
    for txt, _c in segments:
        sl, st, sr, sb = draw.textbbox((0, 0), txt, font=LABEL_FONT)
        tw += sr - sl
    bx1, by1 = x, y - th - pad_y * 2
    bx2, by2 = x + tw + pad_x * 2, y
    if by1 < 0:
        by1, by2 = y, y + th + pad_y * 2
    draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, LABEL_BG_ALPHA))
    cx = bx1 + pad_x
    for txt, col in segments:
        draw.text((cx, by1 + pad_y - t0), txt, font=LABEL_FONT,
                  fill=(col[0], col[1], col[2], 255))
        sl, st, sr, sb = draw.textbbox((0, 0), txt, font=LABEL_FONT)
        cx += sr - sl
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_block_box(img, x1, y1, x2, y2, label):
    """반투명 노란 브래킷 박스 + PIL 라벨."""
    overlay = img.copy()
    L = max(12, int(min(x2 - x1, y2 - y1) * 0.25))
    th = 2
    c = BOX_COLOR
    cv2.line(overlay, (x1, y1), (x1 + L, y1), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x1, y1), (x1, y1 + L), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x2, y1), (x2 - L, y1), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x2, y1), (x2, y1 + L), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x1, y2), (x1 + L, y2), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x1, y2), (x1, y2 - L), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x2, y2), (x2 - L, y2), c, th, cv2.LINE_AA)
    cv2.line(overlay, (x2, y2), (x2, y2 - L), c, th, cv2.LINE_AA)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0, img)
    if isinstance(label, list):
        _put_label_seg_pil(img, label, x1, y1)
    else:
        _put_label_pil(img, label, x1, y1)


def draw_laser_marker(img, center, radius=8, label="Laser Point"):
    """레이저 지점을 반투명 노란 빈 원 + 라벨로 표시 (블록 박스와 통일 디자인)."""
    cx, cy = int(center[0]), int(center[1])
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)      # 빈 원
    cv2.circle(overlay, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)         # 중심 점
    cv2.addWeighted(overlay, BOX_ALPHA, img, 1 - BOX_ALPHA, 0, img)
    cv2.putText(img, label, (cx + radius + 4, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def find_laser_pointer(frame):
    """HSV로 빨간 레이저 포인터 중심 좌표 검출. 없으면 None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LASER_LOWER1, LASER_UPPER1)
    mask2 = cv2.inRange(hsv, LASER_LOWER2, LASER_UPPER2)
    laser_mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    laser_mask = cv2.erode(laser_mask, kernel, iterations=1)
    laser_mask = cv2.dilate(laser_mask, kernel, iterations=1)
    # 붉은 테두리 마스크 (넓게: 반사점 포함)
    red_mask = cv2.dilate(laser_mask, kernel, iterations=2)
    # 흰 코어 마스크 (저채도 S<60, 고명도 V>235) = 진짜 레이저 중심
    core_mask = cv2.inRange(hsv, np.array([0, 0, 235]), np.array([180, 60, 255]))
    core_mask = cv2.erode(core_mask, kernel, iterations=1)
    core_mask = cv2.dilate(core_mask, kernel, iterations=1)
    # 흰 코어 중 붉은 테두리에 감싸인 것만 진짜 레이저
    core_mask = cv2.bitwise_and(core_mask, red_mask)
    contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None


class FollowNode(Node):
    def __init__(self):
        super().__init__("follow_node")
        self.cmd_pub = self.create_publisher(Twist, "/target_cmd", 10)
        self.cam_health_pub = self.create_publisher(Float32, "/cam_health", 10)
        self.hb_pub = self.create_publisher(String, "/heartbeat/follow_node", 10)
        self.enable_sub = self.create_subscription(
            Bool,
            "/follow_enable", self.on_follow_enable, 10)
        self.agv_mode_sub = self.create_subscription(
            Int32, "/agv_mode", self.on_agv_mode, 10)
        self.agv_status_sub = self.create_subscription(
            Int32, "/AGV_status", self.on_agv_status, 10)
        self.pickup_status_sub = self.create_subscription(
            String, "/pickup_status", self.on_pickup_status, 10)
        self.rear_person_pub = self.create_publisher(Bool, "/rear_person", 10)
        self.guide_mode = False
        self.follow_cat = False   # 직전 프레임이 추종 카테고리(10번대)였는지
        self.pickup_cat = False   # 현재 픽업 카테고리(50번대) 여부
        self.enabled = True

        self.get_logger().info("loading YOLO...")
        self.model = YOLO(MODEL_PATH, task="detect")
        self.block_model = YOLO(BLOCK_MODEL_PATH, task="detect")
        self.get_logger().info("block model loaded")
        self.block_pub = self.create_publisher(Point, "/selected_block", 10)
        self.block_class_pub = self.create_publisher(String, "/selected_block_class", 10)
        self.laser_pub = self.create_publisher(Point, "/laser_point", 10)
        # BoxMOT 런타임 패치 (venv 재생성에도 영속)
        _bp = boxmot_patch.apply_all()
        self.get_logger().info(f"[BOXMOT-PATCH] {_bp}")
        # ===== Re-ID 트래커 (Deep OC-SORT + OSNet) =====
        _reid_w = Path(os.path.expanduser(
            "~/venv/lib/python3.10/site-packages/models/clip_msmt17.pt"))
        self.tracker = create_tracker(
            tracker_type="deepocsort",
            tracker_config=get_tracker_config("deepocsort"),
            reid_weights=_reid_w,
            device="cuda:0",
            half=True,
        )
        # Re-ID ID 유지 강화: 화면 이탈 후 재등장 대비
        self.tracker.max_age = 300   # 20Hz 기준 약 15초
        self.tracker.min_hits = 2
        self.locked_id = None
        self.target_miss = 0          # 타겟 연속 미검출 프레임 수
        self.MISS_LIMIT = 60          # 이 프레임 넘게 안 보이면 폐기(20Hz=3초)
        self.has_locked_once = False   # 최초 1회 락 이후엔 자동 락 금지(방향 B)
        self._relock_count = 0
        self._relock_cand_id = None   # 연속 재락 후보 track_id
        self._relock_streak = 0       # 연속 PASS 프레임 수
        self.RELOCK_STREAK = 5        # 이만큼 연속 PASS해야 실제 재락(순간 튐 방어)
        self._unlock_count = 0
        # ===== B: OSNet 임베딩 기반 재식별 =====
        self.reid = self.tracker.model       # get_features(xyxys, img) -> (N,512)
        self.locked_feat = None              # 저장된 타겟 임베딩 (512,)
        self.REID_SIM_THRESH = 0.28           # 코사인 유사도 재락 임계값 (CLIP 분포 기준)
        self.GUIDE_REID_THRESH = 0.28         # 안내모드 재식별 임계값 (앞모습 매칭, 저녁 튜닝)
        self.OSNET_STRONG = 0.78              # 이 이상이면 색상 게이트 면제(타겟0.77+ vs 타인0.74- 실측 경계)
        self.OSNET_MID = 0.55                 # side/dist 통과 시 color 미달이어도 본인 인정하는 중간 osnet 임계 (검은옷 구제)
        self.DIROSNET_COLOR_FLOOR = 0.30      # dir-osnet 뒷문 색상 하한: 이 미만이면 타인으로 차단(A데이터 타인 color p95=0.30)
        self.REID_EMA = 0.9                  # 임베딩 EMA 계수 (클수록 과거 유지)
        self.EMA_UPDATE_GATE = 0.5           # EMA 갱신 게이트: 현재 sim이 이 값 미만이면 갱신 스킵(오염 방지)
        # 색상 히스토그램 보조 (게이트)
        self.locked_hist = None              # 타겟 HSV 색상 지문
        self.COLOR_SIM_THRESH = 0.5          # 색상 게이트 통과 임계값 (검증 후 확정)
        self.COLOR_EMA = 0.9                 # 색상 히스토그램 EMA 계수

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        _profile = self.pipeline.get_active_profile()
        _color_stream = _profile.get_stream(rs.stream.color)
        self.color_intrin = _color_stream.as_video_stream_profile().get_intrinsics()
        self._depth_sensor = _profile.get_device().first_depth_sensor()
        for _ in range(30):
            self.pipeline.wait_for_frames()

        self.frame_count = 0
        self.started = False
        self.last_seen_side = None    # "left" / "right"
        self.lost_count = 0           # 사람 안 보인 프레임 수
        self.last_cx = None           # 마지막 타겟 x좌표
        self.last_dist = None         # 마지막 타겟 거리(m)
        # 가림 감지: 타겟을 가린 track은 재락 금지(타인 락 방지)
        self.occlusion_blacklist = set()  # 재락 금지 track_id 집합
        self.OCC_CX_MARGIN = 150      # 가림 판정 x근접(px)
        self.OCC_DIST_MARGIN = 1.0    # 가림 판정 거리근접(m)
        self.SIDE_MARGIN = 80         # 진입 방향 판정 여유(px)
        self.DIST_GATE = 0.8          # 재식별 허용 거리차(m) - 가림-진입 타인 방어
        self.searching = False        # 탐색 중 여부
        self.find_until = 0.0         # find(정지) 유지 종료 시각
        self.display_class = None     # 레이저로 지정한 타겟 클래스명 (재탐색 표시용)
        self.show_target_box = False  # AGV_status 1 이후 타겟 박스 표시 모드
        # ===== 팔 작업(status 1) 구간 박스 안정성 게이팅 =====
        self.arm_phase = False        # status 1 이후 = 팔 작업 구간
        self.box_stable_buf = []      # 최근 (bcx, bcy, bw, bh) 이력
        self.BOX_STABLE_WIN = 7       # 판정 윈도우 프레임 수
        self.BOX_CENTER_THRESH = 2.0  # 중심 이동 허용치(px)
        self.BOX_SIZE_THRESH = 3.0    # 폭/높이 변동 허용치(px)
        self._box_wait_log_n = 0
        self._box_ok_log_n = 0
        self.box_last_time = 0.0      # 마지막 박스 관측 시각
        self.BOX_GAP_SEC = 0.3        # 이 시간 이상 끊기면 버퍼 리셋
        self.pickup_status_msg = None   # /pickup_status 최근 상태 (영상 표시용)
        self.pickup_status_until = 0.0  # 상태 표시 유지 종료 시각
        self.PICKUP_STATUS_HOLD = 2.0   # 상태 표시 유지 시간(초)
        self._cut_log_n = 0             # CUT 로그 스로틀 카운터
        self.cut_show_until = 0.0       # CUT 화면 표시 유지 종료 시각

        _start_stream_server(8080)
        self.get_logger().info("MJPEG stream: http://192.168.0.103:8080/stream")
        self.get_logger().info("follow_node started. target = nearest person")

        # ===== Re-ID 캘리브레이션 로거 (REID_CALIB 환경변수) =====
        _ctag = reid_calib.tag()
        self._calib = reid_calib.CalibLogger(_ctag) if _ctag else None
        if self._calib is not None:
            self.get_logger().warn(
                f"[CALIB] Re-ID 캘리브레이션 로깅 ON -> {self._calib.path}")
        self.timer = self.create_timer(1.0 / 20.0, self.loop)
        self._fps_last_count = 0
        self._fps_last_time = time.time()
        self._infer_ms_sum = 0.0
        self._infer_ms_n = 0
        self.fps_timer = self.create_timer(5.0, self.publish_cam_health)

    def publish_cam_health(self):
        now = time.time()
        elapsed = now - self._fps_last_time
        if elapsed <= 0:
            return
        fps = (self.frame_count - self._fps_last_count) / elapsed
        self._fps_last_count = self.frame_count
        self._fps_last_time = now
        msg = Float32()
        msg.data = float(fps)
        self.cam_health_pub.publish(msg)
        _temp = -1.0
        try:
            if self._depth_sensor.supports(rs.option.asic_temperature):
                _temp = float(self._depth_sensor.get_option(rs.option.asic_temperature))
        except Exception:
            _temp = -1.0
        hb = String()
        hb.data = json.dumps({
            "stamp": now,
            "node": "follow_node",
            "cam_temp": round(_temp, 1),
            "fps": round(fps, 1),
            "status": "OK" if fps >= 8.0 else "LOW",
        })
        self.hb_pub.publish(hb)
        if self._infer_ms_n > 0:
            avg_ms = self._infer_ms_sum / self._infer_ms_n
            infer_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            self.get_logger().info(f"\033[90m📊 [PERF] loop_fps={fps:.1f} infer_latency={avg_ms:.1f}ms (n={self._infer_ms_n})\033[0m")
            self._infer_ms_sum = 0.0
            self._infer_ms_n = 0

    def median_depth(self, depth_frame, cx, cy, k=5):
        half = k // 2
        cx = int(min(max(cx, half), IMG_WIDTH - 1 - half))
        cy = int(min(max(cy, half), 480 - 1 - half))
        vals = []
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                d = depth_frame.get_distance(cx + dx, cy + dy)
                if d > 0:
                    vals.append(d)
        return float(np.median(vals)) if vals else 0.0

    def refined_depth(self, depth_frame, cx, cy, k=7, band=0.004):
        """중앙값으로 이상치를 거른 뒤 근방 샘플만 평균 -> 1mm 양자화 완화."""
        half = k // 2
        cx = int(min(max(cx, half), IMG_WIDTH - 1 - half))
        cy = int(min(max(cy, half), 480 - 1 - half))
        vals = []
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                d = depth_frame.get_distance(cx + dx, cy + dy)
                if d > 0:
                    vals.append(d)
        if not vals:
            return 0.0
        med = float(np.median(vals))
        inliers = [v for v in vals if abs(v - med) <= band]
        if not inliers:
            return med
        return float(np.mean(inliers))

    def process_block_selection(self, img, depth_frame, annotated):
        """레이저가 있을 때만 block predict. 선택 블록의 X,Y,Z(m) 발행 + 오버레이."""
        if not self.pickup_cat:
            return
        laser_pt = find_laser_pointer(img)
        if laser_pt is None:
            return
        draw_laser_marker(annotated, laser_pt)
        # 레이저 지점 3D 좌표 발행 (50 유도: pickup_node -> /target_cmd)
        _ld = self.median_depth(depth_frame, laser_pt[0], laser_pt[1])
        if _ld > 0:
            _lp3d = rs.rs2_deproject_pixel_to_point(
                self.color_intrin, [laser_pt[0], laser_pt[1]], _ld)
            _lmsg = Point()
            _lmsg.x = float(_lp3d[0])
            _lmsg.y = float(_lp3d[1])
            _lmsg.z = float(_lp3d[2])
            self.laser_pub.publish(_lmsg)
        # 레이저 지점 기준 ROI crop -> 배경 오탐 차단
        h, w = img.shape[:2]
        rx1 = max(0, laser_pt[0] - ROI_HALF)
        ry1 = max(0, laser_pt[1] - ROI_HALF * 3 // 2)
        rx2 = min(w, laser_pt[0] + ROI_HALF)
        ry2 = min(h, laser_pt[1] + ROI_HALF // 2)
        # ROI 시각화 (흰색 얇은 사각형)
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1, cv2.LINE_AA)
        roi_img = img[ry1:ry2, rx1:rx2].copy()
        results = self.block_model.predict(roi_img, conf=BLOCK_CONF, verbose=False)
        sel_name = "None"
        sel_box = None
        # 레이저 지점을 ROI 로컬 좌표로 변환
        laser_local = (laser_pt[0] - rx1, laser_pt[1] - ry1)
        for result in results:
            for box in result.boxes:
                lx1, ly1, lx2, ly2 = map(int, box.xyxy[0])
                cid = int(box.cls[0])
                px1 = lx1 - LASER_SIDE_PAD
                py1 = ly1 - LASER_TOP_PAD
                px2 = lx2 + LASER_SIDE_PAD
                py2 = ly2 + LASER_SIDE_PAD
                if px1 <= laser_local[0] <= px2 and py1 <= laser_local[1] <= py2:
                    sel_name = self.block_model.names[cid]
                    sel_conf = float(box.conf[0])
                    # ROI 로컬 -> 원본 좌표 복원 (오프셋 더하기)
                    sel_box = (lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1)
                    self.get_logger().info(
                        f"\033[92m🔍 [MATCH] {sel_name} conf={float(box.conf[0]):.3f}\033[0m")
                    break
            if sel_box is not None:
                break
        if sel_box is None:
            return
        x1, y1, x2, y2 = sel_box
        # 블록은 정사각형 -> 화면 경계에 잘리면 온전한 축 길이로 중심 복원
        _h, _w = img.shape[:2]
        _EDGE = 3
        bw, bh = (x2 - x1), (y2 - y1)
        _cut_top = (y1 <= _EDGE)
        _cut_bot = (y2 >= _h - _EDGE)
        _cut_left = (x1 <= _EDGE)
        _cut_right = (x2 >= _w - _EDGE)
        if _cut_bot and not _cut_top:
            bcy = y1 + bw // 2
        elif _cut_top and not _cut_bot:
            bcy = y2 - bw // 2
        else:
            bcy = (y1 + y2) // 2
        if _cut_right and not _cut_left:
            bcx = x1 + bh // 2
        elif _cut_left and not _cut_right:
            bcx = x2 - bh // 2
        else:
            bcx = (x1 + x2) // 2
        # 복원 중앙이 화면 밖이면 절반 이상 잘린 것 -> 발행 스킵
        if not (0 <= bcx < _w and 0 <= bcy < _h):
            self._cut_log_n += 1
            if self._cut_log_n % 100 == 1:
                self.get_logger().info(
                    "\033[93m🚫 [CUT] 복원 중앙 화면 밖(절반 이상 잘림) -> 스킵\033[0m")
            self.cut_show_until = time.time() + 0.5
            return
        dist = self.median_depth(depth_frame, bcx, bcy)
        if dist <= 0 or dist > BLOCK_MAX_Z:
            self.get_logger().info(f"\033[93m🚫 [GATE] dist={dist:.3f} 차단(max={BLOCK_MAX_Z})\033[0m")
            return
        pt3d = rs.rs2_deproject_pixel_to_point(self.color_intrin, [bcx, bcy], dist)
        msg = Point()
        msg.x = float(pt3d[0])
        msg.y = float(pt3d[1])
        msg.z = float(pt3d[2])
        cmsg = String()
        cmsg.data = sel_name
        self.block_class_pub.publish(cmsg)
        self.display_class = sel_name   # 재탐색 표시용 타겟 클래스 기억
        self.block_pub.publish(msg)
        _seg_label = [
            (sel_name, SEG_COLOR_NAME),
            (" | ", SEG_COLOR_SEP),
            (f"{int(sel_conf * 100)}%", SEG_COLOR_CONF),
            (" | ", SEG_COLOR_SEP),
            (f"{dist:.2f}m", SEG_COLOR_DIST),
        ]
        draw_block_box(annotated, x1, y1, x2, y2, _seg_label)
        self.get_logger().info(
            f"\033[96m📦 [PUB] {sel_name} XYZ=({pt3d[0]:.3f},{pt3d[1]:.3f},{pt3d[2]:.3f})\033[0m")

    def draw_target_box(self, img, depth_frame, annotated):
        """재탐색 표시 모드: display_class와 같은 클래스 블록을 YOLO 스타일 박스로 표시."""
        if not self.show_target_box or self.display_class is None:
            return
        results = self.block_model.predict(img, conf=BLOCK_CONF, verbose=False)
        for result in results:
            for box in result.boxes:
                cid = int(box.cls[0])
                name = self.block_model.names[cid]
                if name != self.display_class:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # 블록은 정사각형 -> 화면 경계에 잘리면 온전한 축 길이로 중심 복원
                _h, _w = img.shape[:2]
                _EDGE = 3
                bw, bh = (x2 - x1), (y2 - y1)
                _cut_top = (y1 <= _EDGE)
                _cut_bot = (y2 >= _h - _EDGE)
                _cut_left = (x1 <= _EDGE)
                _cut_right = (x2 >= _w - _EDGE)
                # 세로 잘림: 가로(bw)를 한 변으로 보고 안 잘린 변부터 정사각형 복원
                if _cut_bot and not _cut_top:
                    bcy = y1 + bw // 2
                elif _cut_top and not _cut_bot:
                    bcy = y2 - bw // 2
                else:
                    bcy = (y1 + y2) // 2
                # 가로 잘림: 세로(bh)를 한 변으로 보고 복원
                if _cut_right and not _cut_left:
                    bcx = x1 + bh // 2
                elif _cut_left and not _cut_right:
                    bcx = x2 - bh // 2
                else:
                    bcx = (x1 + x2) // 2
                # 복원 중앙이 화면 밖이면 절반 이상 잘린 것 -> 발행 스킵, 다음 프레임 대기
                if not (0 <= bcx < _w and 0 <= bcy < _h):
                    self._cut_log_n += 1
                    if self._cut_log_n % 100 == 1:
                        self.get_logger().info(
                            "\033[93m🚫 [CUT] 복원 중앙 화면 밖(절반 이상 잘림) -> 스킵\033[0m")
                    self.cut_show_until = time.time() + 0.5
                    continue
                dist = self.refined_depth(depth_frame, bcx, bcy)
                if dist > 0:
                    label = [
                        (name, SEG_COLOR_NAME),
                        (" | ", SEG_COLOR_SEP),
                        (f"{int(conf * 100)}%", SEG_COLOR_CONF),
                        (" | ", SEG_COLOR_SEP),
                        (f"{dist:.2f}m", SEG_COLOR_DIST),
                    ]
                else:
                    label = [
                        (name, SEG_COLOR_NAME),
                        (" | ", SEG_COLOR_SEP),
                        (f"{int(conf * 100)}%", SEG_COLOR_CONF),
                    ]
                draw_block_box(annotated, x1, y1, x2, y2, label)
                # 재탐색: 찾은 블록의 뎁스 원본 XYZ + 클래스 발행 (pickup_node용)
                if dist > 0:
                    # 팔 작업 구간: 바운딩박스가 고정됐을 때만 발행
                    if self.arm_phase and not self._box_is_stable(
                            bcx, bcy, x2 - x1, y2 - y1):
                        return
                    pt3d = rs.rs2_deproject_pixel_to_point(
                        self.color_intrin, [bcx, bcy], dist)
                    msg = Point()
                    msg.x = float(pt3d[0])
                    msg.y = float(pt3d[1])
                    msg.z = float(pt3d[2])
                    cmsg = String()
                    cmsg.data = name
                    self.block_class_pub.publish(cmsg)
                    self.block_pub.publish(msg)
                return

    def _box_is_stable(self, bcx, bcy, bw, bh):
        """최근 BOX_STABLE_WIN 프레임의 박스 중심/크기 변동이 임계 이하면 True."""
        _now = time.time()
        # 검출이 끊겼다 재등장하면 불연속 좌표 -> 버퍼 초기화
        if _now - self.box_last_time > self.BOX_GAP_SEC:
            self.box_stable_buf = []
        self.box_last_time = _now
        self.box_stable_buf.append((float(bcx), float(bcy), float(bw), float(bh)))
        if len(self.box_stable_buf) > self.BOX_STABLE_WIN:
            self.box_stable_buf.pop(0)
        if len(self.box_stable_buf) < self.BOX_STABLE_WIN:
            return False
        cxs = [b[0] for b in self.box_stable_buf]
        cys = [b[1] for b in self.box_stable_buf]
        bws = [b[2] for b in self.box_stable_buf]
        bhs = [b[3] for b in self.box_stable_buf]
        d_cx = max(cxs) - min(cxs)
        d_cy = max(cys) - min(cys)
        d_bw = max(bws) - min(bws)
        d_bh = max(bhs) - min(bhs)
        ok = (d_cx <= self.BOX_CENTER_THRESH and d_cy <= self.BOX_CENTER_THRESH
              and d_bw <= self.BOX_SIZE_THRESH and d_bh <= self.BOX_SIZE_THRESH)
        if not ok:
            self._box_wait_log_n += 1
            if self._box_wait_log_n % 20 == 1:
                self.get_logger().info(
                    f"\033[93m⏳ [BOX] 박스 흔들림 -> 발행 대기 "
                    f"(dcx={d_cx:.1f} dcy={d_cy:.1f} dw={d_bw:.1f} dh={d_bh:.1f})\033[0m")
        else:
            self._box_wait_log_n = 0
            self._box_ok_log_n += 1
            if self._box_ok_log_n % 40 == 1:
                self.get_logger().info(
                    f"\033[92m✅ [BOX] 안정 -> 발행 허용 "
                    f"(dcx={d_cx:.1f} dcy={d_cy:.1f} dw={d_bw:.1f} dh={d_bh:.1f})\033[0m")
        return ok

    def on_pickup_status(self, msg):
        # pickup_node 상태 수신 -> 영상 표시용 저장 (표시 유지시간 갱신)
        self.pickup_status_msg = msg.data
        self.pickup_status_until = time.time() + self.PICKUP_STATUS_HOLD
        # 재탐색 종료(DONE) -> 팔 작업 구간 해제 (전체프레임 추론 중단)
        if msg.data == "DONE":
            self.arm_phase = False
            self.show_target_box = False
            self.box_stable_buf = []
            self.get_logger().info("\033[90m[ARM] 재탐색 종료 -> 타겟 박스 추론 중단\033[0m")
        # 재파지 시작(REGRIP) -> 다시 탐지 재개
        elif msg.data == "REGRIP":
            self.arm_phase = True
            self.box_stable_buf = []
            if self.display_class is not None:
                self.show_target_box = True
            self.get_logger().info("\033[95m[ARM] 재파지 -> 타겟 박스 추론 재개\033[0m")

    def on_agv_status(self, msg):
        # 재도킹 좌표요청(4)/밀착 완료(1) -> 타겟 블록 박스 표시 모드 ON
        if msg.data in (1, 4):
            self.arm_phase = (msg.data == 1)
            self.box_stable_buf = []
            if self.display_class is not None:
                self.show_target_box = True
                self.get_logger().info(
                    f"\033[95m🎯 [SHOW] 타겟 박스 표시 ON (class={self.display_class})\033[0m")

    def on_agv_mode(self, msg):
        # 추종(10번대) 새로 진입하는 엣지에서 락 상태 리셋 -> 새 타겟 재락 허용
        new_follow = (msg.data // 10 == 1)
        if new_follow and not self.follow_cat:
            self.locked_id = None
            self.locked_feat = None
            self.occlusion_blacklist = set()
            self.locked_hist = None
            self.has_locked_once = False
            self.target_miss = 0
            self.get_logger().info("[MODE] 추종 재진입 -> 락 상태 리셋")
        self.follow_cat = new_follow
        new_guide = (msg.data // 10 == 2)
        if new_guide != self.guide_mode:
            self.guide_mode = new_guide
            if new_guide:
                # 안내 진입: 추종 타겟 버리고 안내 타겟 새로 락
                self.locked_id = None
                self.locked_feat = None
                self.locked_hist = None
                self.occlusion_blacklist = set()
                self.has_locked_once = False
                self.target_miss = 0
                self.get_logger().info("[MODE] 안내모드 -> 후방 감시 (락 초기화)")
                self.stop_cmd()
            else:
                self.get_logger().info("[MODE] 안내모드 해제")
        # 픽업(50번대) 이탈 시 타겟 박스 표시 OFF
        self.pickup_cat = (msg.data // 10 == 5)
        if msg.data // 10 != 5:
            self.show_target_box = False
            self.arm_phase = False
            self.box_stable_buf = []

    def on_follow_enable(self, msg):
        self.enabled = bool(msg.data)
        self.get_logger().info(f"[ENABLE] follow_enable = {self.enabled}")
        if not self.enabled:
            self.stop_cmd()

    def _extract_hist(self, xyxy, img):
        # 박스 안쪽(배경 제외) HSV 색상 히스토그램 (정규화)
        x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
        h, w = img.shape[:2]
        # 가장자리 15% 잘라 배경/팔 영향 줄임
        mx = int((x2 - x1) * 0.15)
        my = int((y2 - y1) * 0.15)
        x1, y1 = max(0, x1 + mx), max(0, y1 + my)
        x2, y2 = min(w, x2 - mx), min(h, y2 - my)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # H, S 2D 히스토그램 (V는 조명 영향 커서 제외)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    def _update_locked_hist(self, hist):
        if hist is None:
            return
        if self.locked_hist is None:
            self.locked_hist = hist
        else:
            self.locked_hist = (self.COLOR_EMA * self.locked_hist
                                + (1.0 - self.COLOR_EMA) * hist)
    def _color_sim(self, xyxy, img):
        # 후보 박스와 locked_hist의 색상 유사도 (-1~1, 상관계수)
        if self.locked_hist is None:
            return 1.0  # 색상 기준 없으면 게이트 통과
        h = self._extract_hist(xyxy, img)
        if h is None:
            return 0.0
        return float(cv2.compareHist(
            self.locked_hist.astype("float32"),
            h.astype("float32"), cv2.HISTCMP_CORREL))
    def _extract_feat(self, xyxy, img):
        arr = np.array([xyxy[:4]], dtype=np.float32)
        feats = self.reid.get_features(arr, img)
        if feats is None or len(feats) == 0:
            return None
        f = feats[0].astype(np.float32)
        n = np.linalg.norm(f)
        return f / n if n > 0 else None
    def _extract_feats_batch(self, xyxy_list, img):
        # 여러 박스를 한 번의 get_features 호출로 배치 추론 (프레임당 OSNet forward 1회)
        if not xyxy_list:
            return []
        arr = np.array([b[:4] for b in xyxy_list], dtype=np.float32)
        feats = self.reid.get_features(arr, img)
        if feats is None or len(feats) == 0:
            return [None] * len(xyxy_list)
        out = []
        for i in range(len(xyxy_list)):
            if i >= len(feats):
                out.append(None)
                continue
            f = feats[i].astype(np.float32)
            n = np.linalg.norm(f)
            out.append(f / n if n > 0 else None)
        return out
    def _update_locked_feat(self, feat):
        if feat is None:
            return
        if self.locked_feat is None:
            self.locked_feat = feat
        else:
            # 오염 방지: 현재 임베딩이 기존 locked_feat와 충분히 유사할 때만 갱신
            _sim = float(np.dot(self.locked_feat, feat))
            if _sim < self.EMA_UPDATE_GATE:
                return  # 타인 의심 -> EMA 오염 차단(갱신 스킵)
            self.locked_feat = (self.REID_EMA * self.locked_feat
                                + (1.0 - self.REID_EMA) * feat)
            n = np.linalg.norm(self.locked_feat)
            if n > 0:
                self.locked_feat /= n
    def _gate_decide(self, sim, csim, side_block, dist_block):
        # _best_match_id / _calib_log 공용 게이트 판정
        # 공간 게이트(side/dist)를 최우선: osnet이 높아도 위치가 어긋나면 타인
        # (가림-진입 방어: 타겟 앞을 지나는 타인은 거리/방향이 달라 차단)
        if side_block:
            return "BLOCK(side)"
        if dist_block:
            return "BLOCK(dist)"
        if sim >= self.OSNET_STRONG:
            return "PASS(osnet-strong)"
        if csim >= self.COLOR_SIM_THRESH:
            return "PASS"
        return "BLOCK"
    def _calib_log(self, people, img):
        # 캘리브레이션 전용: 락 여부와 무관하게 전 후보 sim/color 기록
        if self._calib is None or self.locked_feat is None or not people:
            return
        _fl = self._extract_feats_batch(
            [(p[3], p[4], p[5], p[6]) for p in people], img)
        phase = "UNLOCKED" if self.locked_id is None else "LOCKED"
        now = time.time()
        for _i, p in enumerate(people):
            f = _fl[_i]
            if f is None:
                continue
            tid = p[7]
            p_dist, p_cx = p[0], p[1]
            sim = float(np.dot(self.locked_feat, f))
            csim = self._color_sim((p[3], p[4], p[5], p[6]), img)
            side_block = False
            if self.last_seen_side == "left" and p_cx > CENTER_X + self.SIDE_MARGIN:
                side_block = True
            elif self.last_seen_side == "right" and p_cx < CENTER_X - self.SIDE_MARGIN:
                side_block = True
            dist_block = False
            if self.last_dist is not None and p_dist > 0:
                if abs(p_dist - self.last_dist) > self.DIST_GATE:
                    dist_block = True
            self._calib.log(
                ts=f"{now:.3f}", frame=self.frame_count, phase=phase,
                n_people=len(people),
                locked_id=(self.locked_id if self.locked_id is not None else -1),
                tid=tid,
                is_target=int(tid == self.locked_id),
                osnet=f"{sim:.4f}", color=f"{csim:.4f}",
                dist=f"{p_dist:.3f}", cx=p_cx,
                last_dist=(f"{self.last_dist:.3f}"
                           if self.last_dist is not None else ""),
                last_seen_side=(self.last_seen_side or ""),
                side_block=int(side_block), dist_block=int(dist_block),
                gate=self._gate_decide(sim, csim, side_block, dist_block))
    def _handle_guide(self, people, img, depth):
        """안내모드 후방 처리: 손님 최초 락 + 재식별 -> /rear_person 발행 + 후방 오버레이."""
        rear = img.copy()
        target = None
        if people:
            people.sort(key=lambda p: p[0])  # 거리 오름차순
            if self.locked_feat is None:
                # 최초 락: 가장 가까운 사람 = 손님
                target = people[0]
                self.locked_id = target[7]
                f = self._extract_feat(target[3:7], img)
                self._update_locked_feat(f)
                self._update_locked_hist(self._extract_hist(target[3:7], img))
                self.has_locked_once = True
                self.get_logger().info(
                    f"\033[96m[GUIDE-LOCK] 손님 락 id={self.locked_id}\033[0m")
            else:
                # 재식별
                tid, sim = self._guide_match(people, img)
                if tid is not None:
                    self.locked_id = tid
                    target = [p for p in people if p[7] == tid][0]
                    # 임베딩/색상 EMA 갱신 (오염 방지 게이트 내장)
                    self._update_locked_feat(self._extract_feat(target[3:7], img))
                    self._update_locked_hist(self._extract_hist(target[3:7], img))
        target_found = target is not None
        rmsg = Bool()
        rmsg.data = bool(target_found)
        self.rear_person_pub.publish(rmsg)
        # 후방 화면 오버레이
        for _p in people:
            _d, _, _, _x1, _y1, _x2, _y2, _tid, _cf = _p
            _col = (0, 255, 0) if (target_found and _tid == target[7]) else (150, 150, 150)
            cv2.rectangle(rear, (_x1, _y1), (_x2, _y2), _col, 2)
        rstat = "GUIDE: CUSTOMER" if target_found else "GUIDE: WAITING"
        rcol = (0, 255, 0) if target_found else (0, 0, 255)
        cv2.putText(rear, rstat, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rcol, 2)
        with _stream_lock:
            global _stream_frame
            _stream_frame = rear.copy()

    def _guide_match(self, people, img):
        """안내 전용 재식별: side/dist 게이트 없이 CLIP 유사도(+색상)만.
        안내는 손님이 좌우/거리 자유롭게 움직이므로 방향/거리 게이트를 쓰지 않는다.
        반환: (best_tid, best_sim) 또는 (None, best_sim)."""
        if self.locked_feat is None:
            return None, 0.0
        _fl = self._extract_feats_batch(
            [(p[3], p[4], p[5], p[6]) for p in people], img)
        best_tid, best_sim = None, -1.0
        for _i, p in enumerate(people):
            f = _fl[_i]
            if f is None:
                continue
            tid = p[7]
            if tid in self.occlusion_blacklist:
                continue  # 가린 track 재락 금지
            sim = float(np.dot(self.locked_feat, f))
            csim = self._color_sim((p[3], p[4], p[5], p[6]), img)
            # 게이트: CLIP 강함이면 통과, 아니면 색상 확인
            if sim >= self.OSNET_STRONG:
                gate = "PASS(clip-strong)"
            elif csim >= self.COLOR_SIM_THRESH:
                gate = "PASS(color)"
            else:
                gate = "BLOCK"
            _col = "\033[91m" if gate.startswith("BLOCK") else "\033[92m"
            self.get_logger().info(
                f"{_col}\U0001f9ed [GUIDE-CAND] id={tid} clip={sim:.2f} "
                f"color={csim:.2f} {gate}\033[0m")
            if gate.startswith("BLOCK"):
                continue
            if sim > best_sim:
                best_sim, best_tid = sim, tid
        if best_tid is not None and best_sim >= self.GUIDE_REID_THRESH:
            return best_tid, best_sim
        return None, best_sim

    def _best_match_id(self, people, img):
        if self.locked_feat is None:
            return None, 0.0
        best_tid, best_sim = None, -1.0
        # 후보 전체 임베딩을 1회 배치 추출 (프레임당 OSNet forward 1회로 축소)
        _feat_list = self._extract_feats_batch(
            [(p[3], p[4], p[5], p[6]) for p in people], img)
        for _pi, p in enumerate(people):
            x1, y1, x2, y2, tid = p[3], p[4], p[5], p[6], p[7]
            p_dist, p_cx = p[0], p[1]
            f = _feat_list[_pi]
            if f is None:
                continue
            sim = float(np.dot(self.locked_feat, f))
            # 블랙리스트 track이라도 osnet-strong이면 타겟 본인 복귀로 보고 조기 해제
            # (카메라 팬/짧은 가림 후 타겟이 같은 track_id로 재등장하는 경우 방어)
            if tid in self.occlusion_blacklist:
                if sim >= self.OSNET_STRONG:
                    self.occlusion_blacklist.discard(tid)
                    self.get_logger().info(
                        f"\033[92m[OCC-RELEASE] id={tid} osnet={sim:.2f} "
                        f">= strong -> 블랙리스트 조기해제\033[0m")
                else:
                    continue  # 타인 추정 -> 재락 금지 유지
            # ===== 방향 게이트: 나간 쪽 반대에서 등장하면 타인 =====
            side_block = False
            if self.last_seen_side == "left" and p_cx > CENTER_X + self.SIDE_MARGIN:
                side_block = True
            elif self.last_seen_side == "right" and p_cx < CENTER_X - self.SIDE_MARGIN:
                side_block = True
            # ===== 거리 게이트: 마지막 거리와 크게 다르면 타인 =====
            dist_block = False
            if self.last_dist is not None and p_dist > 0:
                if abs(p_dist - self.last_dist) > self.DIST_GATE:
                    dist_block = True
            # 공간 게이트(side/dist) 확정 시 색상 히스토그램 연산 생략(부하 절감)
            # side/dist_block이면 _gate_decide가 osnet 무관하게 BLOCK 확정 -> 색상 불필요
            if side_block or dist_block:
                csim = 0.0  # 공간 탈락 확정 -> 색상 계산 생략
            else:
                csim = self._color_sim((x1, y1, x2, y2), img)
            # 색상 게이트: 색이 확실히 다르면 후보 탈락 (다른 사람 거부)
            # 방향/거리 게이트가 걸리면 osnet 높아도 무조건 차단 (옷색 같은 타인 거부)
            gate = self._gate_decide(sim, csim, side_block, dist_block)
            _cand_col = "\033[91m" if gate.startswith("BLOCK") else "\033[92m"
            self.get_logger().info(
                f"{_cand_col}👤 [CAND] id={tid} osnet={sim:.2f} color={csim:.2f} {gate}\033[0m")
            if gate.startswith("BLOCK"):
                continue
            if sim > best_sim:
                best_sim, best_tid = sim, tid
        if best_tid is not None and best_sim >= self.REID_SIM_THRESH:
            return best_tid, best_sim
        return None, best_sim
    def pixel_to_angle(self, cx):
        cx_err = cx - CENTER_X
        return -(cx_err / IMG_WIDTH) * math.radians(HFOV_DEG)

    def publish(self, mode, distance=0.0, angle=0.0, force=False):
        # linear.z=모드, linear.x=거리, angular.z=각도
        # enabled=False면 발행 안 함 (단 force=True는 정지신호용으로 통과)
        if (not self.enabled or self.pickup_cat) and not force:
            return
        msg = Twist()
        msg.linear.x = float(distance)
        msg.linear.z = float(mode)
        msg.angular.z = float(angle)
        self.cmd_pub.publish(msg)

    def loop(self):
        global _stream_frame
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return

        img = np.asanyarray(color.get_data())
        if not (self.follow_cat or self.guide_mode or self.pickup_cat):
            with _stream_lock:
                _stream_frame = img.copy()
            self.frame_count += 1
            return
        _t0 = time.perf_counter()
        results = self.model.predict(
            img, classes=[0], verbose=False)
        self._infer_ms_sum += (time.perf_counter() - _t0) * 1000.0
        self._infer_ms_n += 1
        boxes = results[0].boxes

        # ===== Re-ID 트래커 update → 검출에 track_id 부여 =====
        dets = np.empty((0, 6), dtype=np.float32)
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, conf, cls]).astype(np.float32)
        # Re-ID(tracker)는 추종(10번대)·안내(20번대) 모드에서만 실행
        if self.follow_cat or self.guide_mode:
            tracks = self.tracker.update(dets, img)  # (N, 8): xyxy, id, conf, cls, idx
        else:
            tracks = np.empty((0, 8), dtype=np.float32)
        annotated = img.copy()
        self.process_block_selection(img, depth, annotated)
        self.draw_target_box(img, depth, annotated)
        self._draw_pickup_status(annotated)
        self.frame_count += 1

        # 워밍업
        if not self.started:
            if self.frame_count >= LOCK_FRAMES:
                self.started = True
                self.get_logger().info("[START] 추종 시작")
            if SHOW_WINDOW:
                cv2.putText(annotated, f"WARMING UP {self.frame_count}/{LOCK_FRAMES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            with _stream_lock:
                _stream_frame = annotated.copy()
            if SHOW_WINDOW:
                cv2.imshow("follow", annotated)
                cv2.waitKey(1)
            return

        # 유효한 사람 목록 (depth > 0) — tracks 기반, track_id 포함
        people = []
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            conf = float(t[5]) if len(t) > 5 else 0.0
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dist = self.median_depth(depth, cx, cy)
            if dist > 0:
                people.append((dist, cx, cy, x1, y1, x2, y2, tid, conf))

        # 가림 블랙리스트 해제: 화면에서 사라진 track은 풀어줌
        if self.occlusion_blacklist:
            _alive = {p[7] for p in people}
            _gone = self.occlusion_blacklist - _alive
            if _gone:
                self.occlusion_blacklist -= _gone
                self.get_logger().info(
                    f"[OCC-CLEAR] 가림 track 화면이탈 -> 재락금지 해제 {_gone}")
        # ===== 안내모드: 후방 손님 재식별 =====
        if self.guide_mode:
            self._handle_guide(people, img, depth)
            return
        self._calib_log(people, img)
        distance, angle = 0.0, 0.0
        status = "SEARCHING"
        now = time.time()

        # ===== find(재발견 정지) 유지 중 =====
        if self.find_until > 0.0:
            if now < self.find_until:
                status = "FIND_HOLD"
                self.publish(MODE_FIND)
                if people:
                    people.sort(key=lambda p: p[0])
                    d, cx, cy, x1, y1, x2, y2, _tid, _cf = people[0]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
                self._draw_and_stream(annotated, status, 0.0, 0.0)
                return
            else:
                self.find_until = 0.0
                if people:
                    self.searching = False
                else:
                    self.searching = True

        if people:
            # 디버그: 검출된 모든 사람 회색 박스 + ID (타겟은 아래서 빨강으로 덮어씀)
            for _p in people:
                _d, _, _, _gx1, _gy1, _gx2, _gy2, _gtid, _gcf = _p
                _label = f"ID:{_gtid}  conf:{_gcf:.2f}  {_d:.2f}m"
                draw_fancy_box(annotated, _gx1, _gy1, _gx2, _gy2, _label, (180, 180, 180), thickness=2)
            # ===== 추종 (Re-ID 자동 락) =====
            people.sort(key=lambda p: p[0])
            target = None  # RELOCK-WAIT 등 미확정 시 기본값(UnboundLocal 방지)
            if self.locked_id is None:
                # 언락 상태: 먼저 임베딩으로 기존 타겟 재식별 시도
                rematch_tid, sim = self._best_match_id(people, img)
                if rematch_tid is not None:
                    # 연속 프레임 재락: 같은 후보가 RELOCK_STREAK회 연속 PASS해야 실제 락
                    if rematch_tid == self._relock_cand_id:
                        self._relock_streak += 1
                    else:
                        self._relock_cand_id = rematch_tid
                        self._relock_streak = 1
                    if self._relock_streak >= self.RELOCK_STREAK:
                        self.locked_id = rematch_tid
                        target = [p for p in people if p[7] == rematch_tid][0]
                        self.get_logger().info(
                            f"[RELOCK] id={self.locked_id} sim={sim:.2f} "
                            f"(streak={self._relock_streak})")
                        self._relock_count += 1
                        self._relock_cand_id = None
                        self._relock_streak = 0
                    else:
                        self.get_logger().info(
                            f"[RELOCK-WAIT] id={rematch_tid} sim={sim:.2f} "
                            f"streak={self._relock_streak}/{self.RELOCK_STREAK}")
                else:
                    # 재식별 실패: 최초 1회만 새 타겟 락, 이후엔 새 사람 안 잡고 대기(방향 B)
                    if not self.has_locked_once:
                        target = people[0]
                        self.locked_id = target[7]
                        self.locked_feat = None
                        self.locked_hist = None
                        self.has_locked_once = True
                        self.get_logger().info(f"[LOCK] target id={self.locked_id}")
                        f = self._extract_feat(target[3:7], img)
                        self._update_locked_feat(f)
                        self._update_locked_hist(self._extract_hist(target[3:7], img))
                    else:
                        target = None
                        self.get_logger().info("[WAIT] 타겟 재식별 대기 (새 사람 무시)")
            else:
                # 락: locked_id와 같은 사람만 선택
                matched = [p for p in people if p[7] == self.locked_id]
                if matched:
                    self.target_miss = 0
                    target = matched[0]
                    # 타겟 보이는 동안 임베딩 EMA 갱신 (부하: 5프레임마다만 CLIP forward)
                    if self.frame_count % 5 == 0:
                        f = self._extract_feat(target[3:7], img)
                        self._update_locked_feat(f)
                        self._update_locked_hist(self._extract_hist(target[3:7], img))
                else:
                    # 락된 사람 안 보임 → people 비우고 lost/search로
                    people = []
                    target = None
                    # 락 ID가 현재 tracks에 아예 없으면(폐기됨) 락 해제 → 재락 허용
                    # 락 ID가 tracks에서 빠져도 바로 폐기하지 않고 버퍼(가림 대응)
                    # 락된 track_id가 트래커에서 사라지면 즉시 id만 해제(재식별 경로 개방).
                    # locked_feat(임베딩 지문)은 유지 -> 재등장 시 같은 사람으로 복귀 매칭.
                    # 시간(MISS_LIMIT) 기반 폐기 아님: 모드 지속 중 이용자 정체성은 계속 유지.
                    alive_ids = {int(t[4]) for t in tracks}
                    if self.locked_id not in alive_ids:
                        # 가림 감지: 타겟 마지막 위치 근처에 다른 track이 있으면 그게 가린 것
                        occluders = []
                        if self.last_cx is not None and self.last_dist is not None:
                            for _t in tracks:
                                _tid = int(_t[4])
                                _tx1, _ty1, _tx2, _ty2 = map(int, _t[:4])
                                _tcx = (_tx1 + _tx2) // 2
                                _tcy = (_ty1 + _ty2) // 2
                                _tdepth = self.median_depth(depth, _tcx, _tcy)
                                _near_cx = abs(_tcx - self.last_cx) <= self.OCC_CX_MARGIN
                                _near_d = (_tdepth <= 0 or
                                           abs(_tdepth - self.last_dist) <= self.OCC_DIST_MARGIN)
                                if _near_cx and _near_d:
                                    occluders.append(_tid)
                                    self.occlusion_blacklist.add(_tid)
                        if occluders:
                            self.get_logger().info(
                                f"\033[93m[OCCLUSION] id={self.locked_id} 가림 감지 "
                                f"-> 재락금지 {occluders}\033[0m")
                        self.get_logger().info(f"[UNLOCK] id={self.locked_id} track 소멸 → 재식별 대기(feat 유지)")
                        self._unlock_count += 1
                        self.locked_id = None
                        self.target_miss = 0
            if target is not None:
                dist, cx, cy, x1, y1, x2, y2, _tid, _tcf = target

                if self.searching:
                    self.find_until = now + FIND_HOLD_SEC
                    self.searching = False
                    self.get_logger().info("[FIND] 사람 재발견 → 정지")
                    self.publish(MODE_FIND)
                    status = "FIND_START"
                else:
                    distance = dist
                    angle = self.pixel_to_angle(cx)
                    status = "FOLLOWING"
                    self.lost_count = 0
                    self.last_cx = cx
                    self.last_dist = dist
                    if cx < CENTER_X - 40:
                        self.last_seen_side = "left"
                    elif cx > CENTER_X + 40:
                        self.last_seen_side = "right"
                    self.publish(MODE_FOLLOW, distance, angle)

                _label = f"TARGET LOCKED  {dist:.2f}m"
                draw_fancy_box(annotated, x1, y1, x2, y2, _label, (0, 0, 255), thickness=3, filled_label=True)
        if not people:
            # ===== 사람 없음 =====
            self.lost_count += 1
            if self.lost_count >= LOST_DELAY:
                self.searching = True
                if self.last_seen_side == "right":
                    status = "SEARCH_RIGHT"
                    self.publish(MODE_SEARCH_RIGHT)
                else:
                    status = "SEARCH_LEFT"
                    self.publish(MODE_SEARCH_LEFT)
            else:
                status = "LOST_WAIT"

        self._draw_and_stream(annotated, status, distance, angle)

    def _draw_pickup_status(self, annotated):
        # 우측 상단: CUT(블록 잘림) 표시 (최우선)
        now = time.time()
        yy = 30
        if now <= self.cut_show_until:
            _put_label_seg_right_pil(
                annotated, [("[!] 블록 잘림 - 대기중", (255, 60, 60))], yy)
            yy += 30
        # 우측 상단: /pickup_status 최근 상태를 작은 세그먼트 라벨로 표시
        if self.pickup_status_msg is None or now > self.pickup_status_until:
            return
        parts = self.pickup_status_msg.split("|")
        kind = parts[0]
        if kind == "STABLE":
            seg = [("● 안정화 대기중 ", (255, 200, 90)),
                   (f"이동 {parts[1]}m", SEG_COLOR_VAL_D)]
        elif kind == "PUB":
            seg = [("● 좌표 발행 ", SEG_COLOR_STATUS),
                   (f"{parts[1]} ", SEG_COLOR_NAME),
                   (f"[{parts[2]}/3] ", SEG_COLOR_CONF),
                   (f"{parts[3]}", SEG_COLOR_VAL_D)]
        elif kind == "MISMATCH":
            seg = [("● 클래스 불일치  ", (255, 60, 60)),
                   ("타겟 ", SEG_COLOR_KEY), (f"{parts[1]}", SEG_COLOR_NAME),
                   (" / 화면 ", SEG_COLOR_KEY), (f"{parts[2]}", (0, 200, 255))]
        else:
            return
        _put_label_seg_right_pil(annotated, seg, yy)

    def _draw_and_stream(self, annotated, status, distance, angle):
        # 텍스트(각도/거리/status)를 먼저 그려서 스트림에도 표시되게 함
        cv2.line(annotated, (CENTER_X, 0), (CENTER_X, 480), (255, 0, 0), 1)
        _put_label_seg_pil(annotated, [(status, SEG_COLOR_STATUS)], 10, 34)
        _put_label_seg_pil(annotated, [
            ("dist ", SEG_COLOR_KEY),
            (f"{distance:.2f}m", SEG_COLOR_VAL_D),
            ("   ang ", SEG_COLOR_KEY),
            (f"{angle:+.3f}rad", SEG_COLOR_VAL_A),
        ], 10, 66)

        global _stream_frame
        with _stream_lock:
            _stream_frame = annotated.copy()

        if SHOW_WINDOW:
            cv2.imshow("follow", annotated)
            cv2.waitKey(1)

    def stop_cmd(self):
        try:
            for _ in range(10):
                self.publish(MODE_FIND, force=True)  # 정지(find=정지)
                rclpy.spin_once(self, timeout_sec=0.02)
                time.sleep(0.02)
        except Exception:
            pass

    def destroy_node(self):
        if getattr(self, "_calib", None) is not None:
            self.get_logger().info(
                f"[CALIB] {self._calib.rows} rows -> {self._calib.path}")
            self._calib.close()
        u = self._unlock_count
        r = self._relock_count
        rate = (r / u * 100.0) if u > 0 else 0.0
        self.get_logger().info(f"[RELOCK-STAT] relock={r} unlock={u} success_rate={rate:.1f}%")
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("종료 - 정지 신호 발행")
        node.stop_cmd()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
