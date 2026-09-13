import time
import os
import subprocess
import threading
import json as _json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String, Float32
from geometry_msgs.msg import Twist, Point, PoseStamped

WIFI_IFACE = "wlP1p1s0"

# 이벤트성 토픽: (토픽명, 타입, 노드, 작업그룹)
EVENT_TOPICS = [
    ("/nano_send_status", Int32, "외부", "모드/제어"),
    ("/agv_mode", Int32, "fsm", "모드/제어"),
    ("/AGV_mode_ack", Int32, "agv", "모드/제어"),
    ("/AGV_status", Int32, "agv", "모드/제어"),
    ("/tts_text", String, "fsm", "모드/제어"),
    ("/follow_enable", Bool, "fsm", "추종"),
    ("/guide_pause", Bool, "fsm", "안내"),
    ("/rear_person", Bool, "follow", "안내"),
    ("/selected_block", Point, "follow", "픽업-블록인식"),
    ("/selected_block_class", String, "follow", "픽업-블록인식"),
    ("/laser_point", Point, "follow", "픽업-블록인식"),
    ("/pickup_status", String, "pickup", "픽업-도킹/파지"),
    ("/pickup_active", Bool, "fsm", "픽업-도킹/파지"),
    ("/pickup_target", Point, "pickup", "픽업-도킹/파지"),
    ("/object_pose", PoseStamped, "pickup", "픽업-도킹/파지"),
    ("/ARM_status", Int32, "arm", "픽업-도킹/파지"),
]

# 스트리밍 토픽
TARGET_CMD = "/target_cmd"
CAM_HEALTH = "/cam_health"

TARGET_HZ_OK = (7.0, 30.0)
CAM_FPS_OK = 8.0

PING_TARGETS = [
    ("대시보드", "192.168.0.125"),
    ("AGV", "192.168.0.101"),
    ("로봇팔", "192.168.0.102"),
]
TARGET_EXPECT_HZ = 10.0
# ANSI 색상
C_RESET = "\033[0m"
C_DIM = "\033[90m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"
# 노드별 색 (노드+토픽 동일색으로 구분)
NODE_COLOR = {
    "외부": C_WHITE,
    "agv": C_YELLOW,
    "fsm": C_CYAN,
    "follow": C_GREEN,
    "pickup": C_MAGENTA,
    "arm": "\033[38;5;208m",  # 주황
    "tts": "\033[38;5;213m",  # 분홍
}
# ===== 값 의미 해설 =====
MODE_CAT_NAMES = {1: "추종", 2: "안내", 3: "자율주행", 4: "위치재조정", 5: "픽업", 6: "수동"}

def desc_mode(v):
    if v == 0:
        return "정지"
    cat, sub = v // 10, v % 10
    base = MODE_CAT_NAMES.get(cat)
    if base is None:
        return "?"
    if cat in (2, 3):
        return f"{base}·목적지{sub}"
    if cat == 5:
        return {0: "픽업·유도", 1: "픽업·좌표전송"}.get(sub, base)
    return base

def desc_ack(v):
    if v == 52:
        return "픽업·좌표락온"
    return desc_mode(v)

def desc_pickup_status(v):
    parts = str(v).split("|")
    k = parts[0]
    if k == "STABLE":
        return f"안정화대기(이동{parts[1]}m)" if len(parts) > 1 else "안정화대기"
    if k == "PUB":
        return f"발행 {parts[1]}[{parts[2]}/3]" if len(parts) > 2 else "발행"
    if k == "MISMATCH":
        return f"클래스불일치(타겟{parts[1]}/화면{parts[2]})" if len(parts) > 2 else "클래스불일치"
    if k == "DONE":
        return "재탐색종료(추론중단)"
    if k == "REGRIP":
        return "재파지(추론재개)"
    return ""

VAL_DESC = {
    "/nano_send_status": desc_mode,
    "/agv_mode": desc_mode,
    "/AGV_mode_ack": desc_ack,
    "/AGV_status": {0: "주행시작", 1: "도착/완료", 2: "장애물감지", 4: "재도킹 좌표요청"},
    "/tts_text": "멘트",
    "/follow_enable": {True: "추종ON", False: "추종OFF"},
    "/guide_pause": {True: "대기중", False: "주행중"},
    "/rear_person": {True: "사람있음", False: "사람없음"},
    "/selected_block": "블록좌표(카메라)",
    "/selected_block_class": "블록종류",
    "/laser_point": "레이저좌표",
    "/pickup_status": desc_pickup_status,
    "/pickup_active": {True: "좌표송출중", False: "송출중단"},
    "/pickup_target": "도킹좌표(AGV)",
    "/object_pose": "파지좌표(팔)",
    "/ARM_status": {1: "팔 작업완료", 0: "파지실패/재파지요청"},
}

HEARTBEAT_NODES = ["follow_node", "fsm_node", "tts_node", "pickup_node"]  # heartbeat 발행 노드 목록
HEARTBEAT_TIMEOUT = 6.0            # 이 시간 내 미수신 시 DEAD 판정


class CommMonitor(Node):
    def __init__(self):
        super().__init__("comm_monitor_node")

        # 이벤트성: 마지막 수신시각 + 누적횟수
        self.ev_last = {t[0]: None for t in EVENT_TOPICS}
        self.ev_count = {t[0]: 0 for t in EVENT_TOPICS}
        self.ev_val = {t[0]: None for t in EVENT_TOPICS}
        for name, msgtype, _node, _grp in EVENT_TOPICS:
            self.create_subscription(
                msgtype, name,
                lambda msg, n=name: self._on_event(n, msg), 10)

        # /target_cmd: 수신 시각 기록 -> Hz/지터 계산
        self.tc_follow = []
        self.tc_pickup = []
        # 마지막 유효 target_cmd 통계 캐시 (미수신 시 마지막 값 유지 표시용)
        self.tc_follow_last = None  # (hz, jitter, jpct)
        self.tc_pickup_last = None
        self.create_subscription(
            Twist, TARGET_CMD, self._on_target_cmd, 10)

        # /cam_health: 마지막 fps 값 + 수신시각
        self.cam_fps = None
        self.cam_last = None
        self.create_subscription(
            Float32, CAM_HEALTH, self._on_cam_health, 10)
        # /heartbeat/<node>: 노드별 자체 상태 (JSON)
        self.hb_data = {n: None for n in HEARTBEAT_NODES}
        self.hb_last = {n: None for n in HEARTBEAT_NODES}
        for n in HEARTBEAT_NODES:
            self.create_subscription(
                String, f"/heartbeat/{n}",
                lambda msg, nn=n: self._on_heartbeat(nn, msg), 10)

        # CPU 사용률 계산용 직전값
        self._prev_cpu = self._read_cpu_raw()

        # ping 결과 캐시: name -> (loss%, avg_ms, mdev_ms) or None
        self.ping_result = {name: None for name, _ in PING_TARGETS}
        self.bridge_status = None  # domain-bridge.service active 여부 캐시
        self._ping_stop = False
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()
        self._bridge_stop = False
        self._bridge_thread = threading.Thread(target=self._bridge_loop, daemon=True)
        self._bridge_thread.start()

        # Phase2: 시계열 링 버퍼 (최근 60초, 1초 주기 적재)
        self.hist = deque(maxlen=60)
        self.create_timer(1.0, self._push_history)
        self.create_timer(1.0, self.print_status)
        # 웹 대시보드 서버 (CLI와 병행)
        _start_dashboard_server(self, port=8090)
        self.get_logger().info("dashboard: http://192.168.0.103:8090")

    def _on_event(self, name, msg):
        self.ev_last[name] = time.time()
        self.ev_count[name] += 1
        self.ev_val[name] = self._fmt_msg(msg)

    def _fmt_msg(self, msg):
        # 타입별 값 문자열화 (Int32/String/Bool: data / Point: xyz / PoseStamped: pos xyz)
        if hasattr(msg, "data"):
            return str(msg.data)
        if hasattr(msg, "x") and hasattr(msg, "z"):
            return f"x={msg.x:.3f} y={msg.y:.3f} z={msg.z:.3f}"
        if hasattr(msg, "pose"):
            p = msg.pose.position
            return f"x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}"
        return "?" 

    def _val_desc(self, name, val):
        d = VAL_DESC.get(name)
        if d is None or val is None:
            return ""
        if isinstance(d, str):
            return d
        if callable(d):
            try:
                return d(int(val))
            except (TypeError, ValueError):
                return ""
        if val in ("True", "False"):
            return d.get(val == "True", "")
        try:
            return d.get(int(val), "")
        except (TypeError, ValueError):
            return ""

    def _on_target_cmd(self, msg):
        now = time.time()
        mode = self.ev_val.get("/agv_mode")
        cat = None
        try:
            cat = int(mode) // 10
        except (TypeError, ValueError):
            cat = None
        if cat == 5:
            self.tc_pickup.append(now)
        else:
            self.tc_follow.append(now)
        # 최근 3초치만 유지 (_target_stats 윈도우와 동일)
        cut = now - 3.0
        self.tc_follow = [t for t in self.tc_follow if t >= cut]
        self.tc_pickup = [t for t in self.tc_pickup if t >= cut]

    def _on_cam_health(self, msg):
        self.cam_fps = msg.data
        self.cam_last = time.time()

    # ---------- 시스템 지표 ----------
    def _on_heartbeat(self, name, msg):
        import json
        try:
            self.hb_data[name] = json.loads(msg.data)
            self.hb_last[name] = time.time()
        except Exception:
            pass

    def _read_cpu_raw(self):
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()[1:]
            vals = list(map(int, parts))
            idle = vals[3] + vals[4]
            total = sum(vals)
            return idle, total
        except Exception:
            return None

    def cpu_percent(self):
        cur = self._read_cpu_raw()
        if not cur or not self._prev_cpu:
            self._prev_cpu = cur
            return None
        idle0, total0 = self._prev_cpu
        idle1, total1 = cur
        self._prev_cpu = cur
        dt = total1 - total0
        di = idle1 - idle0
        if dt <= 0:
            return None
        return 100.0 * (1.0 - di / dt)

    def mem_percent(self):
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":")
                    info[k] = int(v.strip().split()[0])
            total = info["MemTotal"]
            avail = info["MemAvailable"]
            return 100.0 * (1.0 - avail / total)
        except Exception:
            return None

    def wifi_rssi(self):
        try:
            with open("/proc/net/wireless") as f:
                for line in f:
                    if WIFI_IFACE in line:
                        cols = line.split()
                        link = cols[2].rstrip(".")
                        level = cols[3].rstrip(".")
                        return int(float(link)), int(float(level))
        except Exception:
            pass
        return None, None

    # ---------- 출력 ----------
    def _pad(self, text, width):
        import unicodedata
        w = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in text)
        return text + ' ' * max(0, width - w)

    def _push_history(self):
        # 1초 주기: 스냅샷에서 시계열 값만 추출해 링 버퍼에 적재
        try:
            snap = self._snapshot()
        except Exception:
            return
        st = snap.get("streams", {})
        sysm = snap.get("system", {})
        tf = st.get("target_follow")
        tp = st.get("target_pickup")
        cam = st.get("cam")
        # ping RTT를 대상명 기준으로 추출 (대시보드/AGV/로봇팔)
        ping_map = {p.get("name"): p.get("avg") for p in snap.get("ping", [])}
        self.hist.append({
            "ts": snap.get("ts"),
            "hz_follow": (tf["hz"] if tf else None),
            "hz_pickup": (tp["hz"] if tp else None),
            "jit_follow": (tf["jitter_pct"] if tf else None),
            "jit_pickup": (tp["jitter_pct"] if tp else None),
            "cam_fps": (cam["fps"] if cam else None),
            "cpu": sysm.get("cpu"),
            "mem": sysm.get("mem"),
            "rssi": sysm.get("wifi_rssi"),
            "ping_dash": ping_map.get("대시보드"),
            "ping_agv": ping_map.get("AGV"),
            "ping_arm": ping_map.get("로봇팔"),
        })

    def _snapshot(self):
        # 웹 대시보드용 현재 상태 JSON 스냅샷
        now = time.time()
        events = []
        for name, _mt, node, grp in EVENT_TOPICS:
            last = self.ev_last[name]
            val = self.ev_val.get(name)
            events.append({
                "topic": name, "node": node, "group": grp,
                "count": self.ev_count[name],
                "ago": (round(now - last, 1) if last else None),
                "fresh": (bool(last and (now - last) <= 30.0)),
                "val": (str(val) if val is not None else None),
                "desc": self._val_desc(name, val),
            })
        def _stream(times, cache):
            st = self._target_stats(times)
            if st[0] is None:
                return None
            hz, jit, jpct = st
            return {"hz": round(hz, 1), "jitter_pct": round(jpct, 1),
                    "ok": bool(TARGET_HZ_OK[0] <= hz <= TARGET_HZ_OK[1])}
        cam_ago = (now - self.cam_last) if self.cam_last else None
        cam = None
        if self.cam_fps is not None:
            cam = {"fps": round(self.cam_fps, 1),
                   "ago": (round(cam_ago, 1) if cam_ago else None),
                   "ok": bool(self.cam_fps >= CAM_FPS_OK
                              and cam_ago is not None and cam_ago < 6.0)}
        hbs = []
        for n in HEARTBEAT_NODES:
            d = self.hb_data.get(n)
            last = self.hb_last.get(n)
            dead = (last is None) or (now - last > HEARTBEAT_TIMEOUT)
            hbs.append({
                "node": n, "dead": bool(dead),
                "ago": (round(now - last, 1) if last else None),
                "fps": (d.get("fps") if d else None),
                "status": (d.get("status") if d else None),
            })
        link, level = self.wifi_rssi()
        pings = []
        for pname, ip in PING_TARGETS:
            r = self.ping_result.get(pname)
            if r is None:
                pings.append({"name": pname, "ip": ip, "ok": False,
                              "loss": None, "avg": None, "mdev": None})
            else:
                loss, avg, mdev = r
                pings.append({"name": pname, "ip": ip,
                              "loss": loss, "avg": avg, "mdev": mdev,
                              "ok": bool(loss == 0.0 and avg is not None)})
        return {
            "ts": now,
            "events": events,
            "streams": {
                "target_follow": _stream(self.tc_follow, None),
                "target_pickup": _stream(self.tc_pickup, None),
                "cam": cam,
            },
            "heartbeats": hbs,
            "system": {"cpu": self.cpu_percent(),
                       "mem": self.mem_percent(),
                       "wifi_link": link, "wifi_rssi": level},
            "ping": pings,
        }

    def print_status(self):
        os.system("clear")
        now = time.time()
        lines = []
        lines.append("=== ROS2 통신 모니터 ===")
        lines.append("")

        # 이벤트성
        lines.append(C_WHITE + "📡" + self._pad("노드", 10) + self._pad("토픽", 24) + self._pad("수신", 12) + self._pad("마지막", 10) + self._pad("누적", 6) + self._pad("값", 32) + "의미" + C_RESET)
        prev_grp = None
        for name, _mt, node, grp in EVENT_TOPICS:
            if grp != prev_grp:
                if prev_grp is not None:
                    lines.append("")
                lines.append("  " + C_WHITE + "▶ " + grp + C_RESET)
                prev_grp = grp
            last = self.ev_last[name]
            cnt = self.ev_count[name]
            col = NODE_COLOR.get(node, C_WHITE)
            nodecol = col + self._pad("[" + node + "]", 10) + self._pad(name, 24) + C_RESET
            if last is None:
                lines.append("  " + nodecol + C_DIM + self._pad("⚪미수신", 12) + self._pad("-", 10) + self._pad("x" + str(cnt), 6) + C_RESET)
            else:
                ago = now - last
                fresh = ago <= 30.0
                tag = (C_GREEN + "🟢OK" if fresh else C_YELLOW + "🟡과거") + C_RESET
                val = self.ev_val.get(name)
                lines.append("  " + nodecol + self._pad(tag + " ", 12 + len(C_GREEN) + len(C_RESET)) + self._pad(f"{ago:.1f}s전", 10) + self._pad("x" + str(cnt), 6) + C_WHITE + self._pad(f"val={val}", 32) + C_RESET + C_DIM + self._val_desc(name, val) + C_RESET)
        lines.append("")

        # target_cmd
        lines.append(C_WHITE + "📶 [스트리밍]" + C_RESET)
        gc = NODE_COLOR["follow"]
        pc = NODE_COLOR["pickup"]
        for label, times, col, cache_attr in [("추종 target_cmd", self.tc_follow, gc, "tc_follow_last"), ("픽업 target_cmd", self.tc_pickup, pc, "tc_pickup_last")]:
            stats = self._target_stats(times)
            if stats[0] is not None:
                setattr(self, cache_attr, stats)  # 유효값 캐시 갱신
            disp = stats if stats[0] is not None else getattr(self, cache_attr)
            if disp is None or disp[0] is None:
                lines.append("  " + col + self._pad(label, 18) + C_RESET + C_DIM + self._pad("⚪미수신", 30) + "지터 --.-% [" + C_DIM + "대기" + C_RESET + C_DIM + "]" + C_RESET)
            else:
                hz, jitter, jpct = disp
                live = stats[0] is not None  # 현재 실시간 수신 여부
                ok = TARGET_HZ_OK[0] <= hz <= TARGET_HZ_OK[1]
                if live:
                    tag = (C_GREEN + "🟢OK" if ok else C_RED + "🔴!!") + C_RESET
                else:
                    tag = C_DIM + "⚪최근" + C_RESET  # 캐시된 마지막 값 표시
                dev = (hz - TARGET_EXPECT_HZ) / TARGET_EXPECT_HZ * 100.0
                if jpct < 10.0:
                    jt = C_GREEN + "좋음" + C_RESET
                elif jpct < 25.0:
                    jt = C_YELLOW + "보통" + C_RESET
                else:
                    jt = C_RED + "나쁨" + C_RESET
                lines.append("  " + col + self._pad(label, 18) + C_RESET + tag + f"  {hz:4.1f}Hz (편차 {dev:+4.1f}%)  지터 {jpct:4.1f}% [" + jt + "]")

        # cam_health
        if self.cam_fps is None:
            lines.append("  " + gc + self._pad(CAM_HEALTH, 18) + C_RESET + C_DIM + "⚪미수신" + C_RESET)
        else:
            cam_ago = now - self.cam_last if self.cam_last else 999
            ok = self.cam_fps >= CAM_FPS_OK and cam_ago < 6.0
            tag = (C_GREEN + "🟢OK" if ok else C_RED + "🔴!!") + C_RESET
            lines.append("  " + gc + self._pad(CAM_HEALTH, 18) + C_RESET + tag + f"  {self.cam_fps:4.1f}fps ({cam_ago:.1f}s전)")
        lines.append("")

        # 시스템
        lines.append(C_WHITE + "💓 [하트비트]" + C_RESET)
        now_hb = time.time()
        for n in HEARTBEAT_NODES:
            d = self.hb_data.get(n)
            last = self.hb_last.get(n)
            ncol = NODE_COLOR.get(n.replace("_node", ""), C_WHITE)
            head = ncol + self._pad(n, 16) + C_RESET
            if d is None or last is None:
                lines.append("  " + head + C_DIM + "⚪미수신" + C_RESET)
            elif now_hb - last > HEARTBEAT_TIMEOUT:
                lines.append("  " + head + C_RED + f"🔴DEAD ({now_hb - last:.1f}s 무응답)" + C_RESET)
            else:
                fps = d.get("fps", "-")
                st = d.get("status", "-")
                extra = ""
                if n == "pickup_node":
                    rg = d.get("regrip", 0)
                    rs = d.get("research_sent", 0)
                    extra = C_MAGENTA + f"  재파지 {rg}회  발행 {rs}/3" + C_RESET
                if n == "follow_node":
                    ct = d.get("cam_temp", None)
                    if ct is not None and ct >= 0:
                        tcol = C_GREEN if ct < 70 else (C_YELLOW if ct < 85 else C_RED)
                        extra = tcol + f"  D435 {ct:.1f}°C" + C_RESET
                lines.append("  " + head + C_GREEN + "🟢OK" + C_RESET + f"  fps={fps}  {st}" + extra)
        lines.append("")
        lines.append(C_WHITE + "🖥️ [시스템]" + C_RESET)
        cpu = self.cpu_percent()
        mem = self.mem_percent()
        link, level = self.wifi_rssi()
        def _uc(v):
            if v is None:
                return C_DIM + "  -" + C_RESET
            c = C_GREEN if v < 60 else (C_YELLOW if v < 85 else C_RED)
            return c + f"{v:4.1f}%" + C_RESET
        lines.append("  CPU " + _uc(cpu) + "   MEM " + _uc(mem))
        if level is not None:
            if level >= -50:
                wtag, wc = "좋음", C_GREEN
            elif level >= -60:
                wtag, wc = "양호", C_GREEN
            elif level >= -70:
                wtag, wc = "보통", C_YELLOW
            else:
                wtag, wc = "나쁨", C_RED
            lines.append("  📡 WiFi(" + WIFI_IFACE + ")  link " + str(link) + "  RSSI " + str(level) + "dBm  " + wc + "[" + wtag + "]" + C_RESET)
        else:
            lines.append("  📡 WiFi(" + WIFI_IFACE + ")  " + C_DIM + "-" + C_RESET)
        bs = self.bridge_status
        if bs == "active":
            lines.append("  🌉 domain-bridge  " + C_GREEN + "🟢active" + C_RESET)
        elif bs is None:
            lines.append("  🌉 domain-bridge  " + C_DIM + "-" + C_RESET)
        else:
            lines.append("  🌉 domain-bridge  " + C_RED + f"🔴{bs}" + C_RESET)
        lines.append("")
        lines.append(C_WHITE + "🌐 [네트워크 ping]" + C_RESET)
        for name, ip in PING_TARGETS:
            res = self.ping_result.get(name)
            head = self._pad(name, 8) + C_DIM + "(" + ip + ")" + C_RESET
            if res is None:
                lines.append("  " + head + "  " + C_RED + "🔴응답없음" + C_RESET)
            else:
                loss, avg, mdev = res
                loss_s = f"{loss:.0f}%" if loss is not None else "-"
                avg_s = f"{avg:.1f}ms" if avg is not None else "-"
                mdev_s = f"{mdev:.1f}ms" if mdev is not None else "-"
                ok = (loss == 0.0 and avg is not None)
                tag = (C_GREEN + "🟢OK" if ok else C_RED + "🔴!!") + C_RESET
                lines.append("  " + head + "  " + tag + f"  loss {loss_s}  RTT {avg_s} (±{mdev_s})")
        print("\n".join(lines))

    def _ping_loop(self):
        while not self._ping_stop:
            for name, ip in PING_TARGETS:
                self.ping_result[name] = self._ping_once(ip)
            time.sleep(3.0)
    def _bridge_loop(self):
        while not self._bridge_stop:
            try:
                out = subprocess.run(
                    ["systemctl", "is-active", "domain-bridge.service"],
                    capture_output=True, text=True, timeout=3)
                self.bridge_status = out.stdout.strip()
            except Exception:
                self.bridge_status = None
            time.sleep(3.0)

    def _ping_once(self, ip):
        try:
            out = subprocess.run(
                ["ping", "-c", "2", "-w", "3", ip],
                capture_output=True, text=True, timeout=5)
            if out.returncode != 0:
                return None
            txt = out.stdout
            loss = None
            avg = None
            mdev = None
            for line in txt.splitlines():
                if "packet loss" in line:
                    seg = line.split("%")[0].split(",")[-1].strip()
                    loss = float(seg)
                if "rtt" in line or "round-trip" in line:
                    nums = line.split("=")[1].strip().split()[0]
                    parts = nums.split("/")
                    avg = float(parts[1])
                    if len(parts) >= 4:
                        mdev = float(parts[3])
            return (loss, avg, mdev)
        except Exception:
            return None

    def _target_stats(self, times):
        now = time.time()
        cut = now - 3.0
        times[:] = [t for t in times if t >= cut]
        ts = sorted(times)
        if len(ts) < 2:
            return None, None
        span = ts[-1] - ts[0]
        if span <= 0:
            return None, None
        hz = (len(ts) - 1) / span
        # 지터: 간격의 표준편차
        gaps = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        jitter = var ** 0.5
        jitter_pct = (jitter / mean * 100.0) if mean > 0 else 0.0
        return hz, jitter, jitter_pct



# ============================================================
# 웹 대시보드 서버 (comm_monitor 데이터를 브라우저로 노출)
# http://192.168.0.103:8090
# ============================================================
_DASH_NODE = None

class _DashHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/history"):
            self._send_history()
        elif self.path.startswith("/api/status"):
            self._send_json()
        elif self.path == "/" or self.path.startswith("/index"):
            self._send_html()
        else:
            self.send_response(404)
            self.end_headers()

    def _send_history(self):
        try:
            data = list(_DASH_NODE.hist) if _DASH_NODE else []
            body = _json.dumps(data).encode("utf-8")
        except Exception as e:
            body = _json.dumps({"error": str(e)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def _send_json(self):
        try:
            data = _DASH_NODE._snapshot() if _DASH_NODE else {}
            body = _json.dumps(data).encode("utf-8")
        except Exception as e:
            body = _json.dumps({"error": str(e)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        body = _DASH_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_dashboard_server(node, port=8090):
    global _DASH_NODE
    _DASH_NODE = node
    srv = ThreadingHTTPServer(("0.0.0.0", port), _DashHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


_DASH_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>ROS2 Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:'Segoe UI',monospace; background:#0a0f18; color:#c8d2e0;
         width:1200px; height:498px; overflow:hidden; }
  .wrap { display:grid; grid-template-columns:340px 1fr 1fr; gap:10px;
          padding:10px; height:498px; }
  .col { display:flex; flex-direction:column; gap:10px; min-height:0; }
  .panel { background:#0f1826; border:1px solid #1c2b3f; border-radius:8px;
           padding:8px 10px; position:relative; overflow:hidden; }
  .panel::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
                   background:linear-gradient(90deg,#3db8e0,#2a7fb8); }
  .ph { font-size:11px; font-weight:700; color:#3db8e0; letter-spacing:0.5px;
        text-transform:uppercase; margin-bottom:6px; display:flex;
        justify-content:space-between; align-items:center; }
  .ph .sub { color:#5a6b82; font-weight:400; font-size:10px; }
  table { border-collapse:collapse; width:100%; font-size:10.5px; }
  th,td { text-align:left; padding:1.5px 4px; }
  th { color:#5a6b82; font-weight:600; border-bottom:1px solid #1c2b3f; }
  td { border-bottom:1px solid #131f2e; }
  .ok { color:#3dd67f; } .bad { color:#e05a5a; } .dim { color:#4a5875; }
  .scroll::-webkit-scrollbar { width:5px; }
  .scroll::-webkit-scrollbar-track { background:#0a1420; border-radius:3px; }
  .scroll::-webkit-scrollbar-thumb { background:#2a4058; border-radius:3px; }
  .scroll::-webkit-scrollbar-thumb:hover { background:#3db8e0; }
  .scroll { scrollbar-width:thin; scrollbar-color:#2a4058 #0a1420; }
  .grp { color:#7a8aa0; font-weight:700; font-size:9.5px; padding-top:4px; }
  .chartbox { flex:1; min-height:0; position:relative; }
  .chartbox canvas { position:absolute; inset:0; }
  .sysrow { display:flex; gap:8px; font-size:11px; margin-bottom:2px; }
  .metric { flex:1; background:#0a1420; border-radius:5px; padding:4px 8px; }
  .metric .v { font-size:15px; font-weight:700; color:#e0e8f2; }
  .metric .k { font-size:9px; color:#5a6b82; text-transform:uppercase; }
  h1 { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <!-- 좌열: 이벤트 + 하트비트 -->
  <div class="col">
    <div class="panel scroll" style="flex:2.6;overflow-y:auto">
      <div class="ph">이벤트 토픽 <span class="sub" id="ev-fresh"></span></div>
      <table id="tbl-ev"><tbody></tbody></table>
    </div>
    <div class="panel" style="flex:0 0 auto">
      <div class="ph">노드 하트비트</div>
      <table id="tbl-hb"><tbody></tbody></table>
    </div>
  </div>
  <!-- 중열: 스트리밍 그래프 -->
  <div class="col">
    <div class="panel" style="flex:1">
      <div class="ph">target_cmd 주파수 <span class="sub">Hz · 60s</span></div>
      <div class="chartbox"><canvas id="chHz"></canvas></div>
    </div>
    <div class="panel" style="flex:1">
      <div class="ph">카메라 FPS <span class="sub">fps · 60s</span></div>
      <div class="chartbox"><canvas id="chCam"></canvas></div>
    </div>
    <div class="panel" style="flex:1">
      <div class="ph">지터 <span class="sub">% · 60s</span></div>
      <div class="chartbox"><canvas id="chJit"></canvas></div>
    </div>
  </div>
  <!-- 우열: 시스템 + 네트워크 그래프 -->
  <div class="col">
    <div class="panel">
      <div class="ph">시스템 리소스</div>
      <div class="sysrow">
        <div class="metric"><div class="v" id="m-cpu">-</div><div class="k">CPU</div></div>
        <div class="metric"><div class="v" id="m-mem">-</div><div class="k">MEM</div></div>
        <div class="metric"><div class="v" id="m-rssi">-</div><div class="k">WiFi RSSI</div></div>
      </div>
    </div>
    <div class="panel" style="flex:1">
      <div class="ph">CPU / MEM <span class="sub">% · 60s</span></div>
      <div class="chartbox"><canvas id="chSys"></canvas></div>
    </div>
    <div class="panel" style="flex:1">
      <div class="ph">네트워크 RTT <span class="sub">ms · 60s</span></div>
      <div class="chartbox"><canvas id="chPing"></canvas></div>
    </div>
    <div class="panel">
      <div class="ph">Ping 상태</div>
      <table id="tbl-ping"><tbody></tbody></table>
    </div>
  </div>
</div>
<script>
const AX = { color:'#4a5875', font:{size:9} };
const GRID = { color:'#131f2e' };
function mkChart(id, series, ymax){
  return new Chart(document.getElementById(id).getContext('2d'), {
    type:'line',
    data:{ labels:[], datasets: series.map(x=>({
      label:x.label, data:[], borderColor:x.color, backgroundColor:x.color,
      borderWidth:1.5, pointRadius:0, tension:0.3, spanGaps:true })) },
    options:{ animation:false, responsive:true, maintainAspectRatio:false,
      layout:{padding:2},
      scales:{ x:{ ticks:{...AX,maxTicksLimit:5}, grid:GRID },
               y:{ ticks:AX, grid:GRID, beginAtZero:true,
                   suggestedMax:(ymax||undefined) } },
      plugins:{ legend:{ labels:{color:'#8a9ab0',boxWidth:10,font:{size:10},
                         padding:6}, position:'top', align:'end' } } }
  });
}
const chHz = mkChart('chHz',[{label:'추종',color:'#3dd67f'},{label:'픽업',color:'#c07fe0'}],15);
const chCam = mkChart('chCam',[{label:'cam fps',color:'#3db8e0'}],25);
const chJit = mkChart('chJit',[{label:'추종',color:'#3dd67f'},{label:'픽업',color:'#c07fe0'}]);
const chSys = mkChart('chSys',[{label:'CPU',color:'#e0a04b'},{label:'MEM',color:'#e05a5a'}],100);
const chPing = mkChart('chPing',[{label:'대시보드',color:'#3db8e0'},
                        {label:'AGV',color:'#e0a04b'},{label:'로봇팔',color:'#c07fe0'}]);
function updChart(ch, rows, keys){
  ch.data.labels = rows.map(r=>{ const d=new Date(r.ts*1000);
    return d.getMinutes()+':'+String(d.getSeconds()).padStart(2,'0'); });
  keys.forEach((k,i)=>{ ch.data.datasets[i].data = rows.map(r=>r[k]); });
  ch.update('none');
}
function cell(v, cls){ return '<td'+(cls?' class="'+cls+'"':'')+'>'+v+'</td>'; }
async function tick(){
  try {
    const r = await fetch('/api/status'); const d = await r.json();
    // 이벤트 (그룹 헤더 포함)
    let h=''; let pg=null;
    for(const e of d.events){
      if(e.group!==pg){ h+='<tr><td colspan="4" class="grp">▶ '+e.group+'</td></tr>'; pg=e.group; }
      const st = e.ago===null ? '<span class="dim">미수신</span>'
               : (e.fresh?'<span class="ok">'+e.ago+'s</span>':'<span class="dim">'+e.ago+'s</span>');
      h+='<tr>'+cell(e.topic.replace(/^\//,''))+cell(e.node,'dim')+cell(st)+cell(e.count,'dim')+'</tr>';
    }
    document.querySelector('#tbl-ev tbody').innerHTML =
      '<tr><th>토픽</th><th>노드</th><th>수신</th><th>N</th></tr>'+h;
    // 하트비트
    let hb='<tr><th>노드</th><th>상태</th><th>fps</th><th>최근</th></tr>';
    for(const b of d.heartbeats){
      hb+='<tr>'+cell(b.node.replace('_node',''))
        + cell(b.dead?'DEAD':'OK', b.dead?'bad':'ok')
        + cell(b.fps!==null?b.fps:'-','dim')
        + cell(b.ago!==null?b.ago+'s':'-','dim')+'</tr>';
    }
    document.querySelector('#tbl-hb tbody').innerHTML = hb;
    // 시스템 메트릭
    const sy=d.system;
    document.getElementById('m-cpu').textContent = sy.cpu!==null?sy.cpu.toFixed(0)+'%':'-';
    document.getElementById('m-mem').textContent = sy.mem!==null?sy.mem.toFixed(0)+'%':'-';
    document.getElementById('m-rssi').textContent = sy.wifi_rssi!==null?sy.wifi_rssi:'-';
    // ping 상태
    let pt='<tr><th>대상</th><th>loss</th><th>RTT</th><th>상태</th></tr>';
    for(const p of d.ping){
      pt+='<tr>'+cell(p.name)+cell(p.loss!==null?p.loss+'%':'-','dim')
        + cell(p.avg!==null?p.avg.toFixed(1):'-','dim')
        + cell(p.ok?'OK':'!!', p.ok?'ok':'bad')+'</tr>';
    }
    document.querySelector('#tbl-ping tbody').innerHTML = pt;
  } catch(e){}
}
async function tickChart(){
  try {
    const r = await fetch('/api/history'); const rows = await r.json();
    if(!Array.isArray(rows)) return;
    updChart(chHz, rows, ['hz_follow','hz_pickup']);
    updChart(chCam, rows, ['cam_fps']);
    updChart(chJit, rows, ['jit_follow','jit_pickup']);
    updChart(chSys, rows, ['cpu','mem']);
    updChart(chPing, rows, ['ping_dash','ping_agv','ping_arm']);
  } catch(e){}
}
tick(); tickChart();
setInterval(tick,1000); setInterval(tickChart,1000);
</script>
</body>
</html>
"""



def main():
    rclpy.init()
    node = CommMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
