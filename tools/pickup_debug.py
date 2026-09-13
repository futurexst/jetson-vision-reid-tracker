#!/usr/bin/env python3
"""
픽업 좌표 디버깅 도구
- /selected_block (뎁스 원본, 카메라 광학)
- /pickup_target  (AGV로 보내는 변환좌표)
- /object_pose    (로봇팔로 보내는 좌표)

실행: python3 pickup_debug.py
종료: Ctrl+C
"""
import sys
import os
import csv
import time
import threading
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

HISTORY_SEC = 30.0    # 그래프에 유지할 시간창(초)
UPDATE_HZ = 10        # 그래프 갱신 주기(Hz)
LOG_DIR = os.path.expanduser("~/follow_ws/tools/logs")


class PickupDebug(Node):
    def __init__(self):
        super().__init__("pickup_debug")

        # 최신값 저장
        self.latest = {
            "block":  {"t": None, "x": None, "y": None, "z": None},
            "target": {"t": None, "x": None, "y": None, "z": None},
            "pose":   {"t": None, "x": None, "y": None, "z": None},
        }

        # 시계열 히스토리 (t, x, y, z)
        self.hist = {
            "block":  deque(),
            "target": deque(),
            "pose":   deque(),
        }
        self.lock = threading.Lock()
        self.t0 = time.time()

        # 로깅 상태
        self.logging = False
        self.log_file = None
        self.log_writer = None
        self.log_lock = threading.Lock()

        self.create_subscription(Point, "/selected_block",
                                 self.on_block, 10)
        self.create_subscription(Point, "/pickup_target",
                                 self.on_target, 10)
        self.create_subscription(PoseStamped, "/object_pose",
                                 self.on_pose, 10)

        self.get_logger().info("pickup_debug started.")

    def _push(self, key, x, y, z):
        t = time.time() - self.t0
        with self.lock:
            self.latest[key] = {"t": t, "x": x, "y": y, "z": z}
            self.hist[key].append((t, x, y, z))
            # 오래된 값 제거
            cutoff = t - HISTORY_SEC
            while self.hist[key] and self.hist[key][0][0] < cutoff:
                self.hist[key].popleft()
        # CSV 로깅
        with self.log_lock:
            if self.logging and self.log_writer is not None:
                self.log_writer.writerow(
                    [f"{t:.3f}", key, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                self.log_file.flush()

    def start_logging(self):
        with self.log_lock:
            if self.logging:
                return None
            os.makedirs(LOG_DIR, exist_ok=True)
            fname = datetime.now().strftime("pickup_%Y%m%d_%H%M%S.csv")
            path = os.path.join(LOG_DIR, fname)
            self.log_file = open(path, "w", newline="")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(["t_sec", "topic", "x", "y", "z"])
            self.logging = True
            self.get_logger().info(f"[LOG] 저장 시작 -> {path}")
            return path

    def stop_logging(self):
        with self.log_lock:
            if not self.logging:
                return None
            self.logging = False
            path = self.log_file.name if self.log_file else None
            if self.log_file:
                self.log_file.close()
            self.log_file = None
            self.log_writer = None
            self.get_logger().info(f"[LOG] 저장 중지 -> {path}")
            return path

    def on_block(self, msg):
        self._push("block", msg.x, msg.y, msg.z)

    def on_target(self, msg):
        self._push("target", msg.x, msg.y, msg.z)

    def on_pose(self, msg):
        p = msg.pose.position
        self._push("pose", p.x, p.y, p.z)


def print_console(node):
    """콘솔에 최신 3개 좌표 요약 출력"""
    with node.lock:
        b = node.latest["block"]
        t = node.latest["target"]
        p = node.latest["pose"]

    def fmt(d):
        if d["t"] is None:
            return "  (없음)"
        age = time.time() - node.t0 - d["t"]
        return (f"  x={d['x']:+.3f}  y={d['y']:+.3f}  z={d['z']:+.3f}"
                f"   ({age:.1f}s 전)")

    # ANSI 화면 지우기
    sys.stdout.write("\033[2J\033[H")
    log_state = "REC ●" if node.logging else "     "
    print("=" * 60)
    print(f" 픽업 좌표 디버그   [{log_state}]  (Space=로그 토글, Ctrl+C 종료)")
    print("=" * 60)
    print("[뎁스 원본]  /selected_block  (카메라 광학)")
    print(fmt(b))
    print()
    print("[AGV 전송]   /pickup_target   (base_link 변환)")
    print(fmt(t))
    print()
    print("[팔 전송]    /object_pose     (PoseStamped)")
    print(fmt(p))
    print("=" * 60)
    sys.stdout.flush()


def key_listener(node):
    """터미널 stdin에서 스페이스바 감지 -> 로깅 토글"""
    import termios, tty, select
    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if r:
                ch = sys.stdin.read(1)
                if ch == " ":
                    if node.logging:
                        node.stop_logging()
                    else:
                        node.start_logging()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)


def main():
    rclpy.init()
    node = PickupDebug()

    # ROS spin은 별도 스레드에서
    spin_th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_th.start()

    # 키 입력 스레드
    key_th = threading.Thread(target=key_listener, args=(node,), daemon=True)
    key_th.start()

    # matplotlib 준비 (X, Y, Z 3서브플롯)
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Pickup Coord Debug  (block=뎁스원본, target=AGV, pose=팔)")

    labels = ["X", "Y", "Z"]
    colors = {"block": "tab:blue", "target": "tab:orange", "pose": "tab:green"}
    lines = {}
    for i, ax in enumerate(axes):
        ax.set_ylabel(f"{labels[i]} [m]")
        ax.grid(True, alpha=0.3)
        for key, col in colors.items():
            (ln,) = ax.plot([], [], label=key, color=col, linewidth=1.5)
            lines[(key, i)] = ln
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time [s]")

    last_console = 0.0
    try:
        while rclpy.ok():
            # 그래프 갱신
            with node.lock:
                snap = {k: list(v) for k, v in node.hist.items()}

            for key in ("block", "target", "pose"):
                data = snap[key]
                if not data:
                    for i in range(3):
                        lines[(key, i)].set_data([], [])
                    continue
                ts = [d[0] for d in data]
                xs = [d[1] for d in data]
                ys = [d[2] for d in data]
                zs = [d[3] for d in data]
                lines[(key, 0)].set_data(ts, xs)
                lines[(key, 1)].set_data(ts, ys)
                lines[(key, 2)].set_data(ts, zs)

            # 축 자동 스케일
            now = time.time() - node.t0
            for ax in axes:
                ax.set_xlim(max(0.0, now - HISTORY_SEC), now + 0.5)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # 콘솔은 1초에 한 번
            if time.time() - last_console > 1.0:
                print_console(node)
                last_console = time.time()

            time.sleep(1.0 / UPDATE_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        if node.logging:
            node.stop_logging()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
