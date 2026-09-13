#!/usr/bin/env python3
import os
import time
import threading
import queue
import subprocess
import hashlib
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import asyncio
import edge_tts

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MSG_FILE = os.path.join(PKG_DIR, "textmsg.txt")
EDGE_VOICE = "ko-KR-SunHiNeural"
CACHE_DIR = os.path.expanduser("~/tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def text_to_cache_path(text):
    # 텍스트를 해시로 변환해 캐시 파일명 생성
    h = hashlib.md5(text.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.mp3")


def generate_mp3(text, mp3_path):
    try:
        subprocess.run(
            ["edge-tts", "--voice", EDGE_VOICE,
             "--text", text, "--write-media", mp3_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15)
        if os.path.getsize(mp3_path) == 0:
            return False
        return True
    except Exception:
        return False


class TtsNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        self.is_playing = False
        self.running = True
        self.last_mtime = 0
        self.current_menu = {}
        # 재생 큐: 연속 발행된 멘트가 버려지지 않도록 순차 재생
        self.tts_queue = queue.Queue(maxsize=10)
        self._last_play_end = 0.0

        self.subscription = self.create_subscription(
            String, '/tts_text', self.topic_callback, 10)
        self.hb_pub = self.create_publisher(String, '/heartbeat/tts_node', 10)
        self.hb_timer = self.create_timer(1.0, self.publish_heartbeat)

        self.current_menu = self.load_messages()

        # 백그라운드로 고정 문구 미리 변환 (첫 요청도 빠르게)
        threading.Thread(target=self.preheat_cache, daemon=True).start()
        threading.Thread(target=self.file_watcher_loop, daemon=True).start()
        threading.Thread(target=self.tts_worker, daemon=True).start()

        self.get_logger().info("tts_node started. subscribed to /tts_text")
        self.get_logger().info(f"CACHE_DIR: {CACHE_DIR}")

    def publish_heartbeat(self):
        hb = String()
        hb.data = json.dumps({
            "stamp": time.time(),
            "node": "tts_node",
            "status": "PLAYING" if self.is_playing else "OK",
        })
        self.hb_pub.publish(hb)

    def load_messages(self):
        messages = {}
        if not os.path.exists(MSG_FILE):
            self.get_logger().warn(f"textmsg.txt 없음: {MSG_FILE}")
            return messages
        try:
            with open(MSG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, msg = line.split(":", 1)
                    messages[key.strip()] = msg.strip()
        except Exception as e:
            self.get_logger().error(f"파일 읽기 오류: {e}")
        return messages

    def preheat_cache(self):
        # 노드 시작 시 textmsg.txt 문구를 미리 wav로 변환해 캐시 저장
        for key, text in self.current_menu.items():
            wav_path = text_to_cache_path(text)
            if not os.path.exists(wav_path):
                self.get_logger().info(f"[PREHEAT] '{key}' 변환 중...")
                generate_mp3(text, wav_path)
                self.get_logger().info(f"[PREHEAT] '{key}' 완료")

    def file_watcher_loop(self):
        while rclpy.ok() and self.running:
            if os.path.exists(MSG_FILE):
                try:
                    mtime = os.path.getmtime(MSG_FILE)
                    if mtime != self.last_mtime:
                        self.last_mtime = mtime
                        self.current_menu = self.load_messages()
                        self.get_logger().info("textmsg.txt 업데이트됨")
                        # 새 문구도 미리 변환
                        threading.Thread(
                            target=self.preheat_cache, daemon=True).start()
                except Exception:
                    pass
            time.sleep(1.0)

    def topic_callback(self, msg):
        text = msg.data.strip()
        if not text:
            return

        # 키(m1/m2...)면 텍스트 변환, 아니면 직접 텍스트로 재생
        play_text = self.current_menu.get(text, text)

        try:
            self.tts_queue.put_nowait(play_text)
        except queue.Full:
            self.get_logger().warn(f"재생 큐 가득참 - 폐기: '{play_text}'")
            return
        self.get_logger().info(
            f"[TTS] 큐 적재 '{play_text}' (대기 {self.tts_queue.qsize()})")

    def tts_worker(self):
        # 단일 소비자 스레드: 큐에서 하나씩 꺼내 순차 재생
        # (mpg123 동시 실행/멘트 유실 방지)
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.play_tts(text)
            except Exception as e:
                self.get_logger().error(f"TTS worker 오류: {e}")
            finally:
                self.tts_queue.task_done()

    def play_tts(self, text):
        self.is_playing = True
        self.get_logger().info(f"[TTS] 재생 '{text}'")
        wav_path = text_to_cache_path(text)
        try:
            if not os.path.exists(wav_path):
                # 캐시 없으면 변환 (처음 한 번만 느림)
                self.get_logger().info("[TTS] 변환 중 (첫 요청)...")
                if not generate_mp3(text, wav_path):
                    self.get_logger().error("TTS 변환 실패")
                    return

            # 캐시에서 즉시 재생
            # 블루투스 A2DP 스트림 워밍업: 본 재생 전 짧은 무음 선재생
            # A2DP 워밍업 무음: 직전 재생 후 3초 이상 지났을 때만
            silence = os.path.expanduser("~/tts_cache/_silence.mp3")
            if (os.path.exists(silence)
                    and time.time() - self._last_play_end > 3.0):
                subprocess.run(["mpg123", "-q", silence], check=False)
            subprocess.run(["mpg123", "-q", wav_path], check=True)
            self.get_logger().info("[TTS] 재생 완료")

        except Exception as e:
            self.get_logger().error(f"TTS 오류: {e}")
        finally:
            self._last_play_end = time.time()
            self.is_playing = False


def main(args=None):
    rclpy.init(args=args)
    node = TtsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
