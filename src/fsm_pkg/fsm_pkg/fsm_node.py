import time
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool, String
from geometry_msgs.msg import Point

# ===== 모드 카테고리 (raw 숫자의 십의 자리) =====
# raw: 0=정지, 10~17=추종, 20~27=안내, 30~37=자율주행, 40~47=위치재조정
# category = raw // 10
CAT_STOP = 0
CAT_FOLLOW = 1
CAT_GUIDE = 2
CAT_AUTO = 3
CAT_RECAL = 4
CAT_PICKUP = 5
CAT_MANUAL = 6

CAT_NAMES = {
    CAT_STOP: "정지",
    CAT_FOLLOW: "추종",
    CAT_GUIDE: "안내",
    CAT_AUTO: "자율주행",
    CAT_RECAL: "위치재조정",
    CAT_PICKUP: "픽업",
    CAT_MANUAL: "수동",
}

# 모드(카테고리) 진입 시 멘트
CAT_TTS = {
    CAT_STOP: "정지모드입니다.",
    CAT_FOLLOW: "추종모드입니다.",
    CAT_GUIDE: "안내모드입니다.",
    CAT_AUTO: "자율주행모드입니다.",
    CAT_RECAL: "위치를 재조정합니다.",
    CAT_PICKUP: "픽업모드입니다.",
    CAT_MANUAL: "수동모드입니다.",
}

# 유효 raw 숫자 범위 체크
def valid_raw(raw):
    if raw == 0:
        return True
    return raw // 10 in (CAT_FOLLOW, CAT_GUIDE, CAT_AUTO, CAT_RECAL, CAT_PICKUP, CAT_MANUAL) and 0 <= raw % 10 <= 7
# ====================


class FsmNode(Node):
    def __init__(self):
        super().__init__("fsm_node")

        # 구독
        self.mode_sub = self.create_subscription(
            Int32, "/nano_send_status", self.on_mode_cmd, 10)
        self.rear_sub = self.create_subscription(
            Bool, "/rear_person", self.on_rear_person, 10)
        self.block_sub = self.create_subscription(
            Point, "/selected_block", self.on_selected_block, 10)
        self.agv_status_sub = self.create_subscription(
            Int32, "/AGV_status", self.on_agv_status, 10)
        self.agv_ack_sub = self.create_subscription(
            Int32, "/AGV_mode_ack", self.on_agv_mode_ack, 10)
        self.arm_status_sub = self.create_subscription(
            Int32, "/ARM_status", self.on_arm_status, 10)

        # 발행
        self.follow_en_pub = self.create_publisher(Bool, "/follow_enable", 10)
        self.tts_pub = self.create_publisher(String, "/tts_text", 10)
        self.guide_pause_pub = self.create_publisher(Bool, "/guide_pause", 10)
        self.agv_mode_pub = self.create_publisher(Int32, "/agv_mode", 10)
        self.pickup_active_pub = self.create_publisher(Bool, "/pickup_active", 10)
        self.hb_pub = self.create_publisher(String, "/heartbeat/fsm_node", 10)

        # 상태
        self.raw = 0           # 마지막 받은 raw 숫자 (0/10/20~27/30~37)
        # ===== /agv_mode 핸드셰이크 상태 =====
        self.last_agv_status = None  # 직전 /AGV_status 값 (중복 멘트 방지)
        self.guide_started = False   # 안내 시작(AGV_status=0) 받았는지 - 후방판단 게이트
        self.pending_agv = None     # ack 기다리는 중인 모드 값 (None=대기 없음)
        self.pickup_51_sent = False # 픽업 50->51 자동전환 1회 트리거 플래그
        self.arm_done_said = False  # 팔 완료 멘트 1회만 (3연발 방지)
        self.agv_retry_count = 0    # 반복 발행 횟수
        self.AGV_RETRY_MAX = 10     # 최대 발행 횟수 (10회=5초)
        self.AGV_RETRY_PERIOD = 0.5 # 반복 발행 주기(초)
        self.category = CAT_STOP   # raw // 10
        self.rear_person = False
        # ===== 안내 모드 상태 =====
        self.guide_paused = False           # 현재 정지(기다림) 상태인지
        self.last_present_time = 0.0        # 마지막으로 사람 본 시각
        self.last_absent_time = 0.0         # 마지막으로 사람 없던 시각
        self.PRESENT_HOLD = 2.0             # 이 시간 연속 있어야 "복귀"
        self.ABSENT_HOLD = 2.0              # 이 시간 연속 없어야 "놓침"

        # 시작 시 정지 상태로 진입
        # /agv_mode 반복 발행 타이머
        self.agv_retry_timer = self.create_timer(
            self.AGV_RETRY_PERIOD, self.agv_retry_loop)
        self.hb_timer = self.create_timer(1.0, self.publish_heartbeat)

        # 기동 멘트: tts_node 구독 연결 후 발행 (미연결 시 유실 방지)
        self._boot_done = False
        self.boot_timer = self.create_timer(0.2, self._boot_enter_stop)
        self.get_logger().info("fsm_node started. mode=정지 (멘트 대기)")

    # ===== 콜백 =====
    def on_mode_cmd(self, msg):
        raw = msg.data
        if not valid_raw(raw):
            self.get_logger().warn(f"알 수 없는 모드 숫자: {raw} (무시)")
            return

        # 숫자가 바뀌면 AGV로 바이패스 (목적지 변경 포함)
        if raw != self.raw:
            self.bypass_to_agv(raw)

        new_cat = raw // 10 if raw != 0 else CAT_STOP
        old_cat = self.category
        # 픽업 유도(50) 새로 진입 시 51 자동전환 플래그 리셋
        if raw == 50:
            self.pickup_51_sent = False
            self.arm_done_said = False  # 새 픽업 사이클 -> 완료 멘트 재허용
            self.last_agv_status = None  # 새 픽업 사이클 -> status 중복차단 초기화
        self.raw = raw

        # 십의 자리(카테고리)가 바뀔 때만 모드 진입 처리 (멘트 등)
        if new_cat != old_cat:
            self.get_logger().info(
                f"모드 전환: {CAT_NAMES[old_cat]} -> {CAT_NAMES[new_cat]}")
            self.enter_mode(raw)

    def on_rear_person(self, msg):
        self.rear_person = msg.data

        # 안내 모드가 아니면 후방 결과 무시
        if self.category != CAT_GUIDE:
            return
        # 안내 시작(AGV_status=0) 전이면 따라옴 판단 안 함
        if not self.guide_started:
            return

        now = time.time()
        if self.rear_person:
            self.last_present_time = now
        else:
            self.last_absent_time = now

        # 현재 따라오는 중(pause 아님) -> 일정 시간 연속 없으면 정지
        if not self.guide_paused:
            if (now - self.last_present_time) >= self.ABSENT_HOLD:
                self.guide_paused = True
                self.set_guide_pause(True)
                self.say("기다리고 있을게요.")
                self.get_logger().info("[안내] 사람 놓침 -> 정지/대기")
        # 현재 대기 중 -> 일정 시간 연속 보이면 재개
        else:
            if (now - self.last_absent_time) >= self.PRESENT_HOLD:
                self.guide_paused = False
                self.set_guide_pause(False)
                self.say("출발하겠습니다.")
                self.get_logger().info("[안내] 사람 복귀 -> 주행 재개")

    def on_selected_block(self, msg):
        # 픽업 유도(50) 중 블록이 잡히면 -> 51로 자동 전환 (1회)
        if self.raw != 50:
            return
        if self.pickup_51_sent:
            return
        self.pickup_51_sent = True
        self.raw = 51
        self.bypass_to_agv(51)
        self.say("지정한 타겟을 찾았습니다.")
        self.get_logger().info("[픽업] 블록 감지 -> 51 자동 전환")

    def on_agv_status(self, msg):
        status = msg.data   # 0=시작, 1=도착

        # 장애물(2): 상태값이 아닌 이벤트 -> 중복체크/상태갱신 건너뜀
        if status == 2:
            if self.category in (CAT_GUIDE, CAT_AUTO):
                self.say("장애물 감지, 회피 주행합니다.")
                self.get_logger().info("[장애물] 감지 -> 회피 멘트")
            return
        # 같은 값이 반복 수신되면 무시 (값이 바뀔 때만 멘트)
        if status == self.last_agv_status:
            return
        self.last_agv_status = status

        # 안내 모드: 시작/도착 멘트
        if self.category == CAT_GUIDE:
            if status == 0:
                self.guide_started = True   # 후방 따라옴 판단 시작
                self.get_logger().info("[안내] AGV_status=0 수신 (멘트는 진입 시 처리)")
            elif status == 1:
                self.say("안내한 목적지에 도착하였습니다.")
                self.get_logger().info("[안내] 목적지 도착 -> 도착 멘트")
                self.guide_started = False  # 도착 -> 후방 판단 종료 (놓침/복귀 멘트 중단)
        # 자율주행 모드: 시작/도착 멘트
        elif self.category == CAT_AUTO:
            if status == 0:
                self.say("자율주행 목적지로 출발합니다.")
                self.get_logger().info("[자율주행] 출발 -> 시작 멘트")
            elif status == 1:
                self.say("자율주행 목적지에 도착했습니다.")
                self.get_logger().info("[자율주행] 목적지 도착 -> 도착 멘트")
        # 위치재조정 모드: 로컬라이제이션 완료 멘트
        elif self.category == CAT_RECAL:
            if status == 0:
                self.say("위치가 재조정되었습니다.")
                self.get_logger().info("[위치재조정] 로컬라이제이션 완료 -> 멘트")
        # 픽업 모드: 밀착 완료(1) -> 완료 멘트 + 대기모드 전환
        elif self.category == CAT_PICKUP:
            if status == 4:
                self.set_pickup_active(True)
                self.get_logger().info("[픽업] 재도킹 좌표요청(4) -> pickup_target 재송출 시작")
            elif status == 1:
                self.set_pickup_active(False)  # 안전차원 송출 중단 재확인
                self.say("적재를 시작합니다.")
                self.get_logger().info("[픽업] 밀착 완료 -> 팔 작업 대기 (픽업모드 유지)")
                # 픽업 종료는 팔 작업 완료(/ARM_status=1)에서 처리
        # 그 외 모드는 무시

    def _boot_enter_stop(self):
        # tts 구독자가 붙을 때까지 대기 후 정지모드 진입(멘트 1회)
        if self._boot_done:
            return
        if self.tts_pub.get_subscription_count() < 1:
            return  # 아직 tts 미연결 -> 다음 타이머 틱에 재시도
        self._boot_done = True
        self.boot_timer.cancel()
        self.enter_mode(0)
        self.get_logger().info("[BOOT] tts 연결 확인 -> 정지모드 멘트 발행")

    # ===== 모드 진입 처리 =====
    def enter_mode(self, raw, announce=True):
        cat = raw // 10 if raw != 0 else CAT_STOP
        self.category = cat

        # follow_node 활성화 여부: 추종 카테고리만 True
        self.set_follow_enable(cat == CAT_FOLLOW)

        # 안내 모드를 벗어날 때 guide_pause 초기화
        if cat != CAT_GUIDE:
            self.guide_paused = False
            self.set_guide_pause(False)

        # 진입 멘트 (버튼 입력 시에만; 내부 자동 전환은 announce=False)
        if announce:
            self.say(CAT_TTS.get(cat, ""))

        # 카테고리별 동작
        if cat == CAT_STOP:
            self._enter_stop()
        elif cat == CAT_FOLLOW:
            self._enter_follow()
        elif cat == CAT_GUIDE:
            self._enter_guide()
        elif cat == CAT_AUTO:
            self._enter_auto()

    def _enter_stop(self):
        pass

    def _enter_follow(self):
        pass

    def _enter_guide(self):
        # 안내 모드 진입: 대기 상태 초기화, 주행 가능 상태로
        now = time.time()
        self.guide_paused = False
        self.last_present_time = now
        self.last_absent_time = now
        self.last_agv_status = None   # 도착/시작 멘트 중복방지 초기화
        self.guide_started = True      # status=0 미수신 -> 안내 진입 즉시 후방판단 시작
        self.set_guide_pause(False)
        self.get_logger().info("[안내] 모드 진입")
        self.say("목적지로 안내를 시작합니다.")

    def _enter_auto(self):
        # 자율주행: AGV가 Nav2로 주행. fsm은 숫자 바이패스 + AGV_status 멘트만 담당.
        # (바이패스/follow_enable/guide_pause 초기화는 enter_mode에서 처리됨)
        self.last_agv_status = None   # 도착/출발 멘트 중복방지 초기화
        self.get_logger().info("[자율주행] 모드 진입")

    # ===== 유틸 =====
    def publish_heartbeat(self):
        hb = String()
        hb.data = json.dumps({
            "stamp": time.time(),
            "node": "fsm_node",
            "mode": CAT_NAMES.get(self.category, "?"),
            "status": "OK",
        })
        self.hb_pub.publish(hb)

    def bypass_to_agv(self, raw):
        # 핸드셰이크 시작: ack 받을 때까지 0.5초마다 반복 발행
        self.pending_agv = int(raw)
        self.agv_retry_count = 0
        self._publish_agv_mode(raw)   # 즉시 1회 발행
        self.get_logger().info(f"/agv_mode = {raw} 발행 시작 (ack 대기)")

    def _publish_agv_mode(self, raw):
        msg = Int32()
        msg.data = int(raw)
        self.agv_mode_pub.publish(msg)

    def agv_retry_loop(self):
        # 타이머: ack 대기 중이면 반복 발행
        if self.pending_agv is None:
            return
        self.agv_retry_count += 1
        if self.agv_retry_count > self.AGV_RETRY_MAX:
            self.get_logger().warn(
                f"/agv_mode = {self.pending_agv} ack 없음 "
                f"({self.AGV_RETRY_MAX}회 발행) -> 중단")
            self.pending_agv = None
            return
        self._publish_agv_mode(self.pending_agv)
        self.get_logger().info(
            f"/agv_mode = {self.pending_agv} 재발행 "
            f"({self.agv_retry_count}/{self.AGV_RETRY_MAX})")

    def on_arm_status(self, msg):
        # 로봇팔 작업 완료(1) -> 완료 멘트 + 픽업모드 종료 (3연발 방지, 1회만)
        if msg.data == 1:
            # 픽업모드일 때만 완료 멘트 + 종료 (오작동 방지)
            if self.category == CAT_PICKUP and not self.arm_done_said:
                self.arm_done_said = True
                self.say("픽업 완료!")
                self.get_logger().info("[픽업] 로봇팔 작업 완료 -> 완료 멘트")
                self.set_pickup_active(False)
                self.enter_mode(0, announce=False)
                self.get_logger().info("[픽업] 픽업모드 종료 -> 정지모드")
        elif msg.data == 0:
            self.arm_done_said = False

    def on_agv_mode_ack(self, msg):
        ack = msg.data
        # 픽업 좌표 락온 성공(52): pickup_target 송출 중단
        if ack == 52:
            self.set_pickup_active(False)
            self.get_logger().info("/AGV_mode_ack = 52 수신 -> pickup_target 송출 중단")
            return
        if self.pending_agv is not None and ack == self.pending_agv:
            self.get_logger().info(f"/AGV_mode_ack = {ack} 수신 -> 발행 중단")
            self.pending_agv = None
            # 픽업 좌표전송 단계(51) ack -> pickup_target 송출 시작
            if ack == 51:
                self.set_pickup_active(True)
                self.get_logger().info("/AGV_mode_ack = 51 수신 -> pickup_target 송출 시작")
    def set_pickup_active(self, active):
        msg = Bool()
        msg.data = bool(active)
        self.pickup_active_pub.publish(msg)

    def set_guide_pause(self, pause):
        msg = Bool()
        msg.data = bool(pause)
        self.guide_pause_pub.publish(msg)
        self.get_logger().info(f"/guide_pause = {pause}")

    def set_follow_enable(self, enable):
        msg = Bool()
        msg.data = bool(enable)
        self.follow_en_pub.publish(msg)
        self.get_logger().info(f"/follow_enable = {enable}")

    def say(self, text):
        if not text:
            return
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FsmNode()
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
