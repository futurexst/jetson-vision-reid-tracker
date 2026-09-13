#!/bin/bash
# 팔로우미 로봇 전체 노드 기동 스크립트
# fsm_node + follow_node + rear_cam_node + tts_node 를 한 번에 실행

source ~/venv/bin/activate
source /opt/ros/humble/setup.bash
source ~/follow_ws/install/setup.bash
export PYTHONPATH=$PYTHONPATH:$HOME/venv/lib/python3.10/site-packages
export FASTDDS_LOG_LEVEL=OFF
export RCUTILS_LOGGING_USE_STDOUT=1

export ROS_DOMAIN_ID=5
# ===== boxmot 칼만 가드 상태 점검 (KF-GUARD) =====
# 정상: monkey-patch(follow_tracker/boxmot_patch.py)로만 적용.
# site-packages 직접 수정이 감지되면 경고 (이중 적용, venv 재생성 시 소실).
_KF="$HOME/venv/lib/python3.10/site-packages/boxmot/motion/kalman_filters/aabb/xysr_kf.py"
if [ -f "$_KF" ] && grep -q "if len(indices) < 2" "$_KF"; then
    echo -e "\033[93m[KF-GUARD] site-packages 직접 패치 감지 -> monkey-patch와 이중 적용 중\033[0m"
    echo -e "\033[93m[KF-GUARD] 원복 권장: cp ${_KF}.bak $_KF\033[0m"
fi

# ===== 블루투스 스피커 싱크 정상화 (BT-SINK) =====
# STONE SL-100 재연결 시 Mute=yes 로 붙는 경우가 있어 강제 해제.
# 미연결이면 조용히 건너뜀.
_BT_MAC="${BT_SPEAKER_MAC:-XX:XX:XX:XX:XX:XX}"  # 환경변수로 주입
_BT_SINK="bluez_sink.$(echo "$_BT_MAC" | tr ':' '_').a2dp_sink"
if pactl list sinks short 2>/dev/null | grep -q "$_BT_SINK"; then
    pactl set-sink-mute "$_BT_SINK" 0 2>/dev/null
    pactl set-default-sink "$_BT_SINK" 2>/dev/null
    echo -e "\033[92m[BT-SINK] $_BT_SINK 정상화 (unmute, default) - 음량은 수동 조절\033[0m"
else
    echo -e "\033[93m[BT-SINK] 블루투스 싱크 미연결 -> TTS는 온보드 출력\033[0m"
    echo -e "\033[93m[BT-SINK] 연결: bluetoothctl connect $_BT_MAC\033[0m"
fi

ros2 launch fsm_pkg all_nodes.launch.py
