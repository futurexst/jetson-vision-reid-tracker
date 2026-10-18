# jetson-vision-reid-tracker

Jetson Orin Nano 실기기에서 동작하는 사람 추종 로봇의 **Vision · FSM 파트**입니다.
RealSense D435 입력부터 검출, 추적, 타겟 재식별, 제어 명령 발행까지의 인지·판단 영역 전체를 담당합니다.

2026년 Physical AI 부트캠프(K-디지털 트레이닝 960시간) 최종 프로젝트 4인 팀 중 본인 담당 범위이며,
AGV 주행(Nav2) · 로봇팔 파지(MoveIt2) · 관제 대시보드는 팀원이 담당했습니다.

▶ **실증 영상 (1분 05초)** : [Person Re-ID 실증 — 가려져도 같은 사람을 찾는다 (YOLO11n · CLIP-ReID · Jetson Orin Nano)](https://youtu.be/ilEmrrfroBM)

---

## 문제와 해결

### 1. 시스템 루프가 목표의 절반에 그침

20Hz로 설정한 루프가 실측 11.1Hz(달성률 55%)에 머물렀습니다.
추측으로 모델을 줄이는 대신 구간별로 측정했습니다.

```
루프 90ms  −  추론 14.2ms  =  후처리 75.8ms
```

(추론 14.2ms는 15W 모드 실운용 로그의 중앙값입니다.)

병목은 모델이 아니라 후처리 구조에 있었습니다.
범용 트래커(BoxMOT/DeepOCSORT)가 타겟 1인만 필요한 상황에서도
모든 사람에게 외형 임베딩 추출을 매 프레임 수행하고 있었습니다.
(임베딩을 끄면 11 → 20Hz. 20Hz는 타이머 상한입니다.)

**해결** — 락·게이트 판정에 필요한 로직만 남긴 커스텀 IoU 트래커로 대체하고 C++로 재구현했습니다.

| 항목 | Python | C++ |
|---|---|---|
| 타이머 설정 | 20Hz | 30Hz |
| 실측 루프 | 11.1Hz | 29Hz |
| 달성률 | 55% | 96% |

추론은 YOLO11n을 TensorRT FP16으로 변환했습니다. 벤치마크 기준 32.2ms → 20.8ms(1.55배, 640×480 입력·50회 평균)입니다.
전력은 10Hz 운용 단계(Re-ID 적용 전, 전·후방 YOLO11n 2모델 + BoT-SORT)에서 15W 모드 기준 실측 VDD_IN 4.6W였습니다.

### 2. 임계값 하나로는 동일인과 타인을 가를 수 없었다

처음 사용한 CLIP-ReID(market1501) 가중치에서는 같은 옷을 입은 타인의 유사도가 0.66~0.75까지 올라
색상 게이트 면제 기준(0.65)을 넘어 락을 빼앗는 사례가 로그로 확인됐습니다.
실내 도메인에 더 맞는 msmt17 가중치로 교체한 뒤에도, 2인 동시 구간에서
본인 최저(0.59)가 타인 최고(0.68)보다 낮았습니다. 단일 임계값으로 자를 선이 존재하지 않았습니다.

**해결** — 점수와 게이트의 역할을 분리했습니다.

- **점수(판정자)** : CLIP-ReID 외형 임베딩 유사도
- **게이트(문지기)** : 방향 · 거리 · 색상이 후보를 사전 탈락

공간 게이트(방향·거리)를 최우선으로 두어, 임베딩 유사도가 높아도
위치가 어긋나면 타인으로 판정합니다. 옷 색이 같은 타인을 거부하기 위한 설계입니다.

```
G_PASS_STRONG   임베딩 유사도가 강함 → 색상 게이트 면제
G_PASS_COLOR    색상 게이트 통과
G_BLOCK_SIDE    나간 쪽 반대에서 등장 → 거부
G_BLOCK_DIST    마지막 거리와 차이가 큼 → 거부
G_BLOCK         임베딩·색상 모두 미달 → 거부
```

**결과** — 가림 이후 재락률 87.5%(32회 언락 중 28회 복귀),
2인 동시 구간 본인/타인 유사도 중앙값 0.99 / 0.21(msmt17).

설계 원칙은 **"잘못된 사람을 따라가느니 차라리 멈춘다"** 입니다.
구별이 안 되면 추종을 포기하는 쪽을 택했습니다.

### 3. 분산 환경에서 통신이 어디서 끊겼는지 알 수 없었다

ROS2 배포판과 DDS 구현이 서로 다른 3대 장비를 domain_bridge로 연결했는데,
통신 단절 시 구간을 특정할 수 없었습니다.

**해결** — 진단 도구(`comm_monitor_node`)를 직접 만들어
토픽 Hz·지터, 노드 하트비트, 장비 Ping, 시스템 자원을 한 화면에 통합했습니다.

이 도구로 실제 결함을 찾았습니다.

- **Wi-Fi 절전모드** : AGV RTT 2,446ms · 손실 33%를 특정, 절전 해제 후 3.4ms · 0%
- **브리지 구독 누락** : publisher보다 domain_bridge가 먼저 기동해 구독자가 0이 되던 레이스를 찾아 systemd `ExecStartPre` 지연으로 해결

---

## 구조

```
src/
├── follow_tracker/          Python 구현 (V1/V2)
│   └── follow_node.py       검출 · 추적 · 재식별 · 3중 게이트 · 제어 명령
├── follow_tracker_cpp/      C++ 재구현 (V3)
│   ├── camera_infer_node    RealSense 입력 + YOLO11n TensorRT 직접 호출
│   ├── iou_tracker          커스텀 IoU 트래커 (범용 트래커 대체)
│   ├── reid_engine          CLIP-ReID TensorRT 엔진 래퍼
│   └── tracker_node         게이트 판정 · 락 관리 · 블랙리스트
├── follow_tracker_msgs/     Detection / DetectionSet 메시지 정의
├── fsm_pkg/
│   ├── fsm_node             모드 번호 단일 통제 지점 · ack 핸드셰이크
│   ├── comm_monitor_node    분산 통신 진단 도구
│   └── rear_cam_node        후방 감시
├── pickup_pkg/              카메라→베이스 좌표 변환 · 도킹/파지 좌표 산출
├── tts_pkg/                 edge-tts 캐시 기반 음성 출력
└── proto/                   초기 검증 스크립트
```

### FSM 모드 체계

십의 자리가 카테고리, 일의 자리가 목적지입니다.

```
0    정지
1x   추종
2x   안내
3x   자율주행
4x   위치 재조정
5x   픽업
```

모든 모드 전환이 `fsm_node` 한 곳을 지나므로 상태 추적과 디버깅이 가능합니다.
모드 발행 후 응답 수신까지 0.5초 주기로 최대 10회 재발행하는 ack 핸드셰이크를 두어,
무선 패킷이 유실되어도 모드가 어긋나지 않습니다.

---

## 환경

| 항목 | 내용 |
|---|---|
| 보드 | Jetson Orin Nano 8GB |
| 카메라 | Intel RealSense D435 (RGB-D, 전방 추종 + 후방 감시 겸용) |
| ROS2 | Humble · Fast-DDS · Domain 5 |
| 검출 | YOLO11n TensorRT FP16 |
| 재식별 | CLIP-ReID (msmt17) TensorRT FP16 |
| 연동 장비 | myAGV 2023 (Galactic · Domain 0), myCobot 280 Pi, Web Dashboard (Jazzy) |

배포판과 DDS가 서로 다른 장비를 `domain_bridge`(Domain 5 ↔ 0, 브리지 토픽 9종)로 연결하고
systemd 서비스로 부팅 시 자동 기동하도록 구성했습니다.

---

## 빌드

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch fsm_pkg all_nodes_cpp.launch.py   # C++ 버전
ros2 launch fsm_pkg all_nodes.launch.py       # Python 버전
```

## 모델 가중치

용량 문제로 `.pt` / `.onnx` / `.engine` 파일은 저장소에 포함하지 않았습니다.
TensorRT 엔진은 실행 환경에서 직접 변환해야 합니다(엔진 파일은 GPU·TensorRT 버전에 종속).

```bash
# YOLO11n → ONNX → TensorRT
yolo export model=yolo11n.pt format=onnx opset=12
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine --fp16
```

변환한 엔진은 각 패키지의 `models/` 경로에 배치합니다.

---

## 담당 범위

본인이 작성한 ROS2 노드 5종입니다.

- `follow_node` — 추종 · 후방 감시 · 레이저 인식 · 재식별
- `fsm_node` — 모드 판단 · 바이패스 · ack 핸드셰이크
- `pickup_node` — 좌표 변환 · 도킹/파지 좌표 산출
- `tts_node` — 음성 출력
- `comm_monitor_node` — 토픽 · 노드 · 네트워크 실시간 진단

여기에 C++ 재구현(`follow_tracker_cpp`), 픽업 9단계 핸드셰이크 시퀀스 설계,
`domain_bridge` systemd 서비스화가 포함됩니다.

AGV 주행(Nav2), 로봇팔 파지(MoveIt2), 관제 대시보드는 팀원 담당입니다.
