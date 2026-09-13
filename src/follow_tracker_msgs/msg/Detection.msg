# 검출된 사람 1명
int32 x          # bbox 좌상단 x (원본 픽셀)
int32 y          # bbox 좌상단 y
int32 w          # bbox 너비
int32 h          # bbox 높이
float32 conf     # 검출 신뢰도
float32 px       # 3D 좌표 X (m, 카메라 광학)
float32 py       # 3D 좌표 Y (m)
float32 pz       # 3D 좌표 Z (m, 전방 거리)
