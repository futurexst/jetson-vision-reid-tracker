"""Re-ID 임계값 캘리브레이션 전용 CSV 로거.

환경변수 REID_CALIB 가 비어있지 않을 때만 활성화된다.
  REID_CALIB=A runrobot     -> ~/reid_calib/reid_calib_A_<ts>.csv
평상시 실행에는 아무 영향이 없다. 캘리브레이션 종료 후 제거 예정.
"""
import csv
import os
import time

CALIB_DIR = os.path.expanduser("~/reid_calib")

HEADER = [
    "ts", "frame", "phase", "n_people",
    "locked_id", "tid", "is_target",
    "osnet", "color",
    "dist", "cx", "last_dist", "last_seen_side",
    "side_block", "dist_block", "gate",
]


def tag():
    """활성화 태그 문자열. 비활성이면 None."""
    v = os.environ.get("REID_CALIB", "").strip()
    return v if v else None


class CalibLogger(object):
    def __init__(self, tag_str):
        os.makedirs(CALIB_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            CALIB_DIR, "reid_calib_%s_%s.csv" % (tag_str, stamp))
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(HEADER)
        self._f.flush()
        self.rows = 0

    def log(self, **kw):
        self._w.writerow([kw.get(k, "") for k in HEADER])
        self.rows += 1
        if self.rows % 20 == 0:
            self._f.flush()

    def close(self):
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
