"""BoxMOT 런타임 패치 모음.

venv 재생성 시 site-packages 직접 수정분이 소실되므로,
필요한 가드를 import 시점에 monkey-patch로 적용한다.
follow_node.py에서 create_tracker() 호출 전에 import할 것.
"""


def _patch_xysr_kf_unfreeze():
    """KalmanFilterXYSR.unfreeze()의 IndexError 가드.

    트랙 생애 실관측이 1개 이하일 때(락 직후 완전 가림)
    `indices[-2]`에서 IndexError 발생. 해당 지점은 상태 복원
    (__dict__ / history_obs) 완료 이후이므로 예외를 무시해도
    궤적 품질 손실이 없다. (max_age=300 설정에서 발생 확률 상승)
    """
    from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR

    if getattr(KalmanFilterXYSR.unfreeze, "_ws_patched", False):
        return "already"

    _orig_unfreeze = KalmanFilterXYSR.unfreeze

    def _safe_unfreeze(self):
        try:
            return _orig_unfreeze(self)
        except IndexError:
            return None

    _safe_unfreeze._ws_patched = True
    KalmanFilterXYSR.unfreeze = _safe_unfreeze
    return "patched"


def apply_all():
    """모든 패치 적용. 결과를 dict로 반환."""
    results = {}
    try:
        results["xysr_kf.unfreeze"] = _patch_xysr_kf_unfreeze()
    except Exception as e:
        results["xysr_kf.unfreeze"] = f"FAILED: {e}"
    return results
