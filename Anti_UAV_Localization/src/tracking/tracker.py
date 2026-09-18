"""
단일 표적(SOT) 트래커 — 세그멘테이션 후보 기반.

DUT-Anti-UAV Tracking 서브셋(test_dut)은 시퀀스당 드론 1기이므로, 다중 트랙을 관리하는
ByteTrack 전체 대신 "트랙 1개 + 전역 재검출" 구조로 구현한다. ByteTrack의 핵심인
고/저신뢰 2단 연관은 그대로 가져온다.

모드
- per_frame : 시간 정보 없음. 매 프레임 가장 좋은 고신뢰 성분 1개를 출력 (세그멘테이션 단독 기준선).
- kalman    : Kalman 예측 + 게이팅 + (선택) 저신뢰 2차 연관 + 가림 구간 coasting + 재초기화.

첫 프레임 GT 초기화는 사용하지 않는다(검출 기반 자동 초기화).
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from .detections import Detection
from .kalman import BoxKalmanFilter, cxcywh_to_xywh, xywh_to_cxcywh


@dataclass
class TrackerConfig:
    mode: str = "kalman"            # "per_frame" | "kalman"
    select: str = "score"           # 후보 1개 선택 기준 (초기화·per_frame): "score" | "area"
    init_thresh: float = 0.5        # 트랙 초기화/재초기화에 필요한 최소 성분 점수 (고신뢰 후보만)
    use_low: bool = True            # 저신뢰 후보 2차 연관 (ByteTrack식)
    gate: bool = True               # False면 게이트 없이 가장 가까운 후보와 연관
    gate_chi2: float = 9.21         # Mahalanobis² 게이트 임계값 (gate_dims=2면 자유도 2의 99% = 9.21, gate_dims=4면 자유도 4 권장 13.28)
    gate_dims: int = 2              # 연관에 쓸 상태 차원: 2=중심만, 4=중심+크기(크기 연속성까지 봄)
    gate_min_radius: float = 0.0    # px. Mahalanobis 게이트 밖이어도 이 거리 이내면 허용 (0=끔)
    max_coast: int = 10             # 미연관 시 예측 박스를 출력할 최대 연속 프레임 수 (0=출력 안 함)
    max_age: int = 30               # 이 프레임 수를 넘게 미연관이면 트랙 소멸(LOST)
    reinit_after: int = 3           # 연속 미연관 이 횟수 이상이면 게이트 밖 고신뢰 후보로 재초기화
    min_hits: int = 1               # 트랙 확정 전(hits < min_hits)에는 출력하지 않음
    output_box: str = "det"         # 연관 성공 시 출력 박스: "det"(성분 박스) | "kf"(보정된 상태)
    mask_mode: str = "selected"     # 마스크 출력 범위: "selected"(고른 성분 1개) | "expand"(출력 박스를 mask_expand배 확대한 영역과 겹치는 성분 전부)
    mask_expand: float = 1.5        # mask_mode="expand"에서 박스 확대 배율 (드론 마스크가 여러 성분으로 쪼개지는 경우 대비)
    coast_output: str = "box"       # 가림 구간(coast) 출력: "box"(박스만, 마스크 없음 — 기존 동작) | "none"(아무것도 출력 안 함) | "mask"(박스 + 예측 영역과 겹치는 성분 마스크)
    kf_std_pos: float = 1.0 / 20
    kf_std_vel: float = 1.0 / 160
    kf_min_size: float = 8.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameOutput:
    box: Optional[np.ndarray]   # [x, y, w, h] 원본 좌표, 없으면 None (= 드론 없음 판단)
    score: float
    status: str                 # none | per_frame | init | high | low | coast | coast_none | miss | reinit | lost | tentative
    det: Optional[Detection] = None            # 연관/선택된 대표 성분 (없으면 None)
    mask_dets: List[Detection] = field(default_factory=list)  # 마스크로 출력할 성분 목록 (고신뢰·저신뢰 섞일 수 있음)


def _expand_box(box: np.ndarray, factor: float) -> np.ndarray:
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    w, h = box[2] * factor, box[3] * factor
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h])


def _intersecting(dets: List[Detection], box: np.ndarray) -> List[Detection]:
    """박스와 한 픽셀이라도 겹치는 성분 목록."""
    x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
    out = []
    for d in dets:
        b = d.box
        if b[0] < x1 and b[0] + b[2] > x0 and b[1] < y1 and b[1] + b[3] > y0:
            out.append(d)
    return out


def _pick(dets: List[Detection], select: str) -> Detection:
    if select == "score":
        return max(dets, key=lambda d: (d.score, d.area_px))
    if select == "area":
        return max(dets, key=lambda d: (d.area_px, d.score))
    raise ValueError(f"Unknown select: {select}")


class SingleTargetTracker:
    def __init__(self, cfg: TrackerConfig):
        if cfg.mode not in ("per_frame", "kalman"):
            raise ValueError(f"Unknown mode: {cfg.mode}")
        if cfg.output_box not in ("det", "kf"):
            raise ValueError(f"Unknown output_box: {cfg.output_box}")
        if cfg.mask_mode not in ("selected", "expand"):
            raise ValueError(f"Unknown mask_mode: {cfg.mask_mode}")
        if cfg.coast_output not in ("box", "none", "mask"):
            raise ValueError(f"Unknown coast_output: {cfg.coast_output}")
        if cfg.gate_dims not in (2, 4):
            raise ValueError(f"Unknown gate_dims: {cfg.gate_dims} (2 | 4)")
        self.cfg = cfg
        self.kf = BoxKalmanFilter(cfg.kf_std_pos, cfg.kf_std_vel, cfg.kf_min_size)
        self.reset()

    def _mask_for(self, box: np.ndarray, primary: Optional[Detection], dets: List[Detection]) -> List[Detection]:
        """출력 마스크에 넣을 성분 목록. selected면 대표 성분만, expand면 확대 박스와 겹치는 성분 전부."""
        if self.cfg.mask_mode == "selected":
            return [primary] if primary is not None else []
        picked = _intersecting(dets, _expand_box(box, self.cfg.mask_expand))
        if primary is not None and not any(p is primary for p in picked):
            picked.append(primary)
        return picked

    def reset(self):
        """시퀀스가 바뀔 때 반드시 호출."""
        self.mean = None
        self.cov = None
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ #
    def _init_track(self, det: Detection, status: str, dets: List[Detection]) -> FrameOutput:
        self.mean, self.cov = self.kf.initiate(xywh_to_cxcywh(det.box))
        self.hits, self.misses = 1, 0
        if self.hits < self.cfg.min_hits:
            return FrameOutput(None, det.score, "tentative")
        return FrameOutput(det.box.copy(), det.score, status, det, self._mask_for(det.box, det, dets))

    def _associate(self, pool: List[Detection]) -> Optional[Detection]:
        if not pool:
            return None
        centers = np.stack([d.center for d in pool])
        if self.cfg.gate_dims == 4:
            d2 = self.kf.gating_distance_box(self.mean, self.cov,
                                             np.stack([xywh_to_cxcywh(d.box) for d in pool]))
        else:
            d2 = self.kf.gating_distance_center(self.mean, self.cov, centers)
        if self.cfg.gate:
            eu = np.linalg.norm(centers - self.mean[:2], axis=1)
            ok = d2 <= self.cfg.gate_chi2
            if self.cfg.gate_min_radius > 0:
                ok |= eu <= self.cfg.gate_min_radius
            if not np.any(ok):
                return None
            d2 = np.where(ok, d2, np.inf)
        return pool[int(np.argmin(d2))]

    # ------------------------------------------------------------------ #
    def step(self, dets: List[Detection]) -> FrameOutput:
        cfg = self.cfg
        high = [d for d in dets if d.is_high]
        low = [d for d in dets if not d.is_high]

        if cfg.mode == "per_frame":
            if not high:
                return FrameOutput(None, 0.0, "none")
            best = _pick(high, cfg.select)
            return FrameOutput(best.box.copy(), best.score, "per_frame", best,
                               self._mask_for(best.box, best, dets))

        init_pool = [d for d in high if d.score >= cfg.init_thresh]

        # 트랙 없음 → 전역 탐색으로 초기화
        if self.mean is None:
            if init_pool:
                return self._init_track(_pick(init_pool, cfg.select), "init", dets)
            return FrameOutput(None, 0.0, "none")

        # 예측
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

        # 1차: 고신뢰, 2차: 저신뢰
        matched, status = self._associate(high), "high"
        if matched is None and cfg.use_low:
            matched, status = self._associate(low), "low"

        if matched is not None:
            self.mean, self.cov = self.kf.update(self.mean, self.cov, xywh_to_cxcywh(matched.box))
            self.hits += 1
            self.misses = 0
            if self.hits < cfg.min_hits:
                return FrameOutput(None, matched.score, "tentative")
            box = matched.box.copy() if cfg.output_box == "det" else cxcywh_to_xywh(self.mean[:4])
            return FrameOutput(box, matched.score, status, matched, self._mask_for(box, matched, dets))

        # 미연관
        self.misses += 1
        if self.misses >= cfg.reinit_after and init_pool:
            return self._init_track(_pick(init_pool, cfg.select), "reinit", dets)
        if self.misses > cfg.max_age:
            self.reset()
            return FrameOutput(None, 0.0, "lost")
        if self.misses <= cfg.max_coast and self.hits >= cfg.min_hits:
            pbox = cxcywh_to_xywh(self.mean[:4])
            if cfg.coast_output == "none":
                return FrameOutput(None, 0.0, "coast_none")
            if cfg.coast_output == "mask":
                picked = _intersecting(dets, _expand_box(pbox, cfg.mask_expand))
                if not picked:
                    # 되살릴 성분이 없으면 박스도 내지 않는다 (박스 지표와 픽셀 지표의 기준을 일치시킴)
                    return FrameOutput(None, 0.0, "coast_none")
                return FrameOutput(pbox, 0.0, "coast", None, picked)
            return FrameOutput(pbox, 0.0, "coast")   # 박스만, 마스크는 비움 (기존 동작)
        return FrameOutput(None, 0.0, "miss")   # 트랙은 유지하지만 출력하지 않음
