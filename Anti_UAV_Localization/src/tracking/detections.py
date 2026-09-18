"""
세그멘테이션 확률맵 → 객체 후보(detection) 변환.

ThinDyUNet은 픽셀 확률맵만 출력하므로, 트래커가 쓸 객체 단위 후보를
연결성분(connected component)으로 만든다. 각 후보는 박스와 함께 **성분 픽셀 인덱스**를
보관하므로, 트래커가 고른 성분만 남긴 마스크를 그대로 출력할 수 있다.

- 고신뢰 후보(high): 확률맵을 high_thresh(기본 0.5, V2 평가와 동일)로 이진화한 연결성분.
- 저신뢰 후보(low): low_thresh로 이진화한 연결성분 중, 고신뢰 픽셀을 하나도 포함하지 않는 것.
  (ByteTrack식 2단 연관의 2차 후보)

좌표계: 확률맵은 원본 프레임 전체를 (H_in, W_in)=512×512로 리사이즈한 공간이다.
- box: 원본 해상도 좌표 [x, y, w, h]
- pix: 512×512 공간의 flat 픽셀 인덱스 (오름차순, uint32)
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(eq=False)   # numpy 필드가 있어 값 비교 대신 동일 객체 비교를 쓴다
class Detection:
    box: np.ndarray        # (4,) float [x, y, w, h], 원본 해상도 좌표
    score: float           # 성분 점수 (score_mode에 따라 max 또는 mean 확률, 0~1)
    area_px: int           # 확률맵(512×512) 공간에서의 성분 픽셀 수
    is_high: bool          # 고신뢰(high_thresh) 성분 여부
    pix: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32), repr=False)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.box[0] + self.box[2] / 2.0, self.box[1] + self.box[3] / 2.0])


def q_thresh(t: float) -> int:
    """확률 threshold t를 uint8 양자화(q = rint(p*255)) 공간 threshold로 변환.

    p >= t  ⇔  p*255 >= t*255  ⇔  rint(p*255) >= floor(t*255 + 0.5)  (반올림 경계 제외)
    """
    return int(np.floor(t * 255.0 + 0.5))


_q_thresh = q_thresh  # 하위 호환


def _components(
    binary: np.ndarray,
    prob_q: np.ndarray,
    score_mode: str,
    min_area: int,
) -> List[Tuple[np.ndarray, float, int, np.ndarray]]:
    """이진 마스크의 연결성분별 (bbox_in[x0,y0,w,h], score, area, pix_flat_idx) 목록."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return []

    flat_lab = labels.reshape(-1)
    fg = np.flatnonzero(flat_lab)                     # 오름차순
    lab = flat_lab[fg]
    vals = prob_q.reshape(-1)[fg].astype(np.float64) / 255.0

    if score_mode == "max":
        scores = np.zeros(n, dtype=np.float64)
        np.maximum.at(scores, lab, vals)
    elif score_mode == "mean":
        sums = np.bincount(lab, weights=vals, minlength=n)
        counts_ = np.bincount(lab, minlength=n)
        scores = sums / np.maximum(counts_, 1)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode} (max | mean)")

    # 성분별 픽셀 인덱스 분리 (stable 정렬 → 각 그룹 내부는 인덱스 오름차순 유지)
    order = np.argsort(lab, kind="stable")
    counts = np.bincount(lab, minlength=n)[1:]
    groups = np.split(fg[order].astype(np.uint32), np.cumsum(counts)[:-1])

    out = []
    for k in range(1, n):
        x0, y0, w, h, area = stats[k]
        if area < min_area:
            continue
        out.append((np.array([x0, y0, w, h], dtype=np.float64), float(scores[k]), int(area), groups[k - 1]))
    return out


def prob_to_detections(
    prob_q: np.ndarray,
    orig_size: Tuple[int, int],
    high_thresh: float = 0.5,
    low_thresh: float = 0.3,
    score_mode: str = "max",
    min_area: int = 1,
    use_low: bool = True,
) -> List[Detection]:
    """
    Args:
        prob_q: (H_in, W_in) uint8, 양자화된 sigmoid 확률 (rint(p*255)).
        orig_size: (W_orig, H_orig) 원본 프레임 크기.
        high_thresh: 고신뢰 성분 이진화 threshold (V2 평가 기본 0.5).
        low_thresh: 저신뢰 성분 이진화 threshold. use_low=False면 무시.
        score_mode: "max" | "mean" — 성분 내 확률 집계 방식.
        min_area: 512×512 공간에서 이 픽셀 수 미만 성분 제거.
    Returns:
        Detection 리스트 (고신뢰 먼저, 각 그룹 내 score 내림차순).
    """
    h_in, w_in = prob_q.shape
    w_orig, h_orig = orig_size
    sx, sy = w_orig / float(w_in), h_orig / float(h_in)
    scale = np.array([sx, sy, sx, sy])

    high_bin = prob_q >= q_thresh(high_thresh)
    dets: List[Detection] = []
    for bbox, score, area, pix in _components(high_bin, prob_q, score_mode, min_area):
        dets.append(Detection(bbox * scale, score, area, True, pix))

    if use_low and low_thresh < high_thresh:
        low_bin = prob_q >= q_thresh(low_thresh)
        high_flat = high_bin.reshape(-1)
        for bbox, score, area, pix in _components(low_bin, prob_q, score_mode, min_area):
            if high_flat[pix].any():
                continue  # 고신뢰 성분을 포함하는 저신뢰 성분은 중복이므로 제외
            dets.append(Detection(bbox * scale, score, area, False, pix))

    dets.sort(key=lambda d: (not d.is_high, -d.score))
    return dets
