"""
픽셀 단위 세그멘테이션 지표 (희소 인덱스 기반).

utils/metrics.py::SegmentationMetrics 와 **같은 정의**를 numpy로 구현한다.
(평가 해상도 512×512, GT 마스크는 NEAREST 리사이즈 후 >0.5 이진화, 예측은 threshold 이진화)

- precision / recall / dice / uav_iou / bg_iou / miou_pixel : 전 프레임 픽셀 풀링(micro)
- uav_iou_per_image / miou_per_image : 이미지별 값의 평균. GT·예측이 모두 비면 UAV IoU = 1

추가 진단 항목 (SegmentationMetrics에는 없음):
- absent_frames_with_pred : GT 마스크가 빈 프레임 중 예측 픽셀이 하나라도 있는 프레임 수
- absent_fp_pixels        : GT 마스크가 빈 프레임에서 나온 FP 픽셀 수 (전체 FP 중 비중 확인용)
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class PixelMetrics:
    def __init__(self, n_pixels_per_image: int):
        self.N = int(n_pixels_per_image)
        self.tp = self.fp = self.fn = self.tn = 0
        self.uav_iou_sum = 0.0
        self.miou_sum = 0.0
        self.n_images = 0
        self.absent_frames = 0
        self.absent_frames_with_pred = 0
        self.absent_fp_pixels = 0

    def update(self, pred_idx: np.ndarray, gt_idx: np.ndarray) -> float:
        """pred_idx, gt_idx: 오름차순 유일 flat 인덱스. 반환: 이 프레임 UAV IoU."""
        n_p, n_g = len(pred_idx), len(gt_idx)
        tp = int(np.intersect1d(pred_idx, gt_idx, assume_unique=True).size) if (n_p and n_g) else 0
        fp = n_p - tp
        fn = n_g - tp
        tn = self.N - tp - fp - fn
        self.tp += tp; self.fp += fp; self.fn += fn; self.tn += tn

        ud = tp + fp + fn
        uav = tp / ud if ud > 0 else 1.0
        bd = tn + fp + fn
        bg = tn / bd if bd > 0 else 1.0
        self.uav_iou_sum += uav
        self.miou_sum += (uav + bg) / 2.0
        self.n_images += 1

        if n_g == 0:
            self.absent_frames += 1
            if n_p > 0:
                self.absent_frames_with_pred += 1
                self.absent_fp_pixels += fp
        return uav

    def merge(self, other: "PixelMetrics") -> None:
        for k in ("tp", "fp", "fn", "tn", "uav_iou_sum", "miou_sum", "n_images",
                  "absent_frames", "absent_frames_with_pred", "absent_fp_pixels"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        uav_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        bg_iou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
        n = max(self.n_images, 1)
        return {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            "uav_iou": uav_iou,
            "bg_iou": bg_iou,
            "miou_pixel": (uav_iou + bg_iou) / 2.0,
            "uav_iou_per_image": self.uav_iou_sum / n,
            "miou_per_image": self.miou_sum / n,
            "n_images": self.n_images,
            "tp": tp, "fp": fp, "fn": fn,
            "absent_frames": self.absent_frames,
            "absent_frames_with_pred": self.absent_frames_with_pred,
            "absent_fp_pixel_share": self.absent_fp_pixels / fp if fp > 0 else 0.0,
        }


# ---------------------------------------------------------------------- #
# GT 마스크 캐시
# ---------------------------------------------------------------------- #
GT_CACHE_VERSION = 1


def _load_mask_idx(path: Path, img_size_hw: Tuple[int, int]) -> np.ndarray:
    """UAVSegmentationDataset.mask_transform과 동일: L 변환 → Resize(NEAREST) → /255 > 0.5."""
    from PIL import Image

    h, w = img_size_hw
    m = Image.open(path).convert("L")
    if m.size != (w, h):
        m = m.resize((w, h), Image.NEAREST)   # torchvision Resize(PIL)와 같은 호출
    arr = np.asarray(m)
    return np.flatnonzero(arr.reshape(-1) > 127).astype(np.uint32)   # v/255 > 0.5 ⇔ v >= 128


def build_or_load_gt_masks(
    cache_path: Path,
    masks_dir: Path,
    rel: str,
    frames: Sequence[Path],
    img_size_hw: Tuple[int, int],
    num_threads: int = 4,
    refresh: bool = False,
) -> Tuple[List[Optional[np.ndarray]], int]:
    """시퀀스 GT 마스크를 희소 인덱스 목록으로 반환. 마스크 파일이 없는 프레임은 None.

    `<stem>_mask.png` 규칙은 UAVSegmentationDataset._img_to_mask_path 와 같다.
    반환: (프레임별 인덱스 목록, 마스크 없는 프레임 수)
    """
    names = [f.name for f in frames]
    if cache_path.exists() and not refresh:
        z = np.load(cache_path, allow_pickle=False)
        if (int(z["version"]) == GT_CACHE_VERSION and tuple(z["in_hw"]) == tuple(img_size_hw)
                and [str(s) for s in z["frame_names"]] == names):
            idx, off, has = z["idx"], z["offsets"], z["has_mask"]
            out = [idx[off[i]:off[i + 1]] if has[i] else None for i in range(len(names))]
            return out, int((~has).sum())

    from concurrent.futures import ThreadPoolExecutor

    mask_paths = [masks_dir / rel / (f.stem + "_mask.png") for f in frames]

    def _one(p: Path):
        return _load_mask_idx(p, img_size_hw) if p.exists() else None

    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as ex:
        out = list(ex.map(_one, mask_paths))

    has = np.array([o is not None for o in out], dtype=bool)
    counts = [len(o) if o is not None else 0 for o in out]
    offsets = np.zeros(len(out) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        idx=np.concatenate([o for o in out if o is not None]) if has.any() else np.zeros(0, np.uint32),
        offsets=offsets, has_mask=has, frame_names=np.asarray(names),
        in_hw=np.asarray(img_size_hw, dtype=np.int32), version=np.int32(GT_CACHE_VERSION),
        masks_dir=np.asarray(str(masks_dir)),
    )
    tmp.replace(cache_path)
    return out, int((~has).sum())
