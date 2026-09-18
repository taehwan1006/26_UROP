"""
시퀀스별 세그멘테이션 확률맵 캐시.

트래커 ablation은 하이퍼파라미터만 바꿔 여러 번 돌리므로, 네트워크 추론은 한 번만 하고
확률맵을 저장해 재사용한다. 512×512 uint8을 그대로 저장하면 24,804장 기준 약 6.5GB이므로,
확률 >= min_prob 인 픽셀만 (flat index, uint8 값) 희소 형식으로 저장한다.

주의: min_prob 미만 픽셀은 0으로 복원되므로, 캐시로 트래킹할 때 low_thresh >= min_prob 이어야 한다.
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

CACHE_VERSION = 1


class SparseProbWriter:
    def __init__(self, in_hw: Tuple[int, int], min_prob: float):
        self.in_hw = tuple(int(x) for x in in_hw)
        self.min_prob = float(min_prob)
        self.min_q = int(np.floor(min_prob * 255.0 + 0.5))
        self._idx: List[np.ndarray] = []
        self._val: List[np.ndarray] = []
        self._counts: List[int] = []

    def add(self, prob_q: np.ndarray) -> None:
        assert prob_q.shape == self.in_hw and prob_q.dtype == np.uint8, (prob_q.shape, prob_q.dtype)
        flat = prob_q.reshape(-1)
        idx = np.flatnonzero(flat >= self.min_q).astype(np.uint32)
        self._idx.append(idx)
        self._val.append(flat[idx])
        self._counts.append(len(idx))

    def save(self, path: Path, frame_names: Sequence[str], orig_sizes: np.ndarray, meta: dict) -> None:
        assert len(frame_names) == len(self._counts) == len(orig_sizes)
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.zeros(len(self._counts) + 1, dtype=np.int64)
        np.cumsum(self._counts, out=offsets[1:])
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            idx=np.concatenate(self._idx) if self._idx else np.zeros(0, np.uint32),
            val=np.concatenate(self._val) if self._val else np.zeros(0, np.uint8),
            offsets=offsets,
            orig_sizes=np.asarray(orig_sizes, dtype=np.int32),
            frame_names=np.asarray(list(frame_names)),
            in_hw=np.asarray(self.in_hw, dtype=np.int32),
            min_prob=np.float64(self.min_prob),
            version=np.int32(CACHE_VERSION),
            meta_keys=np.asarray(list(meta.keys())),
            meta_vals=np.asarray([str(v) for v in meta.values()]),
        )
        tmp.replace(path)  # 중간에 끊겨도 깨진 캐시가 남지 않도록


class SparseProbReader:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        if int(z["version"]) != CACHE_VERSION:
            raise ValueError(f"cache version mismatch: {path}")
        self.idx = z["idx"]
        self.val = z["val"]
        self.offsets = z["offsets"]
        self.orig_sizes = z["orig_sizes"]
        self.frame_names = [str(s) for s in z["frame_names"]]
        self.in_hw = tuple(int(x) for x in z["in_hw"])
        self.min_prob = float(z["min_prob"])
        self.meta = dict(zip([str(k) for k in z["meta_keys"]], [str(v) for v in z["meta_vals"]]))

    def __len__(self) -> int:
        return len(self.frame_names)

    def get_sparse(self, i: int):
        """(flat 인덱스 오름차순 uint32, uint8 값) — 밀집 복원 없이 threshold 적용할 때 사용."""
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.idx[s:e], self.val[s:e]

    def get(self, i: int) -> np.ndarray:
        s, e = self.offsets[i], self.offsets[i + 1]
        flat = np.zeros(self.in_hw[0] * self.in_hw[1], dtype=np.uint8)
        flat[self.idx[s:e]] = self.val[s:e]
        return flat.reshape(self.in_hw)
