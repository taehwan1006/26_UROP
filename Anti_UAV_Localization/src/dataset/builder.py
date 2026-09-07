"""
Config 기반 데이터셋 구성 헬퍼.

여러 데이터셋 소스를 합치거나(ConcatDataset), 한 데이터셋에서 특정 시퀀스만
골라 쓰기 위한 진입점. 학습/평가 스크립트는 이 함수만 호출하면 된다.

Config 예시:

    data:
      img_size: [512, 512]
      num_workers: 4
      sources:
        train:
          - name: dut_tracking
            images_dir: "data/raw/images/train_dut"
            masks_dir: "data/raw/masks/train_dut"
            stride: 1
          - name: dut_detection
            images_dir: "data/detection_raw/images/train"
            masks_dir: "data/detection_raw/masks/train"
        test_dut:
          - name: dut_hard_seqs
            images_dir: "data/raw/images/test_dut"
            masks_dir: "data/raw/masks/test_dut"
            include_sequences: ["video0*"]
            exclude_sequences: ["video07"]
            max_samples: 2000

`sources`에 해당 split이 없으면 기존 방식(images_root/masks_root + split 이름,
`{split}_stride`)으로 폴백하므로 예전 config도 그대로 동작한다.
"""

from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import ConcatDataset, Dataset

from .uav_dataset import UAVSegmentationDataset


def _resolve(project_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def _legacy_sources(data_cfg: dict, split: str) -> list:
    """`sources` 미정의 config를 위한 단일 소스 폴백."""
    images_root = data_cfg.get("images_root", "data/raw/images")
    masks_root = data_cfg.get("masks_root", "data/raw/masks")
    return [{
        "name": split,
        "images_dir": f"{images_root}/{split}",
        "masks_dir": f"{masks_root}/{split}",
        "stride": data_cfg.get(f"{split}_stride", 1),
    }]


def build_split_dataset(
    data_cfg: dict,
    split: str,
    project_root: Path,
    overrides: Optional[dict] = None,
    verbose: bool = True,
) -> Dataset:
    """split 이름에 해당하는 데이터셋을 구성한다. 소스가 여럿이면 ConcatDataset.

    overrides: CLI 등에서 온 값(stride/include_sequences/...)을 모든 소스에 덮어쓴다.
               None인 항목은 무시하므로 부분 override가 가능하다.
    """
    sources = (data_cfg.get("sources") or {}).get(split)
    if not sources:
        sources = _legacy_sources(data_cfg, split)

    img_size: Tuple[int, int] = tuple(data_cfg["img_size"])
    default_stride = data_cfg.get(f"{split}_stride", 1)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    parts = []
    for i, src in enumerate(sources):
        src = {**src, **overrides}
        parts.append(UAVSegmentationDataset(
            images_dir=str(_resolve(project_root, src["images_dir"])),
            masks_dir=str(_resolve(project_root, src["masks_dir"])),
            img_size=img_size,
            stride=src.get("stride", default_stride),
            include_sequences=src.get("include_sequences"),
            exclude_sequences=src.get("exclude_sequences"),
            max_samples=src.get("max_samples"),
            name=src.get("name", f"{split}[{i}]"),
        ))

    if verbose:
        for ds in parts:
            print(
                f"  [{split}] {ds.name}: {len(ds):,} samples "
                f"({len(ds.sequences)} seqs, stride={ds.stride})"
            )

    dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    if verbose:
        print(f"  [{split}] total: {len(dataset):,} samples")
    return dataset
