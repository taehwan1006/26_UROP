"""
UAV Semantic Segmentation Dataset.
RGB/IR 이미지와 바이너리 마스크 쌍을 로딩한다.
"""

from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.image import IMAGENET_MEAN, IMAGENET_STD

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _matches_any(rel_path: str, patterns: Sequence[str]) -> bool:
    """시퀀스 상대경로가 glob 패턴 중 하나에 매칭되는지 확인.

    'video01' 처럼 경로 일부만 준 패턴도 매칭되도록 각 세그먼트와도 비교한다.
    """
    segments = rel_path.split("/")
    for pat in patterns:
        pat = pat.replace("\\", "/").rstrip("/")
        if fnmatch(rel_path, pat) or fnmatch(rel_path, f"{pat}/*"):
            return True
        if any(fnmatch(seg, pat) for seg in segments):
            return True
    return False


class UAVSegmentationDataset(Dataset):
    """
    UAV semantic segmentation dataset.

    실제 폴더 구조 (계층 깊이는 데이터셋마다 다를 수 있음):
        images/{split}/{sequence}/visible/visible-0000.jpg
        images/{split}/{sequence}/infrared/infrared-0000.jpg
        masks/{split}/{sequence}/visible/visible-0000_mask.png
        images/{split}/{sequence}/00001.jpg          (DUT 계열)
        masks/{split}/{sequence}/00001_mask.png
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        stride: int = 1,
        include_sequences: Optional[Sequence[str]] = None,
        exclude_sequences: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            stride: 시퀀스별 프레임 stride. 1=전체, 10=10%, 20=5% 등.
                    시퀀스 폴더 내에서 stride 간격으로 프레임 샘플링.
            include_sequences: 사용할 시퀀스 glob 패턴 목록 (예: ["video0*", "*/visible"]).
                    None이면 전체 사용.
            exclude_sequences: 제외할 시퀀스 glob 패턴 목록. include보다 우선한다.
            max_samples: 최종 샘플 수 상한. 전체에서 균등 간격으로 추린다.
            name: 로그 표시용 라벨.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.stride = max(1, stride)
        self.name = name or self.images_dir.name

        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir does not exist: {self.images_dir}")

        # 시퀀스 폴더 = 이미지 파일이 들어있는 leaf 디렉토리
        # (train은 {seq}/visible, DUT는 {seq} 자체가 leaf)
        leaf_dirs = set()
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                leaf_dirs.add(p.parent)

        self.sequences = []
        all_images = []
        for leaf_dir in sorted(leaf_dirs):
            rel = leaf_dir.relative_to(self.images_dir).as_posix()
            if exclude_sequences and _matches_any(rel, exclude_sequences):
                continue
            if include_sequences and not _matches_any(rel, include_sequences):
                continue

            seq_imgs = sorted(
                p for p in leaf_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
            )
            seq_imgs = seq_imgs[::self.stride]
            if seq_imgs:
                self.sequences.append(rel)
                all_images.extend(seq_imgs)

        # 대응하는 마스크가 존재하는 이미지만 필터링
        self.image_paths = [
            img_path for img_path in all_images
            if self._img_to_mask_path(img_path).exists()
        ]

        if max_samples is not None and 0 < max_samples < len(self.image_paths):
            step = len(self.image_paths) / max_samples
            self.image_paths = [
                self.image_paths[int(i * step)] for i in range(max_samples)
            ]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image/mask pairs found in {self.images_dir} "
                f"(stride={self.stride}, include={include_sequences}, "
                f"exclude={exclude_sequences})"
            )

        # 이미지 변환
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _img_to_mask_path(self, img_path: Path) -> Path:
        """
        이미지 경로 → 마스크 경로 변환.
        visible-0000.jpg → visible-0000_mask.png
        infrared-0000.jpg → infrared-0000_mask.png
        """
        rel_path = img_path.relative_to(self.images_dir)
        mask_name = img_path.stem + "_mask.png"
        mask_path = self.masks_dir / rel_path.parent / mask_name
        return mask_path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        rel_path = img_path.relative_to(self.images_dir)
        mask_path = self._img_to_mask_path(img_path)

        # 이미지 로딩
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")  # IR 1ch → RGB 3ch 변환

        # 마스크 로딩
        mask = Image.open(mask_path).convert("L")  # grayscale

        # 변환 적용
        image = self.img_transform(image)

        mask = self.mask_transform(mask)
        # 바이너리화: 0.5 이상이면 UAV(1), 아니면 배경(0)
        mask = (mask > 0.5).float()

        return image, mask
