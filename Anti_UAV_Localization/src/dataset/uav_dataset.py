"""
UAV Semantic Segmentation Dataset.
RGB/IR 이미지와 바이너리 마스크 쌍을 로딩한다.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UAVSegmentationDataset(Dataset):
    """
    UAV semantic segmentation dataset.

    실제 폴더 구조:
        images/{split}/{sequence}/visible/visible-0000.jpg
        images/{split}/{sequence}/infrared/infrared-0000.jpg
        masks/{split}/{sequence}/visible/visible-0000_mask.png
        masks/{split}/{sequence}/infrared/infrared-0000_mask.png
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        stride: int = 1,
    ):
        """
        Args:
            stride: 시퀀스별 프레임 stride. 1=전체, 10=10%, 20=5% 등.
                    시퀀스 폴더 내에서 stride 간격으로 프레임 샘플링.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.stride = max(1, stride)

        # 이미지 파일 목록 수집 (재귀적으로 모든 이미지 탐색)
        # .json, .mp4 등 비이미지 파일 제외
        img_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        # 시퀀스별로 그룹화하여 stride 적용
        # 예: train/seq1/visible/, train/seq1/infrared/ 가 시퀀스 단위
        # 시퀀스 폴더 = 이미지 파일이 들어있는 leaf 디렉토리 (visible / infrared)
        leaf_dirs = set()
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in img_extensions:
                leaf_dirs.add(p.parent)

        all_images = []
        for leaf_dir in sorted(leaf_dirs):
            seq_imgs = sorted(
                p for p in leaf_dir.iterdir()
                if p.is_file() and p.suffix.lower() in img_extensions
            )
            # stride 간격 샘플링
            seq_imgs = seq_imgs[::self.stride]
            all_images.extend(seq_imgs)

        # 대응하는 마스크가 존재하는 이미지만 필터링
        self.image_paths = []
        for img_path in all_images:
            mask_path = self._img_to_mask_path(img_path)
            if mask_path.exists():
                self.image_paths.append(img_path)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}"
            )

        # 이미지 변환
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
