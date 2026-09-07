"""이미지 정규화 상수와 역변환.

학습 시 Normalize와 시각화 시 denorm이 같은 값을 쓰도록 여기 한 곳에서 관리한다.
"""

import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denorm(image: torch.Tensor) -> np.ndarray:
    """정규화된 (3, H, W) 텐서 → 표시용 (H, W, 3) [0, 1] 배열."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image.cpu().float() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()
