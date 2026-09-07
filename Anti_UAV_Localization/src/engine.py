"""학습/검증 루프.

AMP와 gradient accumulation을 옵션으로 받아 모든 학습 스크립트가 공유한다.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import SegmentationMetrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    log_interval: int = 100,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    """1 epoch 학습 후 평균 loss 반환. scaler를 주면 AMP로 동작한다."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    def step():
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train", leave=False)
    batch_idx = -1
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                loss = criterion(model(images), masks) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            loss = criterion(model(images), masks) / accumulation_steps
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            step()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    # 마지막 잔여 gradient flush
    if batch_idx >= 0 and (batch_idx + 1) % accumulation_steps != 0:
        step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[float, dict]:
    """(평균 loss, 메트릭 dict) 반환."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metrics = SegmentationMetrics(threshold=threshold)

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                preds = model(images)
                loss = criterion(preds, masks)
        else:
            preds = model(images)
            loss = criterion(preds, masks)

        total_loss += loss.item()
        n_batches += 1
        # 메트릭은 fp32로 안전하게 계산
        metrics.update(preds.float(), masks)

    return total_loss / max(n_batches, 1), metrics.compute()
