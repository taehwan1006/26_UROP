"""체크포인트 저장/로드."""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    scheduler=None,
    scaler=None,
    val_metrics: Optional[dict] = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "val_metrics": val_metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    """가중치만 로드한다 (파인튜닝/평가용). 원본 체크포인트 dict를 돌려준다.

    학습 스크립트가 저장한 전체 dict와 state_dict만 담긴 파일을 모두 받는다.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return ckpt


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_path: Path,
    device: torch.device,
    scheduler=None,
    scaler=None,
) -> Tuple[int, float]:
    """가중치 + optimizer/scheduler 상태까지 복원. (다음 epoch, best_val_loss) 반환."""
    ckpt = load_weights(model, ckpt_path, device)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"] + 1, ckpt.get("best_val_loss", float("inf"))
