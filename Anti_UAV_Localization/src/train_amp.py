"""
ThinDyUNet 학습 스크립트 (AMP/mixed precision 버전).
원본 train.py에서 torch.amp.autocast + GradScaler를 적용해 30~50% 학습 가속.

사용:
    python src/train_amp.py --config configs/train_config.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.uav_dataset import UAVSegmentationDataset
from models.thin_dy_unet import ThinDyUNet
from utils.metrics import SegmentationMetrics


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    log_interval: int = 100,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    """
    AMP + Gradient accumulation 학습.
    Effective batch = batch_size × accumulation_steps.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train", leave=False)
    batch_idx = -1
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # AMP forward
        with autocast(device_type="cuda", dtype=amp_dtype):
            preds = model(images)
            loss = criterion(preds, masks) / accumulation_steps

        # Scaled backward
        scaler.scale(loss).backward()

        # accumulation_steps마다 step (scaler.step + update)
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    # 마지막 잔여 gradient flush
    if batch_idx >= 0 and (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metrics = SegmentationMetrics(threshold=0.5)

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with autocast(device_type="cuda", dtype=amp_dtype):
            preds = model(images)
            loss = criterion(preds, masks)

        total_loss += loss.item()
        n_batches += 1
        # metric은 float32로 안전하게 계산
        metrics.update(preds.float(), masks)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Train ThinDyUNet (AMP)")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="AMP dtype (bfloat16은 GradScaler 불필요하나 RTX 4070S는 fp16 권장)",
    )
    parser.add_argument(
        "--save_subdir", type=str, default="amp",
        help="checkpoints/{save_subdir}로 저장 (원본과 분리)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    print(f"AMP dtype: {amp_dtype}")

    # Dataset
    data_cfg = cfg["data"]
    img_size = tuple(data_cfg["img_size"])

    train_dataset = UAVSegmentationDataset(
        images_dir=str(project_root / data_cfg["images_root"] / "train"),
        masks_dir=str(project_root / data_cfg["masks_root"] / "train"),
        img_size=img_size,
        stride=data_cfg.get("train_stride", 1),
    )
    val_dataset = UAVSegmentationDataset(
        images_dir=str(project_root / data_cfg["images_root"] / "val"),
        masks_dir=str(project_root / data_cfg["masks_root"] / "val"),
        img_size=img_size,
        stride=data_cfg.get("val_stride", 1),
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model_cfg = cfg["model"]
    model = ThinDyUNet(
        in_channels=model_cfg["in_channels"],
        n_classes=model_cfg["n_classes"],
        base_ch=model_cfg["base_ch"],
        n_kernels=model_cfg["n_kernels"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # AMP scaler (bfloat16이면 enabled=False로 자동 우회)
    scaler = GradScaler(device="cuda", enabled=(amp_dtype == torch.float16))

    # Checkpoint 디렉토리 (원본과 분리)
    save_dir = project_root / train_cfg["save_dir"] / args.save_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    accum_steps = train_cfg.get("accumulation_steps", 1)
    effective_batch = train_cfg["batch_size"] * accum_steps

    print(f"\n{'='*60}")
    print(f"Starting AMP training for {train_cfg['max_epochs']} epochs")
    print(f"Micro batch: {train_cfg['batch_size']}, Accum steps: {accum_steps}, "
          f"Effective batch: {effective_batch}")
    print(f"LR: {train_cfg['learning_rate']}, Patience: {train_cfg['early_stopping_patience']}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg["max_epochs"]):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=accum_steps,
            log_interval=train_cfg["log_interval"],
            amp_dtype=amp_dtype,
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, amp_dtype=amp_dtype,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1}/{train_cfg['max_epochs']}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Precision: {val_metrics['precision']:.3f} | "
            f"Recall: {val_metrics['recall']:.3f} | "
            f"Dice: {val_metrics['dice']:.3f} | "
            f"mIoU: {val_metrics['miou']:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                save_dir / "best_model.pth",
            )
            print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "val_metrics": val_metrics,
            },
            save_dir / "last_model.pth",
        )

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\nTraining completed!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {save_dir}")


if __name__ == "__main__":
    main()
