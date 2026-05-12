"""
ThinDyUNet 풀 데이터 학습 스크립트 (논문 베이스 + 검증된 개선 통합).

논문 적용:
  - AdamW optimizer
  - DiceLoss (또는 BCE+Dice combo)
  - ReduceLROnPlateau (factor=0.15, patience=10, cooldown=5)
  - stride=1 (전체 train 프레임)

우리 개선 유지:
  - ImageNet Normalize (dataset)
  - shuffle=True
  - N-fold efficient dynamic conv (model)
  - N=3 kernel 후보
  - Gradient accumulation으로 effective batch 동일
  - 3가지 mIoU 정의 모두 출력

추가:
  - CSV epoch 로그
  - 옵션 AMP (config.amp_enabled)
  - 주기적 체크포인트 (config.ckpt_every_epochs)
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.uav_dataset import UAVSegmentationDataset
from models.thin_dy_unet import ThinDyUNet
from utils.metrics import SegmentationMetrics


# --------------------------------------------------------------------
# Loss factory
# --------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Binary Dice loss on sigmoid(logits)."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w_bce = bce_weight
        self.w_dice = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_bce * self.bce(logits, targets) + self.w_dice * self.dice(logits, targets)


def build_loss(loss_cfg: dict) -> nn.Module:
    name = loss_cfg.get("name", "bce_dice").lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
        )
    raise ValueError(f"Unknown loss: {name}")


# --------------------------------------------------------------------
# Train / Val
# --------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    log_interval: int,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                preds = model(images)
                loss = criterion(preds, masks) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            preds = model(images)
            loss = criterion(preds, masks) / accumulation_steps
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    # leftover flush
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    metrics = SegmentationMetrics(threshold=0.5)

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images)
        loss = criterion(preds, masks)
        total_loss += loss.item()
        n_batches += 1
        metrics.update(preds, masks)

    return total_loss / max(n_batches, 1), metrics.compute()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ThinDyUNet (full-data)")
    parser.add_argument("--config", type=str, default="configs/train_config_full.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(__file__).resolve().parent.parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    print(f"Val samples:   {len(val_dataset):,}")

    # Loader
    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=data_cfg["num_workers"] > 0,
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

    # Loss
    criterion = build_loss(train_cfg.get("loss", {"name": "bce_dice"}))
    print(f"Loss: {type(criterion).__name__}")

    # Optimizer (AdamW per paper)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=weight_decay,
    )
    print(f"Optimizer: AdamW (lr={train_cfg['learning_rate']}, wd={weight_decay})")

    # Scheduler (ReduceLROnPlateau per paper)
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.15),
        patience=sched_cfg.get("patience", 10),
        cooldown=sched_cfg.get("cooldown", 5),
        min_lr=sched_cfg.get("min_lr", 1e-7),
    )

    # AMP
    use_amp = bool(train_cfg.get("amp_enabled", False))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP fp16: {use_amp}")

    # Checkpoint dir
    save_dir = project_root / train_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Accumulation
    accum_steps = train_cfg.get("accumulation_steps", 1)
    effective_batch = train_cfg["batch_size"] * accum_steps

    # CSV log
    csv_path = save_dir / "train_log.csv"
    if not args.resume or not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "precision", "recall", "dice",
                "uav_iou_pixel", "miou_pixel",
                "uav_iou_per_image", "miou_per_image",
                "lr", "epoch_time_sec",
            ])

    # ----- Loop -----
    ckpt_every = train_cfg.get("ckpt_every_epochs", 0)
    print(f"\n{'='*60}")
    print(f"Training {train_cfg['max_epochs']} epochs")
    print(f"Batch: {train_cfg['batch_size']} × accum {accum_steps} = effective {effective_batch}")
    print(f"Scheduler: ReduceLROnPlateau factor={sched_cfg.get('factor', 0.15)} "
          f"patience={sched_cfg.get('patience', 10)} cooldown={sched_cfg.get('cooldown', 5)}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg["max_epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            accumulation_steps=accum_steps,
            log_interval=train_cfg["log_interval"],
            scaler=scaler,
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        # Scheduler step (on val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{train_cfg['max_epochs']}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
            f"Dice {val_metrics['dice']:.3f} "
            f"mIoU(px) {val_metrics['miou_pixel']:.3f} "
            f"UAV-IoU(img) {val_metrics['uav_iou_per_image']:.3f} | "
            f"LR {current_lr:.2e} | {epoch_time:.0f}s"
        )

        # CSV row
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}",
                f"{val_metrics['dice']:.6f}",
                f"{val_metrics['uav_iou']:.6f}",
                f"{val_metrics['miou_pixel']:.6f}",
                f"{val_metrics['uav_iou_per_image']:.6f}",
                f"{val_metrics['miou_per_image']:.6f}",
                f"{current_lr:.3e}",
                f"{epoch_time:.1f}",
            ])

        # Best
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                save_dir / "best_model.pth",
            )
            print(f"  -> Best (val_loss {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")

        # Always save last
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "val_metrics": val_metrics,
            },
            save_dir / "last_model.pth",
        )

        # Periodic (epoch_NN.pth)
        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                },
                save_dir / f"epoch_{epoch+1:03d}.pth",
            )

        # Early stop
        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.4f} | Saved in: {save_dir}")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
