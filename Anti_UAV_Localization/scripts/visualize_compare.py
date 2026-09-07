"""
stride=20 vs stride=10 모델의 예측을 같은 샘플에 대해 나란히 비교 시각화.
Columns: Input | GT | stride=20 pred | stride=10 pred | Diff (s20 vs s10)
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.builder import build_split_dataset
from models import build_model
from utils.checkpoint import load_weights
from utils.config import load_config
from utils.image import denorm


def load_model(ckpt_path: Path, device, model_cfg) -> nn.Module:
    model = build_model(model_cfg).to(device)
    load_weights(model, ckpt_path, device)
    model.eval()
    return model


def predict(model, image, device) -> np.ndarray:
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        logit = model(x)
        return (torch.sigmoid(logit).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)


def diff_map(pred_a: np.ndarray, pred_b: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    RGB diff: 빨강=A만 예측, 초록=B만 예측, 파랑=둘 다 예측 (GT 영역 표시).
    """
    h, w = pred_a.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[..., 0] = pred_a * (1 - pred_b)        # A only
    out[..., 1] = (1 - pred_a) * pred_b        # B only
    out[..., 2] = pred_a * pred_b              # both
    # GT 윤곽 흰색
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_stride10.yaml")
    parser.add_argument("--ckpt_a", default="checkpoints/best_model.pth", help="stride=20 모델")
    parser.add_argument("--ckpt_b", default="checkpoints/stride10/best_model.pth", help="stride=10 모델")
    parser.add_argument("--label_a", default="stride=20")
    parser.add_argument("--label_b", default="stride=10")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--output", default="results/vis_compare_s20_vs_s10.png")
    parser.add_argument("--eval_stride", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-sequences", type=str, nargs="+", default=None,
                        help="비교할 시퀀스 glob 패턴 (예: video01 video0*)")
    parser.add_argument("--exclude-sequences", type=str, nargs="+", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(str(project_root / args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = build_split_dataset(
        cfg["data"], args.split, project_root,
        overrides={
            "stride": args.eval_stride,
            "include_sequences": args.include_sequences,
            "exclude_sequences": args.exclude_sequences,
        },
    )

    model_a = load_model(project_root / args.ckpt_a, device, cfg["model"])
    model_b = load_model(project_root / args.ckpt_b, device, cfg["model"])

    # 두 모델 예측이 다른 샘플 위주로 선택 (의미 있는 비교)
    rng = random.Random(args.seed)
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    chosen = []
    diverse_chosen = []  # 두 모델 예측이 다른 샘플
    for idx in candidates:
        image, mask = dataset[idx]
        if mask.sum() < 50:
            continue
        pred_a = predict(model_a, image, device)
        pred_b = predict(model_b, image, device)
        diff_pixels = np.logical_xor(pred_a, pred_b).sum()

        if diff_pixels > 30 and len(diverse_chosen) < args.n_samples // 2:
            diverse_chosen.append((idx, image, mask, pred_a, pred_b))
        elif diff_pixels <= 30 and len(chosen) < args.n_samples - args.n_samples // 2:
            chosen.append((idx, image, mask, pred_a, pred_b))

        if len(chosen) + len(diverse_chosen) >= args.n_samples:
            break

    samples = diverse_chosen + chosen
    print(f"Selected {len(samples)} samples ({len(diverse_chosen)} different, {len(chosen)} similar)")

    # 그리드 시각화: 5컬럼
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = [
        "Input",
        "Ground Truth",
        f"Pred ({args.label_a})",
        f"Pred ({args.label_b})",
        f"Diff: R={args.label_a}, G={args.label_b}, B=both",
    ]

    for row, (idx, image, mask, pred_a, pred_b) in enumerate(samples):
        img_np = denorm(image)
        gt_np = mask.squeeze().cpu().numpy()
        diff_rgb = diff_map(pred_a, pred_b, gt_np)

        axes[row, 0].imshow(img_np)
        axes[row, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred_a, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(pred_b, cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(diff_rgb)

        for c in range(5):
            axes[row, c].axis("off")
            if row == 0:
                axes[row, c].set_title(col_titles[c], fontsize=12)
        axes[row, 0].text(
            -50, img_np.shape[0] // 2, f"#{idx}",
            rotation=90, va="center", ha="right", fontsize=11,
        )

    plt.tight_layout()
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
