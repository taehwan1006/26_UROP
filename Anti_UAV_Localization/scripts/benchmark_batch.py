"""
batch_size별 메모리/속도 벤치마크.
실제 학습 전 OOM 위험과 throughput를 측정해 sweet spot 결정.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.thin_dy_unet import ThinDyUNet


def bench(batch_size: int, n_iter: int = 20, img_size: int = 512) -> dict:
    """주어진 batch_size로 forward+backward를 N번 실행하고 측정."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = ThinDyUNet(in_channels=3, n_classes=1, base_ch=64, n_kernels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    # Warmup
    try:
        for _ in range(3):
            x = torch.randn(batch_size, 3, img_size, img_size, device=device)
            y = torch.randint(0, 2, (batch_size, 1, img_size, img_size), device=device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        # Measure
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(n_iter):
            x = torch.randn(batch_size, 3, img_size, img_size, device=device)
            y = torch.randint(0, 2, (batch_size, 1, img_size, img_size), device=device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        it_per_s = n_iter / elapsed
        img_per_s = batch_size * it_per_s

        return {
            "batch_size": batch_size,
            "ok": True,
            "iter_per_s": it_per_s,
            "img_per_s": img_per_s,
            "peak_mem_mb": peak_mb,
            "ms_per_iter": elapsed / n_iter * 1000,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "ok": False, "error": "OOM"}
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    print(f"{'batch':>6} | {'OK':>4} | {'ms/iter':>9} | {'img/s':>7} | {'peak GB':>8} | effective batch (accum)")
    print("-" * 80)

    # 효과적 배치를 24로 유지하기 위해 24의 약수 위주로 측정
    candidates = [4, 6, 8, 12]
    for bs in candidates:
        r = bench(bs, n_iter=20)
        if r["ok"]:
            accum = max(1, 24 // bs)
            eff = bs * accum
            print(f"{r['batch_size']:>6} | {'OK':>4} | {r['ms_per_iter']:>9.1f} | "
                  f"{r['img_per_s']:>7.1f} | {r['peak_mem_mb']/1024:>8.2f} | "
                  f"{eff} ({bs}x{accum})")
        else:
            print(f"{r['batch_size']:>6} | {'OOM':>4} | {'-':>9} | {'-':>7} | {'-':>8} |")


if __name__ == "__main__":
    main()
