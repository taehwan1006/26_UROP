"""
V2 모델(ThinDyUNet)에 test_dut 프레임을 넣고, 단계별 특징맵과 Dynamic Conv 가중치를 저장한다.
발표 슬라이드의 '모델 구조 단계별 예시' 그림을 만들기 위한 용도.

사용법 (Anti_UAV_Localization 폴더에 저장 후):
    python export_stage_maps.py
    python export_stage_maps.py --frames video20/00800 video10/01300

결과: viz_stage_maps/<video>_<frame>.npz (프레임마다 1개, 약 3MB)
  - input: 512x512 RGB (uint8)       - gt: 512x512 GT 마스크 (uint8)
  - prob:  512x512 출력 확률 (float16)
  - <stage>_mean: 채널 평균 |활성값| 맵      (float16)
  - <stage>_sel:  드론 영역 반응이 가장 큰 채널 맵 (float16)
  - <stage>_sel_idx, <stage>_ratio: 그 채널 번호와 (드론 안 평균 / 밖 평균)
  - alpha_<layer>: Dynamic Conv 커널 가중치 α (N=3)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from models.thin_dy_unet import DynamicConv2d, ThinDyUNet  # noqa: E402
from utils.checkpoint import load_weights  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
STAGES = ["enc1", "enc2", "enc3", "enc4", "bottleneck", "dec4", "dec3", "dec2", "dec1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/frozen/v2_best_dut_v3.pth")
    ap.add_argument("--images", default="data/raw/images/test_dut")
    ap.add_argument("--masks", default="data/raw/masks/test_dut")
    ap.add_argument("--frames", nargs="+", default=["video20/00800", "video10/01300", "video01/00300"])
    ap.add_argument("--out", default="viz_stage_maps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThinDyUNet(in_channels=3, n_classes=1, base_ch=64, n_kernels=3).to(device)
    load_weights(model, HERE / args.checkpoint, device)
    model.eval()

    # 단계별 출력 저장용 hook
    feats = {}
    for name in STAGES:
        getattr(model, name).register_forward_hook(
            lambda m, i, o, name=name: feats.__setitem__(name, o.detach().float().cpu()[0]))

    # Dynamic Conv attention 가중치 α 저장용 hook
    alphas = {}
    for name, mod in model.named_modules():
        if isinstance(mod, DynamicConv2d):
            mod.attention.register_forward_hook(
                lambda m, i, o, name=name: alphas.__setitem__(
                    name, torch.softmax(o.detach().float(), dim=1).cpu()[0].numpy()))

    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    out_dir = HERE / args.out
    out_dir.mkdir(exist_ok=True)

    for fr in args.frames:
        img_path = HERE / args.images / f"{fr}.jpg"
        mask_path = HERE / args.masks / f"{fr}_mask.png"
        if not img_path.exists():
            print(f"[건너뜀] 이미지 없음: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        feats.clear()
        alphas.clear()
        with torch.no_grad():
            logit = model(x)
        prob = torch.sigmoid(logit)[0, 0].float().cpu().numpy()

        gt = np.zeros((512, 512), np.uint8)
        if mask_path.exists():
            m = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)
            gt = (np.array(m) > 127).astype(np.uint8)

        data = {
            "input": np.array(img.resize((512, 512), Image.BILINEAR), dtype=np.uint8),
            "gt": gt,
            "prob": prob.astype(np.float16),
            "orig_size": np.array(img.size),
        }
        for name in STAGES:
            f = feats[name].abs()                      # (C, h, w)
            h, w = f.shape[1:]
            data[f"{name}_mean"] = f.mean(0).numpy().astype(np.float16)
            g = torch.from_numpy(gt)[None, None].float()
            g = torch.nn.functional.interpolate(g, size=(h, w), mode="area")[0, 0] > 0.25
            if g.any() and (~g).any():
                inside = f[:, g].mean(1)
                outside = f[:, ~g].mean(1) + 1e-6
                ratio = inside / outside
                k = int(ratio.argmax())
                data[f"{name}_sel"] = f[k].numpy().astype(np.float16)
                data[f"{name}_sel_idx"] = np.array(k)
                data[f"{name}_ratio"] = np.array(float(ratio[k]))
        for name, a in alphas.items():
            data["alpha_" + name.replace(".", "_")] = a.astype(np.float32)

        save = out_dir / f"{fr.replace('/', '_')}.npz"
        np.savez_compressed(save, **data)
        tp = int(((prob > 0.5) & (gt > 0)).sum())
        print(f"저장: {save}  (GT {int(gt.sum())}px, 예측 {int((prob > 0.5).sum())}px, TP {tp}px)")


if __name__ == "__main__":
    main()
