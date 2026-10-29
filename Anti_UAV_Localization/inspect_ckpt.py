"""
체크포인트 안에 저장된 학습 정보(config, 실행 인자, epoch 등)를 출력한다.
V2가 실제로 어떤 데이터·초기 가중치로 학습됐는지 확인하는 용도.

사용법 (Anti_UAV_Localization 폴더에서):
    python inspect_ckpt.py checkpoints/frozen/v2_best_dut_v3.pth
"""
import json
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/frozen/v2_best_dut_v3.pth")
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:  # 오래된 torch
    ckpt = torch.load(path, map_location="cpu")

print("파일:", path)
if not isinstance(ckpt, dict):
    sys.exit(f"dict가 아닌 체크포인트입니다: {type(ckpt)}")

print("최상위 키:", list(ckpt.keys()))
for key, val in ckpt.items():
    if isinstance(val, (int, float, str, bool)) or val is None:
        print(f"  {key}: {val}")

# config / 실행 인자처럼 보이는 항목을 모두 출력
for key, val in ckpt.items():
    low = key.lower()
    if any(s in low for s in ("config", "cfg", "args", "hparam", "finetune", "pretrain", "data")):
        print(f"\n===== {key} =====")
        if hasattr(val, "__dict__") and not isinstance(val, dict):
            val = vars(val)
        try:
            print(json.dumps(val, indent=2, ensure_ascii=False, default=str))
        except Exception:
            print(val)
