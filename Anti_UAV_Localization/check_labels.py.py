import io, sys, tarfile
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    TAR = Path(sys.argv[1])
else:
    found = sorted(HERE.glob("*Labels*.tar*"))
    print("찾은 Labels 파일:", [p.name for p in found] or "없음")
    if not found:
        print("폴더 안 파일 목록:", [p.name for p in HERE.iterdir()])
        sys.exit(1)
    TAR = found[0]
print("사용할 파일:", TAR)

N = 300        # fill 비율을 잴 마스크 수
STRIDE = 200   # 한 시퀀스에 몰리지 않게 200개마다 1개씩 표본 추출
IMG_EXT = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")

names_shown, seen, examples = 0, 0, []
ratios = {"visible": [], "infrared": [], "other": []}

with tarfile.open(TAR, mode="r|*") as tf:
    for m in tf:
        if names_shown < 30:
            print("  ", m.name)
            names_shown += 1
            if names_shown == 30:
                print("   ... (목록 앞 30개)\n")
        if not m.isfile() or not m.name.lower().endswith(IMG_EXT):
            continue
        seen += 1
        if seen % STRIDE:
            continue
        arr = np.array(Image.open(io.BytesIO(tf.extractfile(m).read())).convert("L")) > 0
        if not arr.any():
            continue  # 빈 마스크(드론 부재) 제외
        ys, xs = np.where(arr)
        box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        fill = arr.sum() / box
        low = m.name.lower()
        key = "visible" if "visible" in low else "infrared" if "infrared" in low else "other"
        ratios[key].append(fill)
        if len(examples) < 8:
            examples.append((m.name, fill))
        if sum(len(v) for v in ratios.values()) >= N:
            break

print(f"\n읽은 마스크 파일 {seen}개 중 표본 추출")
for key, vals in ratios.items():
    if not vals:
        continue
    r = np.array(vals)
    med = np.median(r)
    hi = (r >= 0.95).mean()
    lo = (r < 0.7).mean()
    print(f"[{key}] {len(r)}개")
    print(f"   fill 중앙값 {med:.3f}")
    print(f"   0.95 이상 {hi:.1%} / 0.7 미만 {lo:.1%}")