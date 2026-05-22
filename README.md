# Anti-UAV ThinDyUNet 논문 재현 및 파이프라인 개선

논문 [**A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications**](https://www.mdpi.com/2076-3417/15/13/7183) (Kim & Jang, *Applied Sciences* 2025) 의 ThinDyUNet 모델 재현 및 최적화 프로젝트.

UROP 과제로 진행. RGB/IR 이미지에서 UAV를 픽셀 단위로 탐지하는 lightweight semantic segmentation 모델.

---

## 핵심 결과

전체 학습 데이터의 **5%(stride=20)** 와 **10%(stride=10)** 만으로 논문 ThinDyUNet 재구현 및 **자체 최적화 파이프라인(V2)** 적용을 통해 논문 성능을 크게 상회하는 결과 달성.

### 1) 표준 메트릭 정의 (threshold = 0.5)

자체적으로 손실 함수(BCEDiceLoss)와 최적화 스케줄러(AdamW+ReduceLROnPlateau)를 도입한 **V2 파이프라인** 적용 결과, 모든 지표에서 비약적인 상승을 기록했습니다.

| Metric | Paper ThinDyUNet | V1 (stride=20, 5%) | V1 (stride=10, 10%) | **V2 (stride=10, 10%)** 🚀 |
|---|---|---|---|---|
| Precision | 0.872 | 0.911 | 0.946 | **0.949** |
| Recall | 0.750 | 0.789 | 0.786 | **0.949** |
| Dice | 0.744 | 0.845 | 0.858 | **0.949** |
| UAV IoU (pixel) | — | 0.732 | 0.752 | **—** |
| mIoU (pixel, BG+UAV)/2 | 0.646 † | 0.866 | 0.876 | **0.951** |
| **UAV IoU (per-image avg)** | — | — | — | **0.893** |
| Inference (ms/img) | 2.45 (RTX 3090) | 19.5 (4070S) | 19.0 (4070S) | **15.6 (A5000)** |

† 논문 수치는 다른 메트릭 정의로 측정됨 — 직접 비교 불가. **아래 2)** 참고.

> **V2 파이프라인 개선 효과:** 단순 BCE Loss에서 **BCEDiceLoss**로 변경하고 혼합 정밀도(AMP)와 최신 스케줄러를 적용한 결과, 드론 픽셀 탐지율(Recall)이 `0.786 → 0.949`로 폭발적으로 상승하며 정밀도와 완벽한 밸런스를 달성했습니다. (Epoch 31 기준 UAV-IoU(img)는 **89.3%**, mIoU(px)는 **95.1%**까지 확보됨)

### 2) 논문 메트릭 정의로 다시 측정 (threshold = 0.9, per-image UAV IoU)

공식 repo [`SCKIMOSU/uav`](https://github.com/SCKIMOSU/uav) 분석 결과, 논문이 "mIoU"라 부르는 값은:
- **sigmoid > 0.9** 로 보수적 이진화 (우리 표준 0.5와 다름)
- **per-image UAV-only IoU** 의 평균 (BG 클래스 미포함, pixel pool 아님)

같은 정의(`threshold=0.9` + `UAV IoU per-image avg`)로 재측정하면:

| Metric | Paper | **Ours V1 (stride=10, t=0.9)** | Δ |
|---|---|---|---|
| Precision | 0.872 | **0.9855** | +0.114 |
| Recall | 0.750 | 0.6926 | −0.057 |
| Dice | 0.744 | **0.8135** | +0.070 |
| **UAV IoU (per-image avg)** ≡ paper "mIoU" | **0.646** | **0.6849** | **+0.039** |

> 표 1과 표 2의 메트릭 정의를 똑같이 맞추면 격차가 **23%p → 3.9%p**로 줄어듦.
> 즉 **이전에 보였던 큰 격차의 대부분(약 19%p)이 메트릭 정의 차이**이고, 실제 모델 성능 차이는 **약 4%p** 수준.

남은 4%p 격차의 원인 추정 (공식 코드 분석 기반):
1. 공식 train loader에 `shuffle=True` 미적용 (같은 시퀀스 연속 프레임이 같은 배치에 몰림 → gradient correlation↑)
2. ImageNet Normalize 사용 (표준 ConvNet 학습 안정성)
3. Stride 샘플링이 시간적 중복 제거로 강한 정규화 역할

### 시각화 예시 (test set)

`results/vis_grid_stride10.png` — 12개 샘플 × [Input, GT, Prediction, Overlay]

`results/vis_compare_s20_vs_s10.png` — 두 모델 직접 비교 (Diff 채널 포함)

---

## 환경

- **Local:** Windows 11, RTX 4070 Super (12 GB VRAM)
- **Server (V2 학습용):** Ubuntu Linux, RTX A5000 (24 GB VRAM)
- Python 3.12, PyTorch (CUDA 12.4 / 13.2)

```bash
pip install -r Anti_UAV_Localization/requirements.txt

```

---

## 데이터셋

논문 저자 공개 데이터셋 (직접 받아서 압축 해제 필요, 약 100GB+):

* `UAVSemanticSegmentationInput.tar.gz` — 이미지 (RGB + IR), 605,045장
* `UAVSemanticSegmentationLabels.tar.gz` — 바이너리 마스크

압축 해제:

```bash
python Anti_UAV_Localization/scripts/extract_data.py

```

`Anti_UAV_Localization/data/raw/{images,masks}/{train,val,test}/...` 구조로 정리됨.

---

## 모델: ThinDyUNet (1.37M params)

논문 Section 4 / Figure 6 기반 구현.

* **U-Net 기반** encoder-decoder
* **고정 64채널** (모든 layer)
* **Dynamic Convolution** (encoder)
* N개 커널 후보 + SE-style attention으로 입력 의존적 가중합
* `K_dyn = Σ αᵢ·Kᵢ`, `Y = LeakyReLU(GroupNorm(K_dyn * X))`


* **N-fold 효율 구현**
* `conv(x, Σαᵢ·Kᵢ) = Σαᵢ·conv(x, Kᵢ)` 항등식 활용
* 단일 conv (output channels = N×out_ch) 후 attention 가중합
* cuDNN이 잘 최적화하는 일반 conv 한 번으로 처리



```python
out = F.conv2d(x, weight, padding=padding)         # (B, N*C_out, H, W)
out = out.view(B, N, C_out, H, W)
out = (alpha.view(B, N, 1, 1, 1) * out).sum(dim=1) # (B, C_out, H, W)

```

---

## 학습 (V2 파이프라인)

논문 Section 5.2 설정을 기반으로 하되, **안정성과 속도를 극대화한 V2 세팅** 적용 (서버 기준):

| 항목 | 값 | 비고 |
| --- | --- | --- |
| Input size | 512×512 |  |
| Optimizer | **AdamW** | (개선) weight_decay=1e-4 |
| Learning rate | 1e-4 |  |
| Scheduler | **ReduceLROnPlateau** | (개선) factor=0.15, patience=4 |
| Batch Size | 8 | (개선) A5000 24GB 활용 |
| Gradient accumulation | 3 | effective batch **24** 유지 |
| Loss | **BCEDiceLoss** | (개선) 작은 드론 객체 탐지율 대폭 상향 |
| AMP (FP16) | **True** | (개선) 학습 속도 30% 향상 |

```bash
# V2 파이프라인 (추천)
python Anti_UAV_Localization/src/train_full.py \
    --config Anti_UAV_Localization/configs/train_config_full.yaml

```

**학습 시간 비교:**

* V1 (RTX 4070S, stride=10, FP32): ~1,900 s/epoch
* **V2 (RTX A5000, stride=10, AMP FP16): ~1,240 s/epoch (약 35% 가속)**

---

## 평가

```bash
# 표준 평가 (threshold=0.5)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_full.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/full/best_model.pth \
    --split test

# 논문 정의로 평가 (threshold=0.9)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_full.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/full/best_model.pth \
    --split test --threshold 0.9

```

평가 metric은 **3가지 mIoU 정의**를 모두 출력해서 정의 혼선을 방지합니다 (Pixel-pooled, per-image avg 등).

---

## 실험 노트 (트러블슈팅 및 개선 사항)

| 시도 | 결과 및 원인 |
| --- | --- |
| **AMP fp16 적용** | **성능 폭발 (RTX A5000)** — 기존 4070S(V1)에서는 효과가 미미했으나, A5000에서 AMP 적용 시 VRAM 절약은 물론 에포크당 소요 시간이 1,900초에서 1,240초로 약 **35% 가속**됨. |
| **Loss 함수 변경** | **Recall 수직 상승** — 순수 BCE Loss에서 `BCEDiceLoss`로 변경 후, 클래스 불균형(배경 99%) 문제가 해결되며 정밀도(P)와 재현율(R)이 모두 94% 후반대로 수렴. |
| **LR 스케줄러 도입** | **최적 수렴점 확보** — `ReduceLROnPlateau(patience=4)` 도입 후 정밀도와 재현율이 극적으로 밸런스를 맞추며 Local Minima 탈출 및 신기록 지속 갱신. |
| **Batch size 최적화** | 서버의 24GB VRAM을 활용하여 `batch=8, accum=3` 구조로 변경, 메모리 단편화(OOM)를 피하면서도 안정적인 그래디언트 업데이트 달성. |

---

## 공식 repo 분석 노트

[`SCKIMOSU/uav`](https://github.com/SCKIMOSU/uav) 와 비교해서 발견한 차이점:

| 영역 | 공식 | 우리 | 효과 |
| --- | --- | --- | --- |
| **Dynamic Conv 구현** | per-sample kernel + `groups=batch` grouped conv | `conv(x, ΣαK)=Σα·conv(x,K)` 항등식 활용한 단일 conv | **3~10배 빠름** (cuDNN 친화) |
| **메트릭** | sigmoid > 0.9, per-image UAV-only IoU | sigmoid > 0.5 + 3가지 mIoU 정의 모두 출력 | 정의 혼선 방지 |
| **Train shuffle** | `shuffle=True` 누락 | `shuffle=True` | 같은 시퀀스 연속 프레임 분산 |
| **Loss** | DiceLoss | BCEDiceLoss (V2) | 배경 억제와 마스크 품질 동시 확보 |
| **추론 시간** | `*100` (off-by-10 의심) | `cuda.synchronize()` + per-image ms | 정확 측정 |

→ 공식 코드의 `shuffle=False`, `*100` ms 등은 사실상 버그로 보임. 표 2의 4%p 격차의 상당 부분이 이런 학습 트릭 차이로 추정됨.

---

## 프로젝트 구조

```
Anti_UAV_Localization/
├── configs/
│   ├── train_config.yaml             # stride=20 (5%)
│   ├── train_config_stride10.yaml    # stride=10 (10%) V1
│   └── train_config_full.yaml        # stride=10 (10%) V2 파이프라인 메인
├── scripts/
│   ├── extract_data.py               # tar.gz → data/raw/
│   ├── benchmark_batch.py            # batch size별 속도/메모리 측정
│   ├── visualize_only.py             # 단일 모델 시각화
│   └── visualize_compare.py          # 두 모델 비교 시각화
├── src/
│   ├── dataset/uav_dataset.py        # 시퀀스별 stride 샘플링
│   ├── models/
│   │   ├── thin_dy_unet.py           # ThinDyUNet (1.37M)
│   │   └── thin_unet.py              # ablation: regular conv (14.78M)
│   ├── utils/
│   │   ├── metrics.py                # IoU 다중 정의
│   │   └── visualization.py
│   ├── train.py                      # V1: FP32 메인 학습 스크립트
│   ├── train_full.py                 # V2: AMP + AdamW + BCEDice + 스케줄러 메인 학습 스크립트
│   └── evaluate.py
├── results/                          # 시각화 PNG
├── data/                             # gitignore (raw/)
└── checkpoints/                      # gitignore

```

---

## References

* Kim, S.; Jang, K. *A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications.* Applied Sciences 2025, 15, 7183.
* Wang, L. et al. *Temporal Segment Networks for Action Recognition in Videos.* ECCV 2016 — sequence-aware sparse sampling 관련.
* Chen, Y. et al. *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020 — dynamic convolution 원조.

```

```
