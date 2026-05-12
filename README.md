# Anti-UAV ThinDyUNet 논문 재현

논문 [**A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications**](https://www.mdpi.com/2076-3417/15/13/7183) (Kim & Jang, *Applied Sciences* 2025) 의 ThinDyUNet 모델 재현 프로젝트.

UROP 과제로 진행. RGB/IR 이미지에서 UAV를 픽셀 단위로 탐지하는 lightweight semantic segmentation 모델.

---

## 핵심 결과

전체 학습 데이터의 **5%(stride=20)** 와 **10%(stride=10)** 만으로 논문 ThinDyUNet 재구현.

### 1) 표준 메트릭 정의 (threshold = 0.5)

| Metric | Paper ThinDyUNet | Ours (stride=20, 5%) | **Ours (stride=10, 10%)** |
|---|---|---|---|
| Precision | 0.872 | 0.911 | **0.946** |
| Recall | 0.750 | 0.789 | 0.786 |
| Dice | 0.744 | 0.845 | **0.858** |
| UAV IoU (pixel) | — | 0.732 | **0.752** |
| mIoU (pixel, BG+UAV)/2 | 0.646 † | 0.866 | **0.876** |
| Inference (ms/img) | 2.45 (RTX 3090) | 19.5 (RTX 4070S) | 19.0 (RTX 4070S) / 15.6 (RTX A5000) |

† 논문 수치는 다른 메트릭 정의로 측정됨 — 직접 비교 불가. **아래 2)** 참고.

> 데이터 2배 (5%→10%) 시 mIoU +1.14%, **diminishing return** 명확 — 시퀀스의 시간적 중복으로 5%만으로도 정보 대부분 학습 가능.

### 2) 논문 메트릭 정의로 다시 측정 (threshold = 0.9, per-image UAV IoU)

공식 repo [`SCKIMOSU/uav`](https://github.com/SCKIMOSU/uav) 분석 결과, 논문이 "mIoU"라 부르는 값은:
- **sigmoid > 0.9** 로 보수적 이진화 (우리 표준 0.5와 다름)
- **per-image UAV-only IoU** 의 평균 (BG 클래스 미포함, pixel pool 아님)

같은 정의(`threshold=0.9` + `UAV IoU per-image avg`)로 재측정하면:

| Metric | Paper | **Ours (stride=10, t=0.9)** | Δ |
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

### 시각화 예시 (test set)

`results/vis_grid_stride10.png` — 12개 샘플 × [Input, GT, Prediction, Overlay]

`results/vis_compare_s20_vs_s10.png` — 두 모델 직접 비교 (Diff 채널 포함)

---

## 환경

- Windows 11, RTX 4070 Super (12 GB VRAM)
- Python 3.12, PyTorch (CUDA 12.4)

```bash
pip install -r Anti_UAV_Localization/requirements.txt
```

---

## 데이터셋

논문 저자 공개 데이터셋 (직접 받아서 압축 해제 필요, 약 100GB+):

- `UAVSemanticSegmentationInput.tar.gz` — 이미지 (RGB + IR), 605,045장
- `UAVSemanticSegmentationLabels.tar.gz` — 바이너리 마스크

압축 해제:
```bash
python Anti_UAV_Localization/scripts/extract_data.py
```

`Anti_UAV_Localization/data/raw/{images,masks}/{train,val,test}/...` 구조로 정리됨.

> 데이터 + 체크포인트 + 로그는 `.gitignore` 처리 (용량 문제). 학습된 모델 가중치는 GitHub Release 또는 별도 storage로 받을 것.

---

## 모델: ThinDyUNet (1.37M params)

논문 Section 4 / Figure 6 기반 구현.

- **U-Net 기반** encoder-decoder
- **고정 64채널** (모든 layer)
- **Dynamic Convolution** (encoder)
  - N개 커널 후보 + SE-style attention으로 입력 의존적 가중합
  - `K_dyn = Σ αᵢ·Kᵢ`, `Y = LeakyReLU(GroupNorm(K_dyn * X))`
- **N-fold 효율 구현**
  - `conv(x, Σαᵢ·Kᵢ) = Σαᵢ·conv(x, Kᵢ)` 항등식 활용
  - 단일 conv (output channels = N×out_ch) 후 attention 가중합
  - cuDNN이 잘 최적화하는 일반 conv 한 번으로 처리

```python
out = F.conv2d(x, weight, padding=padding)         # (B, N*C_out, H, W)
out = out.view(B, N, C_out, H, W)
out = (alpha.view(B, N, 1, 1, 1) * out).sum(dim=1) # (B, C_out, H, W)
```

비교용 ablation 모델: `src/models/thin_unet.py` (regular conv, 14.78M params).

---

## 학습

논문 Section 5.2 설정 + RTX 4070S 메모리 제약 반영:

| 항목 | 값 |
|---|---|
| Input size | 512×512 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Micro batch | 4 |
| Gradient accumulation | 6 → effective batch **24** (논문 동일) |
| Loss | BCEWithLogits |
| Epochs | 50 (early stopping patience=10) |

```bash
# 5% 데이터 (stride=20)
python Anti_UAV_Localization/src/train.py \
    --config Anti_UAV_Localization/configs/train_config.yaml

# 10% 데이터 (stride=10) — checkpoints/stride10/에 별도 저장
python Anti_UAV_Localization/src/train.py \
    --config Anti_UAV_Localization/configs/train_config_stride10.yaml
```

학습 시간 (RTX 4070S 기준):
- stride=20: ~951 s/epoch, 31 epoch 만에 early stop (총 ~8시간)
- stride=10: ~1900 s/epoch, 15 epoch에 best (수동 정지 20 epoch, 총 ~10시간)

---

## 평가

```bash
# 표준 평가 (threshold=0.5, 우리 기본값)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_stride10.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/stride10/best_model.pth \
    --split test

# 논문 정의로 평가 (threshold=0.9, per-image UAV IoU 비교)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_stride10.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/stride10/best_model.pth \
    --split test --threshold 0.9
```

평가 metric은 **3가지 mIoU 정의**를 모두 출력해서 정의 혼선을 방지:
- UAV IoU (pixel-pooled) — UAV 클래스만 픽셀 전체 풀링
- mIoU (pixel-pooled, (BG+UAV)/2) — 두 클래스 평균
- **UAV IoU (per-image avg)** — 논문의 "mIoU"에 해당하는 정의
- mIoU (per-image avg) — per-image (BG+UAV)/2 평균

---

## 시각화

```bash
# 단일 모델 그리드
python Anti_UAV_Localization/scripts/visualize_only.py \
    --checkpoint Anti_UAV_Localization/checkpoints/stride10/best_model.pth \
    --n_samples 12

# 두 모델 직접 비교 (Diff 채널 포함)
python Anti_UAV_Localization/scripts/visualize_compare.py \
    --ckpt_a Anti_UAV_Localization/checkpoints/best_model.pth \
    --ckpt_b Anti_UAV_Localization/checkpoints/stride10/best_model.pth \
    --n_samples 10
```

---

## 실험 노트 (RTX 4070S에서 검증된 한계)

| 시도 | 결과 |
|---|---|
| AMP fp16 | **효과 없음** — GroupNorm + N-fold reshape이 텐서 코어 미활용. epoch time 거의 동일 |
| Batch size 증가 (4→6→8) | **오히려 10~30배 느려짐** — N-fold reshape의 메모리 패턴이 큰 batch에서 비효율. batch=12는 OOM |
| Stride=10 (10% data) | **mIoU +1.14%** — 데이터 2배 효과는 수확 체감 명확 |

→ 이 모델 구조에서는 **batch=4 + FP32**가 RTX 4070S 실질적 한계. 추가 가속은 ONNX/TorchScript 등 inference 최적화 영역.

## 공식 repo 분석 노트

[`SCKIMOSU/uav`](https://github.com/SCKIMOSU/uav) 와 비교해서 발견한 차이점:

| 영역 | 공식 | 우리 | 효과 |
|---|---|---|---|
| **Dynamic Conv 구현** | per-sample kernel + `groups=batch` grouped conv | `conv(x, ΣαK)=Σα·conv(x,K)` 항등식 활용한 단일 conv | **3~10배 빠름** (cuDNN 친화) |
| **메트릭** | sigmoid > 0.9, per-image UAV-only IoU | sigmoid > 0.5 + 3가지 mIoU 정의 모두 출력 | 정의 혼선 방지 |
| **Train shuffle** | `shuffle=True` 누락 | `shuffle=True` | 같은 시퀀스 연속 프레임 분산 |
| **Normalize** | 없음 | ImageNet stats | 학습 안정성 |
| **추론 시간** | `*100` (off-by-10 의심) | `cuda.synchronize()` + per-image ms | 정확 측정 |
| **N (kernel 후보)** | 2 (하드코딩) | 3 (config) | 표현력↑ |
| **Loss** | DiceLoss | BCEWithLogits | stride 샘플링 덕에 BCE로도 충분 |

→ 공식 코드의 `shuffle=False`, `*100` ms 등은 사실상 버그로 보임. 표 2의 4%p 격차의 상당 부분이 이런 학습 트릭 차이로 추정됨.

---

## 프로젝트 구조

```
Anti_UAV_Localization/
├── configs/
│   ├── train_config.yaml             # stride=20 (5%)
│   └── train_config_stride10.yaml    # stride=10 (10%)
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
│   ├── train.py                      # FP32 메인 학습
│   ├── train_amp.py                  # AMP 실험용 (효과 X)
│   └── evaluate.py
├── results/                          # 시각화 PNG
├── data/                             # gitignore (raw/)
└── checkpoints/                      # gitignore
```

---

## References

- Kim, S.; Jang, K. *A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications.* Applied Sciences 2025, 15, 7183.
- Wang, L. et al. *Temporal Segment Networks for Action Recognition in Videos.* ECCV 2016 — sequence-aware sparse sampling 관련.
- Chen, Y. et al. *Dynamic Convolution: Attention over Convolution Kernels.* CVPR 2020 — dynamic convolution 원조.
