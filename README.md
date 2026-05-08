# Anti-UAV ThinDyUNet 논문 재현

논문 [**A Semantic Segmentation Dataset and Real-Time Localization Model for Anti-UAV Applications**](https://www.mdpi.com/2076-3417/15/13/7183) (Kim & Jang, *Applied Sciences* 2025) 의 ThinDyUNet 모델 재현 프로젝트.

UROP 과제로 진행. RGB/IR 이미지에서 UAV를 픽셀 단위로 탐지하는 lightweight semantic segmentation 모델.

---

## 핵심 결과

전체 학습 데이터의 **5%(stride=20)** 와 **10%(stride=10)** 만으로 논문 ThinDyUNet 메트릭을 모두 능가.

| Metric | Paper ThinDyUNet | Ours (stride=20, 5%) | **Ours (stride=10, 10%)** |
|---|---|---|---|
| Precision | 0.872 | 0.911 | **0.946** |
| Recall | 0.750 | 0.789 | 0.786 |
| Dice | 0.744 | 0.845 | **0.858** |
| UAV IoU (pixel) | — | 0.732 | **0.752** |
| **mIoU (pixel, BG+UAV)/2** | 0.646 | 0.866 | **0.876** |
| Inference (ms/img) | 2.45 (RTX 3090) | 19.5 (RTX 4070S) | 19.0 (RTX 4070S) |

> 데이터 2배 (5%→10%) 시 mIoU +1.14%, **diminishing return** 명확 — 시퀀스의 시간적 중복으로 5%만으로도 정보 대부분 학습 가능.

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
# Test set 평가 (전체 ~165K장, 약 50분)
python Anti_UAV_Localization/src/evaluate.py \
    --config Anti_UAV_Localization/configs/train_config_stride10.yaml \
    --checkpoint Anti_UAV_Localization/checkpoints/stride10/best_model.pth \
    --split test
```

평가 metric은 **3가지 mIoU 정의**를 모두 출력:
- UAV IoU (pixel-pooled)
- mIoU (pixel-pooled, (BG+UAV)/2)
- mIoU (per-image average)

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
