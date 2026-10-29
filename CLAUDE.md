# 26_UROP — 안티드론 경량 세그멘테이션·추적 통합 파이프라인

> **이 저장소는 public이다.** 이 파일과 README를 포함해 git에 커밋되는 파일에는 미발표 실험 결과, 데이터 검증 과정의 내부 기록, 개인정보를 쓰지 않는다. 그런 내용은 비공개 노트(`notes/`, 별도 private 저장소, git 제외)에만 기록한다.

## 프로젝트
- ThinDyUNet(1.37M 파라미터) 기반 V2 세그멘테이션 파이프라인에 단일 표적 트래커를 결합하고, 이후 ONNX/TensorRT로 엣지 최적화한다.
- 연구 맥락(현재 단계, 평가 방침, 실험 기록)은 각 PC의 `CLAUDE.local.md`가 `notes/CLAUDE.md`를 불러와 제공한다. `notes/`가 없으면 이 파일의 공개 정보만으로 작업하고, 연구 판단이 필요한 일은 먼저 사용자에게 묻는다.

## 저장소 구조
- `Anti_UAV_Localization/src/` — 학습(`train_full.py`), 평가(`evaluate.py`), 트래킹 평가(`track_eval.py`), 트래커(`tracking/`)
- `Anti_UAV_Localization/configs/` — `train_config_full_dut.yaml`이 DUT 파인튜닝·test_dut 평가 기준 config
- git 밖 (PC마다 따로 둔다): `Anti_UAV_Localization/data/`, `Anti_UAV_Localization/checkpoints/`, `DUT-Anti-UAV/`, `Anti_UAV_Localization/results/tracking/_cache/`, `uav/`(외부 저장소)

## 자주 쓰는 명령 (`Anti_UAV_Localization` 폴더에서)
```bash
# V2 기준선 재현 — P 0.7280 / R 0.7671 / UAV IoU 0.5962 가 나와야 한다 (새 환경에서 가장 먼저 확인)
python src/evaluate.py --config configs/train_config_full_dut.yaml \
    --checkpoint checkpoints/frozen/v2_best_dut_v3.pth --split test_dut

# 세그멘테이션 + Kalman 트래커 평가 (확률맵은 results/tracking/_cache에 1회 캐시 후 재사용)
python src/track_eval.py --config configs/train_config_full_dut.yaml \
    --checkpoint checkpoints/frozen/v2_best_dut_v3.pth \
    --mode kalman --mask-mode expand --coast-output mask --score-mode mean \
    --init-thresh 0.9 --gate-dims 4 --gate-chi2 13.28 --run-name <실행이름>
```

## 작업 규칙
1. **수치 출처 표기**: 성능 수치에는 출처를 붙인다 — [선행논문] / [본 연구 실측](split·체크포인트·정밀도·threshold 명시) / [타 논문 보고치]. 평가 프로토콜이 다른 수치를 같은 표에서 직접 비교하지 않는다.
2. **속도 수치**에는 GPU, 정밀도(FP32/FP16/INT8), 배치, 입력 크기를 같이 적는다. 로컬(RTX 4070 Super)과 DL PC(RTX A5000) 수치를 섞지 않는다.
3. **동결 체크포인트**(`checkpoints/frozen/`)는 읽기 전용이다. 수정·이동·덮어쓰기 금지, 학습 config의 `save_dir`로 쓰지 않는다. 새 학습의 `save_dir`에는 PC와 날짜를 넣는다 (예: `checkpoints/a5000_20261029_v3a`).
4. **재현성**: 평가·트래커 코드에 옵션을 추가할 때 기본값은 기존 동작과 같게 한다. 이전 실행 결과가 그대로 재현되어야 한다.
5. **평가셋 사용**: 하이퍼파라미터·설정 선택에 최종 평가셋(test_dut)을 쓰지 않는다. 세부 방침은 비공개 노트를 따른다.
6. **장시간 작업**(학습, 전체 추론)은 포그라운드로 붙잡고 실행하지 않는다. `tmux`/`nohup`으로 시작하고 로그 파일로 진행을 확인한다. 시작 전 `nvidia-smi`로 GPU 점유를 확인한다.
7. **라벨**(SAM 2.1 마스크)은 다시 생성하지 않고 기존 파일을 복사해 쓴다. 다시 만들면 평가 GT가 바뀐다.
8. 불확실한 것은 추측하지 말고 묻는다. 코드는 실행 가능한 완전한 형태로 작성한다.

## 두 PC 동기화
- 작업 시작 전 `bash sync.sh pull` (Windows PowerShell: `.\sync.ps1 pull`) — 코드 저장소와 `notes/`를 함께 pull
- PC를 바꾸기 전 두 저장소 모두 커밋 후 `bash sync.sh push`
- 데이터·체크포인트는 git이 아니라 `rsync -a`로 옮기고 `sha256sum -c`로 대조한다
- 결정사항·실험 결과는 `notes/`에 기록한다. Claude Code의 auto memory는 PC마다 따로라 다른 PC로 전달되지 않는다
