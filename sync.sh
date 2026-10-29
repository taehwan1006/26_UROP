#!/usr/bin/env bash
# 코드 저장소(26_UROP)와 비공개 노트 저장소(notes/)를 함께 pull / push / status 한다.
# 사용: bash sync.sh [pull|push|status]   (기본: status)
set -euo pipefail
cd "$(dirname "$0")"

cmd="${1:-status}"
repos=(".")
if [ -d notes/.git ]; then
  repos+=("notes")
else
  echo "[warn] notes/ 저장소가 없습니다 — 코드 저장소만 처리합니다"
fi

for r in "${repos[@]}"; do
  echo "== ${r}"
  case "$cmd" in
    pull)
      git -C "$r" pull --ff-only
      ;;
    push)
      if [ -n "$(git -C "$r" status --porcelain)" ]; then
        echo "  커밋하지 않은 변경이 있습니다. 먼저 커밋하세요:"
        git -C "$r" status --short
        exit 1
      fi
      git -C "$r" push
      ;;
    status)
      git -C "$r" status --short --branch
      ;;
    *)
      echo "usage: bash sync.sh [pull|push|status]"
      exit 2
      ;;
  esac
done
