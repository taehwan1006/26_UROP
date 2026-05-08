"""
데이터셋 압축 해제 스크립트.
UAVSemanticSegmentationInput.tar.gz (이미지)와
UAVSemanticSegmentationLabels.tar.gz (마스크)를 해제하여
data/ 폴더에 정리한다.
"""

import argparse
import tarfile
from pathlib import Path


def extract_tar(tar_path: Path, dest_dir: Path):
    """tar 파일을 dest_dir에 해제한다. gzip/plain 자동 감지."""
    print(f"Extracting {tar_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # "r:*" 모드는 gzip/bz2/xz/plain 등을 자동 감지한다
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=dest_dir)
    print(f"Done: {tar_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract UAV segmentation dataset")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parent.parent

    input_tar = root / "UAVSemanticSegmentationInput.tar.gz"
    labels_tar = root / "UAVSemanticSegmentationLabels.tar.gz"
    data_dir = root / "data" / "raw"

    if not input_tar.exists():
        print(f"ERROR: {input_tar} not found")
        return
    if not labels_tar.exists():
        print(f"ERROR: {labels_tar} not found")
        return

    extract_tar(input_tar, data_dir / "images")
    extract_tar(labels_tar, data_dir / "masks")

    # 해제 후 구조 확인
    print("\n--- Extracted structure ---")
    for p in sorted(data_dir.rglob("*")):
        if p.is_dir():
            n_files = sum(1 for f in p.iterdir() if f.is_file())
            print(f"  [DIR] {p.relative_to(data_dir)}  ({n_files} files)")


if __name__ == "__main__":
    main()
