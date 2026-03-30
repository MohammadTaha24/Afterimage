import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
RETIZERO_TARGET = PROJECT_ROOT / "RetiZero" / "best_retizero_dr.pth"
QWEN3_LORA_TARGET = PROJECT_ROOT / "Qwen3" / "lr0.0005_epoch4_batchsize8" / "best_lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Afterimage model assets from Hugging Face into the expected local paths."
    )
    parser.add_argument(
        "--retizero-repo",
        required=True,
        help="Hugging Face repo id that contains best_retizero_dr.pth",
    )
    parser.add_argument(
        "--retizero-file",
        default="best_retizero_dr.pth",
        help="Filename inside the RetiZero repo (default: best_retizero_dr.pth)",
    )
    parser.add_argument(
        "--qwen3-repo",
        required=True,
        help="Hugging Face repo id that contains the Qwen3 LoRA files",
    )
    parser.add_argument(
        "--qwen3-subdir",
        default="best_lora",
        help="Folder inside the Qwen3 repo that contains the LoRA adapter files (default: best_lora)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for private repos",
    )
    parser.add_argument(
        "--clear-qwen3-target",
        action="store_true",
        help="Delete the existing local best_lora folder before copying the downloaded files",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def copy_tree(src_dir: Path, dst_dir: Path, clear_first: bool) -> None:
    if clear_first and dst_dir.exists():
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def download_retizero(repo_id: str, filename: str, token: str | None) -> None:
    print(f"Downloading RetiZero weights from {repo_id}:{filename}")
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
    )
    copy_file(cached_path, RETIZERO_TARGET)
    print(f"Saved RetiZero weights to {RETIZERO_TARGET}")


def download_qwen3_lora(repo_id: str, subdir: str, token: str | None, clear_first: bool) -> None:
    subdir = subdir.strip().strip("/\\")
    if subdir in {".", "root"}:
        subdir = ""
    allow_patterns = None if not subdir else [f"{subdir}/*"]

    print(
        "Downloading Qwen3 LoRA assets from "
        f"{repo_id}{(':' + subdir) if subdir else ''}"
    )
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            token=token,
        )
    )

    source_dir = snapshot_dir if not subdir else snapshot_dir / subdir
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Expected LoRA folder '{subdir}' inside repo '{repo_id}', but it was not found."
        )

    copy_tree(source_dir, QWEN3_LORA_TARGET, clear_first=clear_first)
    print(f"Saved Qwen3 LoRA files to {QWEN3_LORA_TARGET}")


def main() -> None:
    args = parse_args()
    download_retizero(args.retizero_repo, args.retizero_file, args.token)
    download_qwen3_lora(args.qwen3_repo, args.qwen3_subdir, args.token, args.clear_qwen3_target)
    print("Model asset download complete.")


if __name__ == "__main__":
    main()
