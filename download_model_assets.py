import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent

RETIZERO_TARGET = PROJECT_ROOT / "RetiZero" / "best_retizero_dr.pth"
QWEN3_LORA_TARGET = PROJECT_ROOT / "Qwen3" / "lr0.0005_epoch4_batchsize8" / "best_lora"

DEFAULT_RETIZERO_REPO = "moht24/retizero-dr-model"
DEFAULT_RETIZERO_FILE = "best_retizero_dr.pth"

DEFAULT_QWEN3_REPO = "moht24/qwen3-vl-dr-lora"
DEFAULT_QWEN3_SUBDIR = "best_lora"
DEFAULT_QWEN3_BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

# Strict expectations for this project.
REQUIRED_RETIZERO_FILENAME = "best_retizero_dr.pth"
REQUIRED_QWEN3_REQUIRED_FILES = {
    "adapter_config.json",
}
REQUIRED_QWEN3_WEIGHT_CANDIDATES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)

LOGGER = logging.getLogger("download_model_assets")


class AssetDownloadError(RuntimeError):
    """Raised when a downloaded asset does not match the expected project layout."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Afterimage model assets from Hugging Face into the exact "
            "local paths expected by this project."
        )
    )
    parser.add_argument(
        "--retizero-repo",
        default=DEFAULT_RETIZERO_REPO,
        help=f"Hugging Face repo id for the RetiZero checkpoint (default: {DEFAULT_RETIZERO_REPO})",
    )
    parser.add_argument(
        "--retizero-file",
        default=DEFAULT_RETIZERO_FILE,
        help=f"Filename inside the RetiZero repo (default: {DEFAULT_RETIZERO_FILE})",
    )
    parser.add_argument(
        "--qwen3-repo",
        default=DEFAULT_QWEN3_REPO,
        help=f"Hugging Face repo id for the Qwen3 LoRA adapter (default: {DEFAULT_QWEN3_REPO})",
    )
    parser.add_argument(
        "--qwen3-subdir",
        default=DEFAULT_QWEN3_SUBDIR,
        help=f"Folder inside the Qwen3 repo containing the LoRA files (default: {DEFAULT_QWEN3_SUBDIR})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Optional Hugging Face token. If omitted, the script uses the current "
            "login session or HF_TOKEN if available."
        ),
    )
    parser.add_argument(
        "--clear-qwen3-target",
        action="store_true",
        help="Delete the existing local best_lora folder before copying the downloaded files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise AssetDownloadError(f"{description} was not found at: {path}")
    if not path.is_file():
        raise AssetDownloadError(f"{description} is not a file: {path}")
    if path.stat().st_size == 0:
        raise AssetDownloadError(f"{description} is empty: {path}")


def validate_directory_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise AssetDownloadError(f"{description} was not found at: {path}")
    if not path.is_dir():
        raise AssetDownloadError(f"{description} is not a directory: {path}")


def validate_retizero_filename(filename: str) -> None:
    actual_name = Path(filename).name
    if actual_name != REQUIRED_RETIZERO_FILENAME:
        raise AssetDownloadError(
            "This project expects the RetiZero checkpoint filename to be exactly "
            f"'{REQUIRED_RETIZERO_FILENAME}', but got '{actual_name}'."
        )


def validate_qwen3_contents(source_dir: Path) -> None:
    validate_directory_exists(source_dir, "Qwen3 LoRA source directory")

    missing = [name for name in REQUIRED_QWEN3_REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise AssetDownloadError(
            "The Qwen3 LoRA folder is missing required file(s): "
            + ", ".join(sorted(missing))
        )

    found_weight_file = None
    for candidate in REQUIRED_QWEN3_WEIGHT_CANDIDATES:
        candidate_path = source_dir / candidate
        if candidate_path.is_file():
            found_weight_file = candidate_path
            break

    if found_weight_file is None:
        raise AssetDownloadError(
            "The Qwen3 LoRA folder must contain one of these weight files: "
            + ", ".join(REQUIRED_QWEN3_WEIGHT_CANDIDATES)
        )

    if found_weight_file.stat().st_size == 0:
        raise AssetDownloadError(f"Qwen3 LoRA weight file is empty: {found_weight_file}")

    LOGGER.debug("Validated Qwen3 LoRA source directory: %s", source_dir)


def safe_copy_file(src: Path, dst: Path) -> None:
    validate_file_exists(src, "Source file")
    ensure_parent(dst)
    shutil.copy2(src, dst)
    validate_file_exists(dst, "Copied file")


def _iter_source_items(src_dir: Path) -> Iterable[Path]:
    for item in src_dir.iterdir():
        yield item


def safe_copy_tree(src_dir: Path, dst_dir: Path, clear_first: bool) -> None:
    validate_directory_exists(src_dir, "Source directory")

    if clear_first and dst_dir.exists():
        LOGGER.info("Clearing existing Qwen3 target directory: %s", dst_dir)
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in _iter_source_items(src_dir):
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)

    validate_directory_exists(dst_dir, "Copied target directory")


def download_retizero(repo_id: str, filename: str, token: str | None) -> None:
    validate_retizero_filename(filename)

    LOGGER.info("Downloading RetiZero weights from %s:%s", repo_id, filename)
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
    )

    validate_file_exists(cached_path, "Downloaded RetiZero checkpoint")
    if cached_path.name != REQUIRED_RETIZERO_FILENAME:
        raise AssetDownloadError(
            "Downloaded RetiZero file name does not match the required project name. "
            f"Expected '{REQUIRED_RETIZERO_FILENAME}', got '{cached_path.name}'."
        )

    safe_copy_file(cached_path, RETIZERO_TARGET)

    if RETIZERO_TARGET.name != REQUIRED_RETIZERO_FILENAME:
        raise AssetDownloadError(
            "Local RetiZero target filename is incorrect. "
            f"Expected '{REQUIRED_RETIZERO_FILENAME}', got '{RETIZERO_TARGET.name}'."
        )

    LOGGER.info("Saved RetiZero weights to %s", RETIZERO_TARGET)


def normalize_subdir(subdir: str) -> str:
    normalized = subdir.strip().strip("/\\")
    if normalized in {".", "root"}:
        return ""
    return normalized


def download_qwen3_lora(repo_id: str, subdir: str, token: str | None, clear_first: bool) -> None:
    normalized_subdir = normalize_subdir(subdir)
    allow_patterns = None if not normalized_subdir else [f"{normalized_subdir}/*"]

    source_desc = repo_id if not normalized_subdir else f"{repo_id}:{normalized_subdir}"
    LOGGER.info("Downloading Qwen3 LoRA assets from %s", source_desc)

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            token=token,
        )
    )

    source_dir = snapshot_dir if not normalized_subdir else snapshot_dir / normalized_subdir
    validate_qwen3_contents(source_dir)

    safe_copy_tree(source_dir, QWEN3_LORA_TARGET, clear_first=clear_first)
    validate_qwen3_contents(QWEN3_LORA_TARGET)

    LOGGER.info("Saved Qwen3 LoRA files to %s", QWEN3_LORA_TARGET)


def print_summary() -> None:
    LOGGER.info("Model asset download complete.")
    LOGGER.info("RetiZero checkpoint path: %s", RETIZERO_TARGET)
    LOGGER.info("Qwen3 LoRA path         : %s", QWEN3_LORA_TARGET)
    LOGGER.info(
        "Base Qwen3 model is not downloaded by this script. "
        "It should be loaded separately by model id: %s",
        DEFAULT_QWEN3_BASE_MODEL,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        download_retizero(args.retizero_repo, args.retizero_file, args.token)
        download_qwen3_lora(
            args.qwen3_repo,
            args.qwen3_subdir,
            args.token,
            args.clear_qwen3_target,
        )
        print_summary()
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Operation cancelled by user.")
        return 130
    except Exception as exc:
        LOGGER.error("Asset setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())