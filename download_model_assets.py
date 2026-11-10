import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from model_assets import (
    AssetValidationError,
    DEFAULT_QWEN3_BASE_MODEL,
    DEFAULT_RETIZERO_TEXT_MODEL,
    QWEN3_LORA_TARGET,
    RETIZERO_TARGET,
    validate_lora,
    validate_snapshot,
)

DEFAULT_RETIZERO_REPO = "moht24/retizero-dr-model"
DEFAULT_RETIZERO_FILE = "best_retizero_dr.pth"

DEFAULT_QWEN3_REPO = "moht24/qwen3-vl-dr-lora"
DEFAULT_QWEN3_SUBDIR = "best_lora"

# Strict expectations for this project.
REQUIRED_RETIZERO_FILENAME = "best_retizero_dr.pth"

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
        "--qwen3-base-repo",
        default=DEFAULT_QWEN3_BASE_MODEL,
        help=(
            "Hugging Face repo id for the Qwen3-VL base model, stored in the standard "
            f"Hugging Face cache (default: {DEFAULT_QWEN3_BASE_MODEL})"
        ),
    )
    parser.add_argument(
        "--retizero-text-repo",
        default=DEFAULT_RETIZERO_TEXT_MODEL,
        help=f"RetiZero text encoder cached for startup (default: {DEFAULT_RETIZERO_TEXT_MODEL})",
    )
    parser.add_argument(
        "--skip-retizero-text",
        action="store_true",
        help="Skip Bio_ClinicalBERT; RetiZero may need internet on first load.",
    )
    parser.add_argument(
        "--skip-qwen3-base",
        action="store_true",
        help="Do not download the Qwen3-VL base model (it is then fetched on first load instead).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download every asset even if it is already present locally.",
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
    try:
        validate_lora(source_dir)
    except AssetValidationError as exc:
        raise AssetDownloadError(str(exc)) from exc
    LOGGER.debug("Validated Qwen3 LoRA source directory: %s", source_dir)


def safe_copy_file(src: Path, dst: Path) -> None:
    validate_file_exists(src, "Source file")
    ensure_parent(dst)
    # An interrupted copy must not leave a non-empty final file that gets skipped.
    with tempfile.NamedTemporaryFile(dir=dst.parent, prefix=f".{dst.name}.", suffix=".partial", delete=False) as temp:
        staging = Path(temp.name)
    try:
        shutil.copy2(src, staging)
        staging.replace(dst)
    finally:
        staging.unlink(missing_ok=True)
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
            safe_copy_file(item, dst_item)

    validate_directory_exists(dst_dir, "Copied target directory")


def retizero_is_present() -> bool:
    return RETIZERO_TARGET.is_file() and RETIZERO_TARGET.stat().st_size > 0


def qwen3_lora_is_present() -> bool:
    try:
        validate_qwen3_contents(QWEN3_LORA_TARGET)
    except AssetDownloadError:
        return False
    return True


def find_cached_model(repo_id: str, kind: str) -> Path | None:
    """Inspect the current local snapshot without contacting Hugging Face."""
    # A local directory override is supported by the servers as well as setup.
    local_path = Path(repo_id).expanduser()
    try:
        snapshot_dir = local_path if local_path.is_dir() else Path(
            snapshot_download(repo_id=repo_id, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        return None
    try:
        validate_snapshot(snapshot_dir, kind)
        return snapshot_dir
    except AssetValidationError as exc:
        LOGGER.debug("Incomplete cached snapshot: %s", exc)
        return None


def find_cached_qwen3_base(repo_id: str) -> Path | None:
    return find_cached_model(repo_id, "qwen3")


def download_qwen3_base(repo_id: str, token: str | None, force: bool) -> None:
    download_cached_model(repo_id, "qwen3", token, force)


def download_cached_model(repo_id: str, kind: str, token: str | None, force: bool) -> None:
    label = "Qwen3 base model" if kind == "qwen3" else "RetiZero text encoder"
    if Path(repo_id).expanduser().is_dir():
        validate_snapshot(Path(repo_id).expanduser(), kind)
        LOGGER.info("%s supplied as a local directory: %s", label, repo_id)
        return
    if not force:
        cached = find_cached_model(repo_id, kind)
        if cached is not None:
            LOGGER.info("%s %s already downloaded (%s), skipping.", label, repo_id, cached)
            return

    LOGGER.info("Downloading %s %s; large models can take several minutes", label, repo_id)
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            token=token,
            force_download=force,
        )
    )

    validate_snapshot(snapshot_dir, kind)
    LOGGER.info("%s available in the Hugging Face cache: %s", label, snapshot_dir)


def download_retizero(repo_id: str, filename: str, token: str | None, force: bool) -> None:
    validate_retizero_filename(filename)

    if not force and retizero_is_present():
        LOGGER.info("RetiZero weights already present at %s, skipping.", RETIZERO_TARGET)
        return

    LOGGER.info("Downloading RetiZero weights from %s:%s", repo_id, filename)
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            force_download=force,
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


def download_qwen3_lora(
    repo_id: str, subdir: str, token: str | None, clear_first: bool, force: bool
) -> None:
    # --clear-qwen3-target explicitly asks for a fresh copy, so it also bypasses the skip.
    if not (force or clear_first) and qwen3_lora_is_present():
        LOGGER.info("Qwen3 LoRA already present at %s, skipping.", QWEN3_LORA_TARGET)
        return

    normalized_subdir = normalize_subdir(subdir)
    allow_patterns = None if not normalized_subdir else [f"{normalized_subdir}/*"]

    source_desc = repo_id if not normalized_subdir else f"{repo_id}:{normalized_subdir}"
    LOGGER.info("Downloading Qwen3 LoRA assets from %s", source_desc)

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            token=token,
            force_download=force,
        )
    )

    source_dir = snapshot_dir if not normalized_subdir else snapshot_dir / normalized_subdir
    validate_qwen3_contents(source_dir)

    safe_copy_tree(source_dir, QWEN3_LORA_TARGET, clear_first=clear_first)
    validate_qwen3_contents(QWEN3_LORA_TARGET)

    LOGGER.info("Saved Qwen3 LoRA files to %s", QWEN3_LORA_TARGET)


def print_summary(args: argparse.Namespace) -> None:
    LOGGER.info("Model asset setup complete.")
    LOGGER.info("RetiZero checkpoint path: %s", RETIZERO_TARGET)
    LOGGER.info("Qwen3 LoRA path         : %s", QWEN3_LORA_TARGET)
    LOGGER.info("RetiZero text encoder   : %s%s", args.retizero_text_repo,
                " (skipped)" if args.skip_retizero_text else " (Hugging Face cache)")
    if args.skip_qwen3_base:
        LOGGER.info(
            "Qwen3 base model download was skipped; it will be fetched on first load by model id: %s",
            args.qwen3_base_repo,
        )
    else:
        LOGGER.info("Qwen3 base model        : %s (Hugging Face cache)", args.qwen3_base_repo)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        download_retizero(args.retizero_repo, args.retizero_file, args.token, args.force)
        download_qwen3_lora(
            args.qwen3_repo,
            args.qwen3_subdir,
            args.token,
            args.clear_qwen3_target,
            args.force,
        )
        if not args.skip_qwen3_base:
            download_qwen3_base(args.qwen3_base_repo, args.token, args.force)
        if not args.skip_retizero_text:
            download_cached_model(args.retizero_text_repo, "retizero-text", args.token, args.force)
        print_summary(args)
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Operation cancelled by user.")
        return 130
    except Exception as exc:
        LOGGER.error("Asset setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
