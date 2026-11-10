"""Portable model locations and lightweight validation shared by setup and runtime."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RETIZERO_TARGET = PROJECT_ROOT / "RetiZero" / "best_retizero_dr.pth"
QWEN3_RUN = PROJECT_ROOT / "Qwen3" / "lr0.0005_epoch4_batchsize8"
QWEN3_LORA_TARGET = QWEN3_RUN / "best_lora"
DEFAULT_QWEN3_BASE_MODEL = os.environ.get(
    "QWEN3_BASE_MODEL", "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
)
DEFAULT_RETIZERO_TEXT_MODEL = os.environ.get(
    "RETIZERO_TEXT_MODEL", "emilyalsentzer/Bio_ClinicalBERT"
)
LORA_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


class AssetValidationError(ValueError):
    """An asset is absent, empty, malformed, or incomplete."""


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssetValidationError(f"Missing or empty file: {path}")


def read_json(path: Path) -> dict:
    require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise AssetValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssetValidationError(f"Expected a JSON object in {path}")
    return value


def validate_lora(path: Path) -> None:
    config = read_json(path / "adapter_config.json")
    if not config.get("peft_type") or not config.get("base_model_name_or_path"):
        raise AssetValidationError(f"Incomplete adapter_config.json in {path}")
    if not any((path / name).is_file() and (path / name).stat().st_size > 0 for name in LORA_WEIGHT_NAMES):
        raise AssetValidationError(f"Missing non-empty adapter weights in {path}")


def _validate_weights(path: Path) -> None:
    # A single shard is not proof that a sharded model finished downloading.
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / name
        if index_path.exists():
            weight_map = read_json(index_path).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise AssetValidationError(f"Missing weight_map in {index_path}")
            for shard in set(weight_map.values()):
                if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise AssetValidationError(f"Invalid shard name in {index_path}: {shard!r}")
                require_file(path / shard)
            return
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return
    raise AssetValidationError(f"Missing complete model weights in {path}")


def validate_snapshot(path: Path, kind: str) -> None:
    """Check files needed at startup, without loading multi-GB tensors.

    This detects missing/empty files and invalid JSON, not arbitrary weight corruption.
    Use the downloader's --force to repair suspected corrupt non-empty weights.
    """
    read_json(path / "config.json")
    _validate_weights(path)
    if kind == "qwen3":
        read_json(path / "tokenizer_config.json")
        read_json(path / "tokenizer.json")
        # Transformers versions use either name for the image processor config.
        configs = [path / name for name in ("preprocessor_config.json", "processor_config.json")]
        if not any(config.is_file() for config in configs):
            raise AssetValidationError(f"Missing image processor configuration in {path}")
        for config in configs:
            if config.is_file():
                read_json(config)
    elif kind == "retizero-text":
        require_file(path / "vocab.txt")
    else:
        raise ValueError(f"Unknown snapshot kind: {kind}")
