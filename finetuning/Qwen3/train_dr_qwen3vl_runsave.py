# FILE: train_dr_qwen3vl_runsave.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import unsloth
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import re
import json
import math
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from hashlib import md5

import pandas as pd
import torch
from PIL import Image

from datasets import Dataset, DatasetDict, load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback

from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

def _print_label_distribution(name: str, ds: Dataset):
    labels = [int(x) for x in ds["label"]]
    s = pd.Series(labels)
    counts = s.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    out = pd.DataFrame({"count": counts, "pct": pct})
    print(f"\n[INFO] {name} label distribution:", flush=True)
    print(out.to_string(), flush=True)

class BalancedSFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if "label" not in self.train_dataset.column_names:
            print("[WARN] No 'label' column found in train_dataset. Falling back to default dataloader.", flush=True)
            return super().get_train_dataloader()

        labels = [int(x) for x in self.train_dataset["label"]]
        label_counts = Counter(labels)

        if len(label_counts) == 0:
            print("[WARN] Empty label counts. Falling back to default dataloader.", flush=True)
            return super().get_train_dataloader()

        class_weights = {label: 1.0 / count for label, count in label_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        print("\n[INFO] Using WeightedRandomSampler for train dataloader.", flush=True)
        print(f"[INFO] Train class counts: {dict(sorted(label_counts.items()))}", flush=True)
        print(f"[INFO] Train class weights: {dict(sorted(class_weights.items()))}", flush=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

INSTRUCTION = (
    "You are a medical imaging assistant.\n"
    "Grade diabetic retinopathy severity from this retinal fundus image.\n\n"
    "Scale:\n"
    "0 - No DR\n"
    "1 - Mild\n"
    "2 - Moderate\n"
    "3 - Severe\n"
    "4 - Proliferative DR\n\n"
    "Return exactly this format:\n"
    "<answer>X - LABEL</answer>"
)

LABEL_MAP = {
    0: "<answer>0 - No DR</answer>",
    1: "<answer>1 - Mild</answer>",
    2: "<answer>2 - Moderate</answer>",
    3: "<answer>3 - Severe</answer>",
    4: "<answer>4 - Proliferative DR</answer>",
}


def _cache_key(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return md5(s.encode("utf-8")).hexdigest()[:16]


def _parse_img_size(img_size: str) -> Tuple[int, int]:
    s = str(img_size).lower().replace(" ", "")
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    v = int(s)
    return v, v


def _resolve_image_path(image_dir: Path, row: pd.Series, ext: str) -> str:
    id_code = str(row.get("id_code", "")).strip()
    fp = row.get("file_path", None)
    fn = row.get("filename", None)

    candidates = []

    for raw in [fp, fn]:
        if raw is not None and str(raw).strip() != "":
            raw_s = str(raw).strip().replace("\\", "/")
            raw_p = Path(raw_s)

            if raw_p.is_absolute():
                candidates.append(raw_p)

            candidates.append(image_dir / raw_s)
            candidates.append(image_dir / raw_p.name)
            candidates.append(image_dir.parent / raw_s)
            candidates.append(image_dir.parent / raw_p.name)

    if id_code:
        candidates.append(image_dir / f"{id_code}{ext}")

    seen = set()
    dedup = []
    for c in candidates:
        cs = str(c)
        if cs not in seen:
            seen.add(cs)
            dedup.append(c)

    for c in dedup:
        try:
            if c.exists():
                return str(c)
        except Exception:
            continue

    return str(dedup[0]) if dedup else ""


def _load_csv_as_dataset(csv_path: str, image_dir: str, ext: str) -> Dataset:
    csv_path = str(csv_path)
    image_dir_p = Path(image_dir)

    df = pd.read_csv(csv_path, dtype=str)

    # -------------------------
    # Resolve image/id column
    # Backward compatible:
    # old: id_code, filename
    # new: image, image_id, image_path, img_path, file_path
    # -------------------------
    if "id_code" in df.columns:
        df["id_code"] = df["id_code"].astype(str)

    elif "filename" in df.columns:
        df["filename"] = df["filename"].astype(str)
        df["id_code"] = df["filename"].map(lambda s: Path(str(s)).stem)

    elif "image" in df.columns:
        df["filename"] = df["image"].astype(str)
        df["id_code"] = df["image"].map(lambda s: Path(str(s)).stem)

    elif "image_id" in df.columns:
        df["id_code"] = df["image_id"].astype(str)

    elif "image_path" in df.columns:
        df["file_path"] = df["image_path"].astype(str)
        df["id_code"] = df["image_path"].map(lambda s: Path(str(s)).stem)

    elif "img_path" in df.columns:
        df["file_path"] = df["img_path"].astype(str)
        df["id_code"] = df["img_path"].map(lambda s: Path(str(s)).stem)

    elif "file_path" in df.columns:
        df["file_path"] = df["file_path"].astype(str)
        df["id_code"] = df["file_path"].map(lambda s: Path(str(s)).stem)

    else:
        raise ValueError(
            f"CSV missing image/id column. Expected one of "
            f"'id_code', 'filename', 'image', 'image_id', 'image_path', 'img_path', 'file_path'. "
            f"Found: {list(df.columns)}"
        )

    # -------------------------
    # Resolve label column
    # Backward compatible:
    # old: diagnosis, dr_grade, grade
    # new: label, class, target
    # -------------------------
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(int)

    elif "dr_grade" in df.columns:
        df["diagnosis"] = df["dr_grade"].astype(int)

    elif "grade" in df.columns:
        df["diagnosis"] = df["grade"].astype(int)

    elif "label" in df.columns:
        df["diagnosis"] = df["label"].astype(int)

    elif "class" in df.columns:
        df["diagnosis"] = df["class"].astype(int)

    elif "target" in df.columns:
        df["diagnosis"] = df["target"].astype(int)

    else:
        raise ValueError(
            f"CSV missing label column. Expected one of "
            f"'diagnosis', 'dr_grade', 'grade', 'label', 'class', 'target'. "
            f"Found: {list(df.columns)}"
        )

    df["image_path"] = df.apply(lambda r: _resolve_image_path(image_dir_p, r, ext), axis=1)

    exists_mask = df["image_path"].apply(os.path.exists)
    missing = int((~exists_mask).sum())
    if missing:
        print(f"[WARN] Missing images in {Path(csv_path).name}: {missing}. Dropping them.", flush=True)

    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"0 rows remain after filtering missing images for {csv_path}.\n"
            f"Check --image_dir and whether the CSV paths are relative to a different base folder."
        )

    return Dataset.from_pandas(df)


def build_datasets(
    csv_path: str,
    image_dir: str,
    ext: str,
    test_size: float,
    seed: int,
    img_size: str,
    val_csv: str = "",
    test_csv: str = "",
    keep_cols: bool = False,
    cache_dir: str = "",
    preproc_num_proc: int = 0,
    preproc_load_from_cache_file: bool = True,
):
    csv_path = str(csv_path)
    image_dir = str(Path(image_dir))
    img_w, img_h = _parse_img_size(img_size)

    cache_root = Path(cache_dir) if cache_dir else None
    cache_payload = {
        "train_csv": csv_path,
        "val_csv": str(val_csv) if val_csv else "",
        "test_csv": str(test_csv) if test_csv else "",
        "image_dir": image_dir,
        "ext": ext,
        "test_size": test_size,
        "seed": seed,
        "img_size": img_size,
        "keep_cols": keep_cols,
        "schema_v": 4,
    }

    cache_path = None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"dr_qwen3vl_{_cache_key(cache_payload)}"
        if cache_path.exists():
            print(f"[INFO] Loading cached datasets from: {cache_path}", flush=True)
            dsdict = load_from_disk(str(cache_path))
            if "val" in dsdict and "test" in dsdict:
                return dsdict["train"], dsdict["val"], dsdict["test"]
            print("[WARN] Cache missing val/test. Rebuilding cache.", flush=True)

    if val_csv and test_csv:
        train_raw = _load_csv_as_dataset(csv_path, image_dir, ext)
        val_raw = _load_csv_as_dataset(val_csv, image_dir, ext)
        test_raw = _load_csv_as_dataset(test_csv, image_dir, ext)
    else:
        ds = _load_csv_as_dataset(csv_path, image_dir, ext)
        tmp = ds.train_test_split(test_size=0.10, seed=seed)
        trainval = tmp["train"]
        test_raw = tmp["test"]
        tmp2 = trainval.train_test_split(test_size=(1.0 / 9.0), seed=seed)
        train_raw = tmp2["train"]
        val_raw = tmp2["test"]

    def to_conversation(example):
        label = int(example["diagnosis"])
        answer = LABEL_MAP[label]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "image", "image": example["image_path"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                },
            ],
            "label": label,
            "image_path": example["image_path"],
            "img_w": img_w,
            "img_h": img_h,
        }

    remove_cols_train = [] if keep_cols else train_raw.column_names
    remove_cols_eval = [] if keep_cols else val_raw.column_names
    remove_cols_test = [] if keep_cols else test_raw.column_names

    num_proc = int(preproc_num_proc) if (preproc_num_proc and preproc_num_proc > 0) else None

    train_ds = train_raw.map(
        to_conversation,
        remove_columns=remove_cols_train,
        num_proc=num_proc,
        load_from_cache_file=preproc_load_from_cache_file,
        desc=f"Formatting train (num_proc={num_proc})" if num_proc else "Formatting train",
    )

    val_ds = val_raw.map(
        to_conversation,
        remove_columns=remove_cols_eval,
        num_proc=num_proc,
        load_from_cache_file=preproc_load_from_cache_file,
        desc=f"Formatting val (num_proc={num_proc})" if num_proc else "Formatting val",
    )

    test_ds = test_raw.map(
        to_conversation,
        remove_columns=remove_cols_test,
        num_proc=num_proc,
        load_from_cache_file=preproc_load_from_cache_file,
        desc=f"Formatting test (num_proc={num_proc})" if num_proc else "Formatting test",
    )

    if cache_path is not None:
        print(f"[INFO] Saving cached datasets to: {cache_path}", flush=True)
        if cache_path.exists():
            shutil.rmtree(cache_path)
        DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds}).save_to_disk(str(cache_path))

    _print_label_distribution("Train", train_ds)
    _print_label_distribution("Val", val_ds)
    _print_label_distribution("Test", test_ds)

    return train_ds, val_ds, test_ds


def _make_sft_config(args, num_train_epochs: int, max_steps):
    sft_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
        save_total_limit=args.save_total_limit,
    )

    if args.eval_strategy != "no":
        if args.eval_steps and args.eval_steps > 0:
            sft_kwargs["eval_steps"] = args.eval_steps

    if args.save_strategy != "no":
        if args.save_steps and args.save_steps > 0:
            sft_kwargs["save_steps"] = args.save_steps

    if args.early_stopping:
        sft_kwargs.update(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    try:
        tmp = dict(sft_kwargs)
        if args.eval_strategy != "no":
            tmp["evaluation_strategy"] = args.eval_strategy
        if args.save_strategy != "no":
            tmp["save_strategy"] = args.save_strategy
        return SFTConfig(**tmp)
    except TypeError:
        tmp = dict(sft_kwargs)
        if args.eval_strategy != "no":
            tmp["eval_strategy"] = args.eval_strategy
        if args.save_strategy != "no":
            tmp["save_strategy"] = args.save_strategy
        return SFTConfig(**tmp)


class BestEpochTrackerCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = None
        self.best_epoch = None
        self.best_step = None
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        eval_loss = metrics.get("eval_loss", None)
        epoch = state.epoch
        step = state.global_step

        self.history.append({
            "step": step,
            "epoch": float(epoch) if epoch is not None else None,
            "eval_loss": float(eval_loss) if eval_loss is not None else None,
        })

        if eval_loss is not None:
            if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                self.best_eval_loss = float(eval_loss)
                self.best_epoch = float(epoch) if epoch is not None else None
                self.best_step = int(step)


def _safe_run_name(lr: float, best_epoch: Optional[float], eff_batch: int, prefix: str = "") -> str:
    lr_str = f"{lr:.10f}".rstrip("0").rstrip(".")
    if best_epoch is None:
        ep_str = "unknown"
    else:
        ep_str = str(int(round(best_epoch)))
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(f"lr{lr_str}")
    parts.append(f"epoch{ep_str}")
    parts.append(f"batchsize{eff_batch}")
    return "_".join(parts)


def _copy_best_checkpoint_to_final_lora(best_ckpt_dir: Path, final_lora_dir: Path):
    final_lora_dir.mkdir(parents=True, exist_ok=True)

    needed = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
    ]

    copied_any = False
    for name in needed:
        src = best_ckpt_dir / name
        if src.exists():
            shutil.copy2(src, final_lora_dir / name)
            copied_any = True

    if not copied_any:
        raise FileNotFoundError(
            f"No LoRA/tokenizer files were found in best checkpoint folder: {best_ckpt_dir}"
        )


def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--val_csv", type=str, default="")
    p.add_argument("--test_csv", type=str, default="")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--ext", type=str, default=".png")
    p.add_argument("--img_size", type=str, default="384")
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Preprocessing
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--preproc_num_proc", type=int, default=0)
    p.add_argument("--preproc_load_from_cache_file", action="store_true")
    p.add_argument("--no_preproc_load_from_cache_file", action="store_true")

    # Model
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--runs_root", type=str, required=True)
    p.add_argument("--run_name_prefix", type=str, default="")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_load_in_4bit", action="store_true")

    # Train
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--scheduler", type=str, default="linear")
    p.add_argument("--optim", type=str, default="adamw_8bit")
    p.add_argument("--max_length", type=int, default=2048)

    # Eval/checkpoints
    p.add_argument("--eval_strategy", type=str, default="epoch")
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=2)

    # Early stopping
    p.add_argument("--early_stopping", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)


    args = p.parse_args()

    preproc_load_from_cache_file = True
    if args.no_preproc_load_from_cache_file:
        preproc_load_from_cache_file = False
    if args.preproc_load_from_cache_file:
        preproc_load_from_cache_file = True

    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    if args.load_in_4bit:
        load_in_4bit = True

    output_dir = Path(args.output_dir)
    runs_root = Path(args.runs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building datasets.", flush=True)
    train_ds, val_ds, test_ds = build_datasets(
        csv_path=args.csv_path,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        ext=args.ext,
        test_size=args.test_size,
        seed=args.seed,
        img_size=args.img_size,
        keep_cols=False,
        cache_dir=args.cache_dir,
        preproc_num_proc=args.preproc_num_proc,
        preproc_load_from_cache_file=preproc_load_from_cache_file,
    )

    print(f"[INFO] Train rows: {len(train_ds)}", flush=True)
    print(f"[INFO] Val rows:   {len(val_ds)}", flush=True)
    print(f"[INFO] Test rows:  {len(test_ds)}", flush=True)

    print("[INFO] Loading model.", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=load_in_4bit,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_r if hasattr(args, "lora_r") else 16,
        lora_alpha=args.lora_alpha if hasattr(args, "lora_alpha") else 16,
        lora_dropout=args.lora_dropout if hasattr(args, "lora_dropout") else 0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )
    FastVisionModel.for_training(model)

    gpu = torch.cuda.get_device_properties(0)
    print(f"[INFO] GPU: {gpu.name} | VRAM: {gpu.total_memory/1024**3:.2f} GB", flush=True)

    callbacks = []
    tracker = BestEpochTrackerCallback()
    callbacks.append(tracker)

    if args.early_stopping:
        if args.eval_strategy == "no":
            raise ValueError("Early stopping requires evaluation. Set --eval_strategy epoch or steps.")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    num_train_epochs = args.epochs
    max_steps = None
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = 1

    sft_args = _make_sft_config(args, num_train_epochs=num_train_epochs, max_steps=max_steps)

    trainer = BalancedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds if (args.eval_strategy != "no") else None,
        args=sft_args,
        callbacks=callbacks,
    )

    print("[INFO] Training.", flush=True)
    stats = trainer.train()
    print("[INFO] Training done.", flush=True)
    print(stats, flush=True)

    best_ckpt_path = trainer.state.best_model_checkpoint
    best_epoch = tracker.best_epoch
    best_step = tracker.best_step
    best_eval_loss = tracker.best_eval_loss

    if best_ckpt_path is None:
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        best_ckpt_path = str(ckpts[-1])

    best_ckpt_dir = Path(best_ckpt_path)
    effective_batch_size = int(args.batch_size) * int(args.grad_accum)

    final_run_name = _safe_run_name(
        lr=args.lr,
        best_epoch=best_epoch,
        eff_batch=effective_batch_size,
        prefix=args.run_name_prefix,
    )
    final_run_dir = runs_root / final_run_name
    final_lora_dir = final_run_dir / "best_lora"

    if final_run_dir.exists():
        raise FileExistsError(f"Run folder already exists: {final_run_dir}")

    final_run_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] Best checkpoint: {best_ckpt_dir}", flush=True)
    print(f"[INFO] Final run dir:   {final_run_dir}", flush=True)

    _copy_best_checkpoint_to_final_lora(best_ckpt_dir, final_lora_dir)

    run_info = {
        "created_at": datetime.now().isoformat(),
        "model_name": args.model_name,
        "train_csv": str(args.csv_path),
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv),
        "image_dir": str(args.image_dir),
        "ext": args.ext,
        "img_size": args.img_size,
        "lr": args.lr,
        "epochs_requested": args.epochs,
        "batch_size_per_device": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": effective_batch_size,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_eval_loss": best_eval_loss,
        "best_checkpoint_dir": str(best_ckpt_dir),
        "output_dir": str(output_dir),
        "final_run_dir": str(final_run_dir),
        "final_lora_dir": str(final_lora_dir),
        "history": tracker.history,
    }

    with open(final_run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"[INFO] Saved final best LoRA to: {final_lora_dir}", flush=True)
    print(f"[INFO] Saved run info to:       {final_run_dir / 'run_info.json'}", flush=True)


if __name__ == "__main__":
    main()