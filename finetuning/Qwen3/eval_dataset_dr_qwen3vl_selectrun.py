# FILE: eval_dataset_dr_qwen3vl_selectrun.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import unsloth
from unsloth import FastVisionModel
from peft import PeftModel

# Reuse your existing eval logic for generation + parsing + qwk fallback
from eval_infer_dr_qwen3vl import (
    predict_one,
    parse_pred_label,
    qwk_numpy,
    SKLEARN_OK,
)

try:
    from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
except Exception:
    confusion_matrix = None
    classification_report = None
    cohen_kappa_score = None


def resolve_lora_dir(run_or_lora_path: str) -> Path:
    p = Path(run_or_lora_path)
    if not p.exists():
        raise FileNotFoundError(f"Run path does not exist: {p}")

    # direct adapter folder
    if p.is_dir() and (p / "adapter_config.json").exists():
        return p

    # run folder containing best_lora/
    best = p / "best_lora"
    if best.is_dir() and (best / "adapter_config.json").exists():
        return best

    raise FileNotFoundError(
        f"Could not resolve a LoRA adapter folder from: {p}\n"
        f"Expected adapter_config.json either in the given folder or in best_lora/."
    )


def _pick_label_col(cols: set[str]) -> str:
    for c in ["diagnosis", "dr_grade", "grade", "label", "y", "target", "class"]:
        if c in cols:
            return c
    raise ValueError("CSV missing label column. Expected diagnosis/dr_grade/grade/label.")


def _pick_id_col(cols: set[str]) -> str:
    for c in ["filename", "id_code", "image", "image_id", "file", "path"]:
        if c in cols:
            return c
    raise ValueError("CSV missing id column. Expected filename or id_code (or similar).")


def load_eval_rows(csv_path: str, image_dir: str, ext: str) -> list[dict]:
    """
    Supports BOTH:
      A) image_path + label
      B) filename/id_code + (diagnosis/dr_grade/grade/label)

    Returns list of dicts: {"image_path": <abs_path>, "diagnosis": int}
    """
    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    label_col = _pick_label_col(cols)

    # Case A: already has image_path
    if "image_path" in cols:
        def to_abs(p):
            p = str(p)
            if p.strip() == "":
                return ""
            pp = Path(p)
            return str(pp) if pp.is_absolute() else str(Path(image_dir) / p)

        df["__img__"] = df["image_path"].astype(str).map(to_abs)

    # Case B: build from filename/id_code
    else:
        id_col = _pick_id_col(cols)

        def build_path(v):
            v = str(v)
            if v.strip() == "":
                return ""
            pv = Path(v)
            if pv.is_absolute():
                return str(pv)
            # If no extension, add ext
            if pv.suffix == "":
                return str(Path(image_dir) / f"{v}{ext}")
            return str(Path(image_dir) / v)

        df["__img__"] = df[id_col].astype(str).map(build_path)

    # normalize labels
    df["__y__"] = df[label_col].astype(int)

    # drop missing images
    exists = df["__img__"].map(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"[WARN] Missing images: {missing}. Dropping them.", flush=True)
    df = df[exists].reset_index(drop=True)

    return [{"image_path": r["__img__"], "diagnosis": int(r["__y__"])} for _, r in df.iterrows()]


def main():
    p = argparse.ArgumentParser()

    # Run selection
    p.add_argument("--run_path", type=str, required=True,
                   help="Either a run folder (containing best_lora/) or a direct adapter folder.")
    p.add_argument("--base_model", type=str,
                   default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_load_in_4bit", action="store_true")

    # Data (keep BOTH flags to match your notebook)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--test_csv", type=str, default="")  # backward compatible
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--ext", type=str, default=".png")
    p.add_argument("--limit", type=int, default=0)

    # Generation config
    p.add_argument("--img_size", type=str, default="384")
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--min_p", type=float, default=0.1)

    # Outputs
    p.add_argument("--save_preds_csv", type=str, default="")
    args = p.parse_args()

    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    if args.load_in_4bit:
        load_in_4bit = True

    # Prefer --test_csv if provided, else fall back to --csv_path
    data_csv = args.test_csv if args.test_csv else args.csv_path

    lora_dir = resolve_lora_dir(args.run_path)

    print("[INFO] Loading base model.", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=load_in_4bit,
    )

    print(f"[INFO] Attaching LoRA from: {lora_dir}", flush=True)
    model = PeftModel.from_pretrained(model, str(lora_dir))
    FastVisionModel.for_inference(model)

    print(f"[INFO] Loading CSV: {data_csv}", flush=True)
    eval_rows = load_eval_rows(data_csv, args.image_dir, args.ext)

    n = len(eval_rows)
    if args.limit and args.limit > 0:
        n = min(n, args.limit)

    print(f"[INFO] Evaluating {n} samples | img_size={args.img_size}", flush=True)

    y_true, y_pred = [], []
    out_rows = []
    bad = 0

    for i in range(n):
        row = eval_rows[i]
        gt = int(row["diagnosis"])
        img_path = row["image_path"]

        gen = predict_one(
            model, tokenizer, img_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            img_size=args.img_size,
        )

        pred = parse_pred_label(gen)
        if pred is None:
            bad += 1
            pred = 0

        y_true.append(gt)
        y_pred.append(pred)

        out_rows.append({
            "image_path": img_path,
            "y_true": gt,
            "y_pred": pred,
            "raw_output": gen,
        })

        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"[INFO] Progress {i+1}/{n} | bad_parses={bad}", flush=True)

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float((y_true_arr == y_pred_arr).mean())
    if SKLEARN_OK and cohen_kappa_score is not None:
        qwk = float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic"))
    else:
        qwk = qwk_numpy(y_true_arr, y_pred_arr, n_classes=5)

    print("\n=== RESULTS ===")
    print(f"Run path:   {args.run_path}")
    print(f"LoRA dir:   {lora_dir}")
    print(f"CSV used:   {data_csv}")
    print(f"Samples:    {n}")
    print(f"Bad parses: {bad}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"QWK:        {qwk:.4f}")

    if confusion_matrix is not None:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3, 4])
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)

    if classification_report is not None:
        print("\nClassification report:")
        print(classification_report(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3, 4], digits=4))

    if args.save_preds_csv:
        out_csv = Path(args.save_preds_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(out_csv, index=False)
        print(f"\n[INFO] Saved predictions CSV to: {out_csv}")

        summary = {
            "run_path": args.run_path,
            "resolved_lora_dir": str(lora_dir),
            "csv_used": data_csv,
            "image_dir": args.image_dir,
            "samples": int(n),
            "bad_parses": int(bad),
            "accuracy": float(acc),
            "qwk": float(qwk),
        }
        summary_path = out_csv.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()