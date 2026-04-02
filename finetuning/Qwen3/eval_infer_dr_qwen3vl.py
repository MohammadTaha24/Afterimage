# FILE: eval_infer_dr_qwen3vl.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from unsloth import FastVisionModel
from peft import PeftModel

try:
    from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


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

ANSWER_RE = re.compile(r"<answer>\s*([0-4])\s*-\s*.*?</answer>", re.IGNORECASE)


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

    candidates = []

    if fp is not None and str(fp).strip() != "":
        fp_s = str(fp).strip().replace("\\", "/")
        fp_p = Path(fp_s)

        if fp_p.is_absolute():
            candidates.append(fp_p)

        candidates.append(image_dir / fp_s)
        candidates.append(image_dir / fp_p.name)
        candidates.append(image_dir.parent / fp_s)
        candidates.append(image_dir.parent / fp_p.name)

    if id_code:
        candidates.append(image_dir / f"{id_code}{ext}")

    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except Exception:
            continue

    return str(candidates[0]) if candidates else ""


def load_test_csv(csv_path: str, image_dir: str, ext: str) -> Dataset:
    csv_path = str(csv_path)
    image_dir_p = Path(image_dir)

    df = pd.read_csv(csv_path, dtype=str)

    # ---- schema mapping (supports multiple CSV formats) ----
    # id field:
    if "id_code" in df.columns:
        df["id_code"] = df["id_code"].astype(str)
    elif "filename" in df.columns:
        df["id_code"] = df["filename"].astype(str).map(lambda s: Path(str(s)).stem)
    else:
        raise ValueError(
            f"CSV missing id column. Expected 'id_code' or 'filename'. Found: {list(df.columns)}"
        )

    # label field:
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].astype(int)
    elif "dr_grade" in df.columns:
        df["diagnosis"] = df["dr_grade"].astype(int)
    elif "grade" in df.columns:
        df["diagnosis"] = df["grade"].astype(int)
    else:
        raise ValueError(
            f"CSV missing label column. Expected 'diagnosis' or 'dr_grade' (or 'grade'). Found: {list(df.columns)}"
        )

    # ---- resolve image paths ----
    # Prefer file_path if present, else filename if present, else id_code+ext
    def _resolve(row):
        fp = row.get("file_path", None)
        if fp is None or str(fp).strip() == "":
            fp = row.get("filename", None)

        candidates = []
        if fp is not None and str(fp).strip() != "":
            fp_s = str(fp).strip().replace("\\", "/")
            fp_p = Path(fp_s)

            if fp_p.is_absolute():
                candidates.append(fp_p)

            candidates.append(image_dir_p / fp_s)
            candidates.append(image_dir_p / fp_p.name)
            candidates.append(image_dir_p.parent / fp_s)
            candidates.append(image_dir_p.parent / fp_p.name)

        id_code = str(row.get("id_code", "")).strip()
        if id_code:
            candidates.append(image_dir_p / f"{id_code}{ext}")

        for c in candidates:
            try:
                if c.exists():
                    return str(c)
            except Exception:
                continue

        return str(candidates[0]) if candidates else ""

    df["image_path"] = df.apply(_resolve, axis=1)

    exists_mask = df["image_path"].apply(os.path.exists)
    missing = int((~exists_mask).sum())
    if missing:
        print(f"[WARN] Missing images in {Path(csv_path).name}: {missing}. Dropping them.", flush=True)
    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            "0 rows remain after filtering missing images. "
            "Check --image_dir and whether filename/file_path are relative to it."
        )

    return Dataset.from_pandas(df)


@torch.no_grad()
def predict_one(model, tokenizer, img_path: str, max_new_tokens: int, temperature: float, min_p: float, img_size: str) -> str:
    w, h = _parse_img_size(img_size)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": INSTRUCTION},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    tokenizer_kwargs = dict(add_special_tokens=False, return_tensors="pt")

    tried = False
    for extra in (
        {"image_size": (h, w)},
        {"size": {"height": h, "width": w}},
        {"image_sizes": [(h, w)]},
    ):
        try:
            inputs = tokenizer(img_path, input_text, **tokenizer_kwargs, **extra).to("cuda")
            tried = True
            break
        except TypeError:
            continue

    if not tried:
        inputs = tokenizer(img_path, input_text, **tokenizer_kwargs).to("cuda")

    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"{k}: device={v.device}, dtype={v.dtype}, shape={tuple(v.shape)}", flush=True)


    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        min_p=min_p,
        use_cache=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def parse_pred_label(gen_text: str) -> Optional[int]:
    m = ANSWER_RE.search(gen_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def qwk_numpy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            O[t, p] += 1.0

    true_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    N = O.sum()
    if N == 0:
        return 0.0
    E = np.outer(true_hist, pred_hist) / N

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = (n_classes - 1) ** 2
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - (num / den))


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--base_model", type=str, default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit")
    p.add_argument("--lora_dir", type=str, default="qwen3vl8b_dr_lora")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--no_load_in_4bit", action="store_true")

    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--test_csv", type=str, default="")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--ext", type=str, default=".png")
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--img_size", type=str, default="384")
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--min_p", type=float, default=0.1)

    args = p.parse_args()

    load_in_4bit = True
    if args.no_load_in_4bit:
        load_in_4bit = False
    if args.load_in_4bit:
        load_in_4bit = True

    test_csv = args.test_csv if args.test_csv else args.csv_path

    print("[INFO] Loading base model...", flush=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=load_in_4bit,
    )
    print("[INFO] Attaching LoRA adapters...", flush=True)
    model = PeftModel.from_pretrained(model, args.lora_dir)
    FastVisionModel.for_inference(model)

    print("[INFO] Attaching LoRA adapters...", flush=True)
    model = PeftModel.from_pretrained(model, args.lora_dir)
    FastVisionModel.for_inference(model)
    model.eval()

    print("Device:", next(model.parameters()).device, flush=True)
    print("Model dtype sample:", next(model.parameters()).dtype, flush=True)

    print(f"[INFO] Loading TEST CSV: {test_csv}", flush=True)
    eval_raw = load_test_csv(
        csv_path=test_csv,
        image_dir=args.image_dir,
        ext=args.ext,
    )

    n = len(eval_raw)
    if args.limit and args.limit > 0:
        n = min(n, args.limit)

    print(f"[INFO] Evaluating {n} samples | img_size={args.img_size}", flush=True)

    y_true: List[int] = []
    y_pred: List[int] = []
    bad = 0

    for i in range(n):
        row = eval_raw[i]
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

        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"[INFO] Progress {i+1}/{n} | bad_parses={bad}", flush=True)

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float((y_true_arr == y_pred_arr).mean())
    qwk = float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic")) if SKLEARN_OK else qwk_numpy(y_true_arr, y_pred_arr, 5)

    if SKLEARN_OK:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3, 4])
        report = classification_report(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3, 4])
    else:
        cm = np.zeros((5, 5), dtype=int)
        for t, p_ in zip(y_true_arr, y_pred_arr):
            cm[t, p_] += 1
        report = None

    print("\n[RESULTS]", flush=True)
    print("Accuracy:", acc, flush=True)
    print("QWK:", qwk, flush=True)
    print("Bad parses:", bad, flush=True)
    print("Confusion matrix (rows=true, cols=pred):\n", cm, flush=True)
    if report is not None:
        print("\nClassification report:\n", report, flush=True)


if __name__ == "__main__":
    main()