import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_qwen_array(img_bgr: np.ndarray, *, clahe: bool = True) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty or invalid.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mask.sum() == 0:
        cropped = img_rgb
    else:
        coords = np.argwhere(mask > 0)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        cropped = img_rgb[x0:x1, y0:y1]

    h, w = cropped.shape[:2]
    diff = abs(h - w)

    if h < w:
        padded = cv2.copyMakeBorder(
            cropped,
            diff // 2,
            diff - (diff // 2),
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    else:
        padded = cv2.copyMakeBorder(
            cropped,
            0,
            0,
            diff // 2,
            diff - (diff // 2),
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    target_res = 512
    interpolation = cv2.INTER_AREA if target_res < padded.shape[0] else cv2.INTER_CUBIC
    resized_rgb = cv2.resize(padded, (target_res, target_res), interpolation=interpolation)

    final_rgb = apply_clahe(resized_rgb) if clahe else resized_rgb
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    return final_bgr


def preprocess_for_qwen(input_path: str, output_path: str, *, clahe: bool = True) -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read the image at: {input_path}")

    out = preprocess_qwen_array(img, clahe=clahe)
    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write preprocessed image to: {output_path}")


def preprocess_batch(manifest: Path, *, clahe: bool, workers: int) -> None:
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [(Path(row["input_path"]).resolve(), Path(row["output_path"]).resolve())
                for row in csv.DictReader(stream)]
    sources, targets = {src for src, _ in rows}, {dst for _, dst in rows}
    if workers < 1 or not rows or len(targets) != len(rows) or sources & targets:
        raise ValueError("Use a nonempty manifest with unique outputs separate from the inputs and at least one worker.")
    if any(not src.is_file() for src in sources):
        raise FileNotFoundError("The batch manifest contains a missing input image.")
    if any(dst.exists() or dst.suffix.lower() != ".png" for dst in targets):
        raise ValueError("Batch outputs must be new PNG paths.")
    for parent in {dst.parent for dst in targets}:
        parent.mkdir(parents=True, exist_ok=True)

    def process(row):
        src, dst = row
        preprocess_for_qwen(str(src), str(dst), clahe=clahe)

    def progress(results):
        for count, _ in enumerate(results, 1):
            if count == 1 or count % 50 == 0 or count == len(rows):
                print(f"Processed {count}/{len(rows)} images", flush=True)

    if workers == 1:
        progress(map(process, rows))
    else:
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            progress(pool.map(process, rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image for Qwen")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--manifest", type=Path, help="CSV with absolute input_path and output_path columns")
    parser.add_argument("--clahe", choices=("on", "off"), default="on")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.manifest:
        if args.input_path or args.output_path or args.workers < 1:
            parser.error("Use --manifest without positional paths and with at least one worker.")
        preprocess_batch(args.manifest, clahe=args.clahe == "on", workers=args.workers)
    else:
        if not args.input_path or not args.output_path or args.workers != 1:
            parser.error("Supply input and output paths. Multiple workers require --manifest.")
        preprocess_for_qwen(args.input_path, args.output_path, clahe=args.clahe == "on")
        print(f"Qwen preprocessing complete: {args.output_path}")
