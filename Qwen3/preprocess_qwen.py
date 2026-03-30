import argparse

import cv2
import numpy as np


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_qwen_array(img_bgr: np.ndarray) -> np.ndarray:
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

    final_rgb = apply_clahe(resized_rgb)
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    return final_bgr


def preprocess_for_qwen(input_path: str, output_path: str) -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read the image at: {input_path}")

    out = preprocess_qwen_array(img)
    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write preprocessed image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image for Qwen")
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    preprocess_for_qwen(args.input_path, args.output_path)
    print(f"Qwen preprocessing complete: {args.output_path}")