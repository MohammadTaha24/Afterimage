import argparse
import os
import sys
import cv2
import numpy as np

def apply_clahe(img_rgb):
    """Applies CLAHE to the L channel of LAB color space."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def preprocess_for_qwen(input_path, output_path):
    # 1. Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: OpenCV could not read the image at {input_path}")
        sys.exit(1)

    # 2. Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Grayscale and Otsu's Thresholding for ROI Mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. ROI Crop
    if mask.sum() == 0:
        cropped = img_rgb
    else:
        coords = np.argwhere(mask > 0)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        cropped = img_rgb[x0:x1, y0:y1]

    # 5. Square Pad
    h, w = cropped.shape[:2]
    diff = abs(h - w)
    if h < w:
        padded = cv2.copyMakeBorder(cropped, diff//2, diff-(diff//2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        padded = cv2.copyMakeBorder(cropped, 0, 0, diff//2, diff-(diff//2), cv2.BORDER_CONSTANT, value=[0,0,0])

    # 6. Resize to 512x512
    TARGET_RES = 512
    h_current = padded.shape[0]
    interpolation = cv2.INTER_AREA if TARGET_RES < h_current else cv2.INTER_CUBIC
    resized_rgb = cv2.resize(padded, (TARGET_RES, TARGET_RES), interpolation=interpolation)

    # 7. Apply CLAHE (After resizing for consistent tile grid behavior)
    final_img_rgb = apply_clahe(resized_rgb)

    # 8. Convert back to BGR for saving with OpenCV
    final_img_bgr = cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR)
    
    # 9. Save the output
    cv2.imwrite(output_path, final_img_bgr)
    print(f"Qwen Preprocessing Complete: Saved to {output_path} (512x512, With CLAHE)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image for Qwen (512x512, With CLAHE)")
    parser.add_argument("input_path", type=str, help="Path to the raw input image")
    parser.add_argument("output_path", type=str, help="Path to save the preprocessed image")
    
    args = parser.parse_args()
    preprocess_for_qwen(args.input_path, args.output_path)