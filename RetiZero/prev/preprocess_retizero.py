import argparse
import os
import sys
import cv2
import numpy as np

def preprocess_for_retizero(input_path, output_path):
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

    # 6. Resize to 224x224
    TARGET_RES = 224
    h_current = padded.shape[0]
    interpolation = cv2.INTER_AREA if TARGET_RES < h_current else cv2.INTER_CUBIC
    final_img_rgb = cv2.resize(padded, (TARGET_RES, TARGET_RES), interpolation=interpolation)

    # 7. Convert back to BGR for saving with OpenCV
    final_img_bgr = cv2.cvtColor(final_img_rgb, cv2.COLOR_RGB2BGR)
    
    # 8. Save the output
    cv2.imwrite(output_path, final_img_bgr)
    print(f"RetiZero Preprocessing Complete: Saved to {output_path} (224x224, No CLAHE)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess image for RetiZero (224x224, No CLAHE)")
    parser.add_argument("input_path", type=str, help="Path to the raw input image")
    parser.add_argument("output_path", type=str, help="Path to save the preprocessed image")
    
    args = parser.parse_args()
    preprocess_for_retizero(args.input_path, args.output_path)