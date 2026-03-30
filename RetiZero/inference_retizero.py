import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Ensure the RetiZero repository is accessible for iden_modules
# Adjust this path if your RetiZero folder is located elsewhere
RETI_ZERO_PATH = './RetiZero'
if RETI_ZERO_PATH not in sys.path:
    sys.path.append(RETI_ZERO_PATH)

try:
    from iden_modules import CLIPRModel
except ImportError:
    print(f"Error: Could not import CLIPRModel. Ensure the RetiZero repo is cloned at {RETI_ZERO_PATH}")
    sys.exit(1)

# --- 1. Human-Readable Grades ---
DR_GRADES = {
    0: "Normal (No DR)",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

# 2. Optimized Inference Architecture
class RetiZeroInference(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Initialize base model without auto-loading weights (saves memory in prod)
        self.retizero = CLIPRModel(vision_type="lora", from_checkpoint=False, R=8)
        self.img_encoder = self.retizero.vision_model.model
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.img_encoder(x)
        logits = self.classifier(features)
        return logits

# 4. Main Inference Function
def predict(image_path, weights_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # Initialize model and load fine-tuned weights
    print("Loading model architecture and weights...")
    model = RetiZeroInference(num_classes=5).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

# Preprocessing Pipeline (Must match training exactly - NO CLAHE)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Preprocessing image...")
    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = val_transforms(raw_image).unsqueeze(0).to(device)

    # Inference
    print("Running inference...\n")
    with torch.no_grad():
        with torch.amp.autocast(device.type):
            logits = model(input_tensor)
            # Add .float() before converting to numpy!
            probabilities = F.softmax(logits, dim=1).squeeze().float().cpu().numpy()

    prediction_idx = probabilities.argmax()
    predicted_grade = DR_GRADES[prediction_idx]
    confidence = probabilities[prediction_idx] * 100

    # Output Results
    print("DIAGNOSTIC RESULTS")
    print(f"Final Prediction : {predicted_grade}")
    print(f"Confidence       : {confidence:.2f}%\n")
    
    print("Probability Breakdown")
    for idx, prob in enumerate(probabilities):
        print(f"{DR_GRADES[idx]:<20}: {prob*100:.2f}%")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetiZero Diabetic Retinopathy Inference")
    parser.add_argument("image_path", type=str, help="Path to the retinal image file")
    parser.add_argument("--weights", type=str, default="best_retizero_dr.pth", help="Path to the finetuned state dict")
    
    args = parser.parse_args()
    predict(args.image_path, args.weights)