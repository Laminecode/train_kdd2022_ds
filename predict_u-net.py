"""
predict_damage.py - Road Damage Prediction Script
Predicts damage types from raw images using trained U-Net model
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("❌ Error: segmentation_models_pytorch not installed!")
    print("Install it with: pip install segmentation-models-pytorch")
    sys.exit(1)

# ============ CONFIG - EDIT THESE PATHS ============
MODEL_PATH = "outputs/u-net/best_unet_model (1).pth"  # Path to your trained model
IMAGE_PATH = "dataset/raw/test/images/China_Drone_000186.jpg"         # Path to image to predict
OUTPUT_DIR = "predictions"            # Where to save results
DEVICE = "cuda"                       # "cuda" or "cpu"

# Model configuration
IMG_SIZE = 640
NUM_CLASSES = 6
ENCODER = 'resnet34'

CLASS_NAMES = [
    'Background',
    'Longitudinal_crack',
    'Transverse_crack',
    'Alligator_crack',
    'Pothole',           # Class 4 (matches your OLD trained model)
    'Other_damage'       # Class 5 (matches your OLD trained model)
]

# Color map for visualization (BGR format for OpenCV)
COLORS = np.array([
    [0, 0, 0],       # Background - Black
    [0, 0, 255],     # Longitudinal crack - Red
    [0, 255, 0],     # Transverse crack - Green
    [255, 0, 0],     # Alligator crack - Blue
    [0, 255, 255],   # Pothole - Yellow (matches OLD model)
    [255, 0, 255]    # Other damage - Magenta (matches OLD model)
], dtype=np.uint8)


def load_model(model_path, device):
    """Load the trained U-Net model."""
    print(f"📦 Loading model from: {model_path}")
    
    # Create model architecture
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,  # We'll load our trained weights
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # PyTorch 2.6+ security: set weights_only=False for trusted checkpoints
    # This is safe for models you trained yourself
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'iou' in checkpoint:
            print(f"   Model trained mIoU: {checkpoint['iou']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
    
    return model


def preprocess_image(image_path):
    """
    Load and preprocess image for model input.
    This adapts the raw image to the format expected by the trained model.
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Convert BGR to RGB (model expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size for later
    original_shape = image.shape[:2]
    
    # Apply same preprocessing as training (without augmentation)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Transform image
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_shape


def predict_mask(model, image_tensor, device):
    """Run inference and get predicted mask."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    return pred_mask


def analyze_prediction(pred_mask):
    """Analyze predicted mask and extract damage statistics."""
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    
    stats = {}
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        stats[class_id] = {
            'name': CLASS_NAMES[class_id],
            'pixels': int(count),
            'percentage': percentage
        }
    
    return stats


def create_visualization(original_image_path, pred_mask, output_path, stats):
    """Create visualization with original image, mask overlay, and statistics."""
    # Read original image
    original = cv2.imread(str(original_image_path))
    
    # Resize to match prediction
    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    
    # Create colored mask
    colored_mask = COLORS[pred_mask]
    
    # Create overlay (blend original with mask)
    alpha = 0.5
    overlay = cv2.addWeighted(original_resized, 1-alpha, colored_mask, alpha, 0)
    
    # Create result image with three panels + stats panel below
    panel_width = IMG_SIZE
    panel_height = IMG_SIZE
    stats_height = 180  # Height for statistics panel
    
    # Main visualization (3 panels side by side)
    result = np.zeros((panel_height + stats_height, panel_width * 3, 3), dtype=np.uint8)
    
    # Top row: three image panels
    result[:panel_height, :panel_width] = original_resized
    result[:panel_height, panel_width:panel_width*2] = colored_mask
    result[:panel_height, panel_width*2:] = overlay
    
    # Add labels to image panels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, "Prediction", (panel_width + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(result, "Overlay", (panel_width*2 + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Bottom panel: statistics with black background
    stats_start_y = panel_height
    result[stats_start_y:, :] = (30, 30, 30)  # Dark gray background
    
    # Add statistics title
    title_y = stats_start_y + 35
    cv2.putText(result, "Damage Analysis:", (20, title_y), font, 0.8, (255, 255, 255), 2)
    
    # Add class legend and percentages
    y = title_y + 40
    x_col1 = 20
    x_col2 = panel_width + 20
    x_col3 = panel_width*2 + 20
    col = 0
    
    for class_id in sorted(stats.keys()):
        if class_id == 0:  # Skip background
            continue
        stat = stats[class_id]
        if stat['percentage'] > 0.1:  # Only show if >0.1%
            # Determine column position
            if col == 0:
                x_pos = x_col1
            elif col == 1:
                x_pos = x_col2
            else:
                x_pos = x_col3
            
            # Draw colored square indicator
            color = tuple(int(c) for c in COLORS[class_id])
            cv2.rectangle(result, (x_pos, y-15), (x_pos+20, y-5), color, -1)
            cv2.rectangle(result, (x_pos, y-15), (x_pos+20, y-5), (255, 255, 255), 1)
            
            # Draw text
            text = f"{stat['name']}: {stat['percentage']:.2f}%"
            cv2.putText(result, text, (x_pos + 30, y), font, 0.5, (255, 255, 255), 1)
            
            # Move to next position
            col += 1
            if col >= 3:
                col = 0
                y += 30
    
    # Save result
    cv2.imwrite(str(output_path), result)
    print(f"💾 Visualization saved: {output_path}")
    
    return result


def predict_single_image(image_path, model_path, output_dir, device='cuda'):
    """Predict damage for a single image."""
    print(f"\n{'='*60}")
    print(f"🔍 Processing: {image_path}")
    print(f"{'='*60}")
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    
    # Load model
    model = load_model(model_path, device)
    
    # Preprocess image
    print("📐 Preprocessing image...")
    image_tensor, original_shape = preprocess_image(image_path)
    
    # Predict
    print("🧠 Running inference...")
    pred_mask = predict_mask(model, image_tensor, device)
    
    # Analyze
    print("📊 Analyzing results...")
    stats = analyze_prediction(pred_mask)
    
    # Print statistics
    print("\n" + "="*60)
    print("📋 DAMAGE DETECTION RESULTS:")
    print("="*60)
    
    damage_detected = False
    for class_id in sorted(stats.keys()):
        if class_id == 0:  # Background
            continue
        stat = stats[class_id]
        if stat['percentage'] > 0.1:
            damage_detected = True
            print(f"  • {stat['name']:20s}: {stat['percentage']:6.2f}% ({stat['pixels']:,} pixels)")
    
    if not damage_detected:
        print("  ✅ No significant damage detected")
    
    print("="*60)
    
    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_prediction.png")
    
    print("\n🎨 Creating visualization...")
    create_visualization(image_path, pred_mask, output_path, stats)
    
    # Save raw mask
    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    cv2.imwrite(mask_path, pred_mask)
    print(f"💾 Raw mask saved: {mask_path}")
    
    print(f"\n✅ Processing complete!")
    
    return pred_mask, stats


def predict_batch(input_dir, model_path, output_dir, device='cuda'):
    """Predict damage for all images in a directory."""
    print(f"\n{'='*60}")
    print(f"📁 Batch Processing: {input_dir}")
    print(f"{'='*60}")
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f"*{ext}"))
        image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📸 Found {len(image_paths)} images")
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    
    # Load model once
    model = load_model(model_path, device)
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {image_path.name}")
        
        try:
            # Preprocess
            image_tensor, _ = preprocess_image(image_path)
            
            # Predict
            pred_mask = predict_mask(model, image_tensor, device)
            
            # Analyze
            stats = analyze_prediction(pred_mask)
            
            # Visualize
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{image_path.stem}_prediction.png")
            create_visualization(image_path, pred_mask, output_path, stats)
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{image_path.stem}_mask.png")
            cv2.imwrite(mask_path, pred_mask)
            
            results.append({
                'image': image_path.name,
                'stats': stats,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            results.append({
                'image': image_path.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    """Main function - runs prediction with paths defined at top of file."""
    print(f"\n{'='*60}")
    print(f"🚀 ROAD DAMAGE PREDICTION")
    print(f"{'='*60}")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📸 Image: {IMAGE_PATH}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Validate paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("   Please check MODEL_PATH at the top of the script")
        sys.exit(1)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file not found: {IMAGE_PATH}")
        print("   Please check IMAGE_PATH at the top of the script")
        sys.exit(1)
    
    # Run prediction
    try:
        predict_single_image(IMAGE_PATH, MODEL_PATH, OUTPUT_DIR, DEVICE)
        print(f"\n✨ All done! Check '{OUTPUT_DIR}' for results")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()