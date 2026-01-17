# preprocess_images.py - Optimized and Fixed
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ============ CONFIG ============
# Use absolute paths to avoid issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET = os.path.join(BASE_DIR, "dataset", "reduced")
PROCESSED = os.path.join(BASE_DIR, "dataset", "processed22")
SEGMENTATION = os.path.join(BASE_DIR, "dataset", "segmentation22")

IMG_SIZE = (640, 640)
SPLITS = ["train", "val", "test"]

# Multi-class segmentation
BINARY_MASK = False  # False = multi-class, True = binary

# YOLO has 5 damage classes (IDs: 0-4)
# Mask has 6 values: 0=background, 1-5=damage types (YOLO ID + 1)
NUM_CLASSES = 6  # Total: 1 background + 5 damage types

# Mapping: YOLO class ID -> Damage type
YOLO_CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "other damage",
    4: "Pothole"
}

# Mapping: Mask value -> Class name
MASK_CLASS_NAMES = {
    0: "Background",
    1: "Longitudinal crack",    # YOLO 0
    2: "Transverse crack",       # YOLO 1
    3: "Alligator crack",        # YOLO 2
    4: "other damage",                # YOLO 3
    5: "Pothole"            # YOLO 4
}

print(f"\n{'='*50}")
print(f"PREPROCESSING CONFIGURATION")
print(f"{'='*50}")
print(f"Image Size: {IMG_SIZE}")
print(f"Mode: {'Binary' if BINARY_MASK else 'Multi-class'}")
print(f"Num Classes: {NUM_CLASSES} (1 background + 5 damage types)")
print(f"\nYOLO Classes (0-4):")
for k, v in YOLO_CLASS_NAMES.items():
    print(f"  {k}: {v}")
print(f"\nMask Values (0-5):")
for k, v in MASK_CLASS_NAMES.items():
    print(f"  {k}: {v}")
print(f"\nPaths:")
print(f"  Input:  {RAW_DATASET}")
print(f"  Output: {PROCESSED}")
print(f"  Masks:  {SEGMENTATION}")
print(f"{'='*50}\n")


def load_yolo_labels(label_path):
    """Load YOLO format labels from file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x, y, w, h = map(float, parts)
                # Validate class_id is in valid range (0-4 for YOLO)
                if 0 <= int(class_id) <= 4:
                    boxes.append((int(class_id), x, y, w, h))
    return boxes


def yolo_to_mask(boxes, img_shape, binary=False):
    """
    Generate segmentation mask from YOLO bounding boxes.
    
    Args:
        boxes: List of (class_id, x, y, w, h) in YOLO normalized format
               class_id ranges from 0-4 (5 YOLO damage classes)
        img_shape: Tuple (H, W) image dimensions
        binary: If True, binary mask (0/1). If False, multi-class (0-5).
    
    Returns:
        Mask numpy array:
        - Binary mode: values 0 (background) or 1 (any damage)
        - Multi-class mode: values 0-5 (0=background, 1-5=damage classes)
    
    Priority handling for overlaps:
        - Higher class_id (rarer damage) overwrites lower class_id
        - Ensures important damage types are not hidden
    """
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # Sort by class_id ASCENDING so higher priority classes overwrite
    boxes_sorted = sorted(boxes, key=lambda b: b[0])

    for class_id, x, y, bw, bh in boxes_sorted:
        # Convert YOLO normalized coords to pixel coords
        x1 = int((x - bw / 2) * W)
        y1 = int((y - bh / 2) * H)
        x2 = int((x + bw / 2) * W)
        y2 = int((y + bh / 2) * H)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 > x1 and y2 > y1:
            if binary:
                # Binary mask: all damage = 1
                mask[y1:y2, x1:x2] = 1
            else:
                # Multi-class: YOLO class_id + 1
                # YOLO 0 → mask 1, YOLO 1 → mask 2, ..., YOLO 4 → mask 5
                mask_value = class_id + 1
                mask[y1:y2, x1:x2] = mask_value

    return mask


def verify_mask(mask, img_name):
    """Verify mask has valid values and log statistics."""
    unique_values = np.unique(mask)
    
    # Check for invalid values
    max_valid = 1 if BINARY_MASK else 5
    invalid = unique_values[unique_values > max_valid]
    
    if len(invalid) > 0:
        return f"[WARN] {img_name}: Invalid mask values {invalid}"
    
    return None


def process_image(args):
    """Process a single image (for parallel processing)."""
    img_name, raw_img_dir, raw_lbl_dir, proc_img_dir, proc_lbl_dir, seg_img_dir, seg_msk_dir = args
    
    try:
        # Load image
        img_path = os.path.join(raw_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            return f"[SKIP] Cannot read: {img_name}"

        # Resize
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

        # Save processed image
        cv2.imwrite(os.path.join(proc_img_dir, img_name), img)
        cv2.imwrite(os.path.join(seg_img_dir, img_name), img)

        # Process labels
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + ".txt"
        label_path = os.path.join(raw_lbl_dir, label_name)

        # Copy label to processed
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(proc_lbl_dir, label_name))
        else:
            # Create empty label file for background-only images
            open(os.path.join(proc_lbl_dir, label_name), "w").close()

        # Generate segmentation mask
        boxes = load_yolo_labels(label_path)
        mask = yolo_to_mask(boxes, IMG_SIZE[::-1], binary=BINARY_MASK)  # (H, W)
        
        # Verify mask
        warning = verify_mask(mask, img_name)
        if warning:
            return warning
        
        # Save mask
        mask_name = base_name + ".png"
        cv2.imwrite(os.path.join(seg_msk_dir, mask_name), mask)
        
        return None  # Success
        
    except Exception as e:
        return f"[ERROR] {img_name}: {str(e)}"


def run(use_parallel=True, max_workers=4):
    """Run preprocessing with optional parallel processing."""
    
    # Verify input directory exists
    if not os.path.exists(RAW_DATASET):
        print(f"❌ ERROR: Input directory not found!")
        print(f"   Expected: {RAW_DATASET}")
        print(f"\n💡 Please check your paths and run reduce.py first.")
        return
    
    total_processed = 0
    total_errors = 0
    total_warnings = 0
    
    for split in SPLITS:
        print(f"\n{'='*50}")
        print(f"🔄 Processing: {split.upper()}")
        print(f"{'='*50}")

        raw_img_dir = os.path.join(RAW_DATASET, split, "images")
        raw_lbl_dir = os.path.join(RAW_DATASET, split, "labels")

        proc_img_dir = os.path.join(PROCESSED, split, "images")
        proc_lbl_dir = os.path.join(PROCESSED, split, "labels")

        seg_img_dir = os.path.join(SEGMENTATION, split, "images")
        seg_msk_dir = os.path.join(SEGMENTATION, split, "masks")

        # Create directories
        for d in [proc_img_dir, proc_lbl_dir, seg_img_dir, seg_msk_dir]:
            os.makedirs(d, exist_ok=True)

        # Check if source exists
        if not os.path.exists(raw_img_dir):
            print(f"  ⚠️ Source not found: {raw_img_dir}")
            continue

        # Get image list
        images = [f for f in os.listdir(raw_img_dir) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            print(f"  ⚠️ No images found in {raw_img_dir}")
            continue

        # Prepare arguments
        args_list = [
            (img, raw_img_dir, raw_lbl_dir, proc_img_dir, proc_lbl_dir, seg_img_dir, seg_msk_dir)
            for img in images
        ]

        errors = []
        warnings = []
        
        if use_parallel:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(
                    executor.map(process_image, args_list),
                    total=len(images),
                    desc=f"  {split}"
                ))
            errors = [r for r in results if r and r.startswith("[ERROR]")]
            warnings = [r for r in results if r and r.startswith("[WARN]")]
        else:
            # Sequential processing
            for args in tqdm(args_list, desc=f"  {split}"):
                result = process_image(args)
                if result:
                    if result.startswith("[ERROR]"):
                        errors.append(result)
                    elif result.startswith("[WARN]"):
                        warnings.append(result)
        
        # Report
        processed = len(images) - len(errors)
        total_processed += processed
        total_errors += len(errors)
        total_warnings += len(warnings)
        
        print(f"  ✅ {split}: {processed}/{len(images)} images processed")
        
        if warnings:
            print(f"  ⚠️ {len(warnings)} warnings")
        
        if errors:
            print(f"  ❌ {len(errors)} errors:")
            for e in errors[:5]:  # Show first 5 errors
                print(f"     {e}")

    print(f"\n{'='*50}")
    print(f"🎉 PREPROCESSING COMPLETE!")
    print(f"{'='*50}")
    print(f"  ✅ Processed: {total_processed} images")
    print(f"  ⚠️  Warnings: {total_warnings}")
    print(f"  ❌ Errors: {total_errors}")
    print(f"\n  📁 Output:")
    print(f"     Detection: {PROCESSED}")
    print(f"     Segmentation: {SEGMENTATION}")
    print(f"\n  🎭 Mask type: {'Binary (0/1)' if BINARY_MASK else f'Multi-class (0-{NUM_CLASSES-1})'}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Run preprocessing
    run(use_parallel=True, max_workers=4)