"""
Reduce dataset size with CLASS-BALANCED sampling.
Ensures each class has a target number of images (or all available if fewer).
"""
import os
import shutil
import random
from collections import defaultdict, Counter

# ============ CONFIG ============
SEED = 42

# Target samples PER CLASS (not total images!)
# Set to None for a class to include ALL available images for that class
# Set to a number to limit that class
SAMPLES_PER_CLASS = {
    0: 700,   # Longitudinal crack - reduce from 2345
    1: 700,   # Transverse crack - reduce from 1065
    2: 700,   # Alligator crack - reduce from 998
    3: 700,   # Pothole - reduce from 1061
    4: 700,   # Other damage - keep all 560 (will use all available)
}

# If True: also include background-only images (empty labels)
# IMPORTANT: You need some background-only images so the model learns what "no damage" looks like
INCLUDE_BACKGROUND_ONLY = True
MAX_BACKGROUND_ONLY = 280  # Max background-only images to include (adjust as needed)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset/raw")
REDUCED_DIR = os.path.join(PROJECT_ROOT, "dataset/reduced")


def analyze_split(src_images, src_labels):
    """Analyze images and build class mappings."""
    images = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Map: class_id -> list of images containing that class
    class_to_images = defaultdict(list)
    # Map: image -> set of classes
    image_to_classes = {}
    # List of background-only images (empty labels)
    background_only = []
    
    for img_name in images:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(src_labels, label_name)
        
        classes = set()
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls = int(float(parts[0]))
                                classes.add(cls)
                            except:
                                pass
            except:
                pass
        
        image_to_classes[img_name] = classes
        
        if not classes:
            background_only.append(img_name)
        else:
            for cls in classes:
                class_to_images[cls].append(img_name)
    
    return class_to_images, image_to_classes, background_only


def balanced_sample(class_to_images, image_to_classes, background_only, 
                    samples_per_class, include_bg=False, max_bg=200):
    """
    Select images ensuring each class has up to `samples_per_class[cls]` samples.
    An image can count toward multiple classes if it contains multiple damage types.
    """
    selected = set()
    class_counts = Counter()
    
    # Sort classes by rarity (rarest first) to prioritize them
    all_classes = sorted(class_to_images.keys(), key=lambda c: len(class_to_images[c]))
    
    print(f"\n  Available per class: {dict((c, len(class_to_images[c])) for c in sorted(class_to_images.keys()))}")
    
    # For each class, select images until we hit the target
    for cls in all_classes:
        target = samples_per_class.get(cls, None)
        if target is None:
            target = len(class_to_images[cls])  # Use all
        
        # Get images for this class, prioritize ones not yet selected
        available = class_to_images[cls]
        random.shuffle(available)
        
        # Sort: images not yet selected come first
        available_sorted = sorted(available, key=lambda img: img in selected)
        
        for img in available_sorted:
            if class_counts[cls] >= target:
                break
            if img not in selected:
                selected.add(img)
            # Count this image for all its classes
            for c in image_to_classes[img]:
                class_counts[c] += 1
    
    # Optionally add background-only images
    if include_bg and background_only:
        bg_sample = random.sample(background_only, min(max_bg, len(background_only)))
        selected.update(bg_sample)
        print(f"  Added {len(bg_sample)} background-only images")
    
    print(f"  Selected per class: {dict(sorted(class_counts.items()))}")
    
    return list(selected)


def reduce_split(split_name, samples_per_class, include_bg=False, max_bg=200, scale=1.0):
    """Reduce a single split with class-balanced sampling."""
    src_images = os.path.join(RAW_DIR, split_name, "images")
    src_labels = os.path.join(RAW_DIR, split_name, "labels")
    dst_images = os.path.join(REDUCED_DIR, split_name, "images")
    dst_labels = os.path.join(REDUCED_DIR, split_name, "labels")
    
    if not os.path.exists(src_images):
        print(f"  ⚠️ {split_name}: source not found, skipping")
        return
    
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    
    # Analyze
    class_to_images, image_to_classes, background_only = analyze_split(src_images, src_labels)
    
    # Scale samples_per_class for val/test (e.g., 0.2 for 20%)
    scaled_spc = {
        c: int(n * scale) if n is not None else None 
        for c, n in samples_per_class.items()
    }
    
    # Sample
    selected = balanced_sample(
        class_to_images, image_to_classes, background_only,
        scaled_spc, include_bg, int(max_bg * scale)
    )
    
    print(f"  {split_name}: {len(selected)} images total")
    
    # Copy files
    for img_name in selected:
        shutil.copy(os.path.join(src_images, img_name), os.path.join(dst_images, img_name))
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_src = os.path.join(src_labels, label_name)
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(dst_labels, label_name))


def main():
    random.seed(SEED)
    
    print("=" * 50)
    print("CLASS-BALANCED DATASET REDUCTION")
    print("=" * 50)
    print(f"\nTarget samples per class: {SAMPLES_PER_CLASS}")
    print(f"Include background-only: {INCLUDE_BACKGROUND_ONLY} (max {MAX_BACKGROUND_ONLY})")
    print(f"Output: {REDUCED_DIR}\n")
    
    # Clear existing
    if os.path.exists(REDUCED_DIR):
        shutil.rmtree(REDUCED_DIR)
    
    # Reduce each split with class balancing
    print("\n--- TRAIN ---")
    reduce_split("train", SAMPLES_PER_CLASS, INCLUDE_BACKGROUND_ONLY, MAX_BACKGROUND_ONLY, scale=1.0)
    
    print("\n--- VAL ---")
    reduce_split("val", SAMPLES_PER_CLASS, INCLUDE_BACKGROUND_ONLY, MAX_BACKGROUND_ONLY, scale=0.2)
    
    print("\n--- TEST ---")
    reduce_split("test", SAMPLES_PER_CLASS, INCLUDE_BACKGROUND_ONLY, MAX_BACKGROUND_ONLY, scale=0.2)
    
    print("\n" + "=" * 50)
    print("✓ Done! Dataset balanced by class.")
    print("  Next: re-run preprocess_images.ipynb to regenerate masks")
    print("  Then: train with the balanced dataset")
    print("=" * 50)


if __name__ == "__main__":
    main()
