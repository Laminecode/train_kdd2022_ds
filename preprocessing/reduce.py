"""
Reduce dataset size with CLASS-BALANCED sampling.
Ensures each class has EXACTLY the target number of instances.
Handles multi-class images intelligently.
"""
import os
import shutil
import random
from collections import defaultdict, Counter

# ============ CONFIG ============
SEED = 42

# Target INSTANCES PER CLASS (not images!)
# Each instance of a class in an image counts toward this target
SAMPLES_PER_CLASS = {
    0: 3000,   # Longitudinal crack
    1: 3000,   # Transverse crack
    2: 3000,   # Alligator crack
    3: 3000,   # Other damage
    4: 3000,   # Pothole
}

INCLUDE_BACKGROUND_ONLY = True
MAX_BACKGROUND_ONLY = 350  # Max background-only images to include

# Paths
# Option 1: Automatic path detection (current method)
PROJECT_ROOT = r"C:\Users\hp\Desktop\DeepLearning\train_kdd_ds"

# Option 2: Manual path specification (UNCOMMENT and MODIFY if needed)
# PROJECT_ROOT = r"C:\Users\hp\Desktop\DeepLearning"

RAW_DIR = os.path.join(PROJECT_ROOT, "dataset/raw")
REDUCED_DIR = os.path.join(PROJECT_ROOT, "dataset/reduced")

# Debug: Print paths to verify
print(f"DEBUG - Script location: {__file__}")
print(f"DEBUG - PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DEBUG - RAW_DIR: {RAW_DIR}")
print(f"DEBUG - Looking for: {os.path.join(RAW_DIR, 'train', 'images')}")


def count_class_instances(label_path):
    """Count how many times each class appears in a label file."""
    class_counts = Counter()
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls = int(float(parts[0]))
                            class_counts[cls] += 1
                        except:
                            pass
        except:
            pass
    return class_counts


def analyze_split(src_images, src_labels):
    """Analyze images and build class mappings with instance counts."""
    images = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Map: class_id -> list of (image, instance_count) tuples
    class_to_images = defaultdict(list)
    # Map: image -> Counter of class instances
    image_to_class_counts = {}
    # List of background-only images
    background_only = []
    
    for img_name in images:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(src_labels, label_name)
        
        class_counts = count_class_instances(label_path)
        image_to_class_counts[img_name] = class_counts
        
        if not class_counts:
            background_only.append(img_name)
        else:
            for cls, count in class_counts.items():
                class_to_images[cls].append((img_name, count))
    
    return class_to_images, image_to_class_counts, background_only


def exact_balanced_sample(class_to_images, image_to_class_counts, background_only,
                          samples_per_class, include_bg=False, max_bg=200):
    """
    Select images to get EXACTLY the target number of instances per class.
    Uses greedy algorithm with smart ordering.
    """
    selected = set()
    class_instance_counts = Counter()
    
    # Calculate total available instances per class
    available_instances = {}
    for cls, img_list in class_to_images.items():
        available_instances[cls] = sum(count for _, count in img_list)
    
    print(f"\n  Available instances per class: {dict(sorted(available_instances.items()))}")
    
    # Sort classes by rarity (rarest first) to prioritize them
    all_classes = sorted(class_to_images.keys(), 
                        key=lambda c: available_instances.get(c, 0))
    
    # Greedy selection: prioritize rarest classes first
    for cls in all_classes:
        target = samples_per_class.get(cls, available_instances[cls])
        
        if target > available_instances[cls]:
            print(f"  ⚠️  Class {cls}: requested {target} but only {available_instances[cls]} available")
            target = available_instances[cls]
        
        # Get all images for this class with their instance counts
        # Sort by: already selected (yes/no), then by instance count (desc for efficiency)
        candidates = sorted(
            class_to_images[cls],
            key=lambda x: (x[0] in selected, -x[1])
        )
        
        for img_name, instance_count in candidates:
            if class_instance_counts[cls] >= target:
                break
            
            # Check how many instances this image would add
            if class_instance_counts[cls] + instance_count <= target:
                # Can add without exceeding
                if img_name not in selected:
                    selected.add(img_name)
                    # Update counts for all classes in this image
                    for c, cnt in image_to_class_counts[img_name].items():
                        class_instance_counts[c] += cnt
            elif class_instance_counts[cls] < target:
                # Adding this would exceed target, but we still need more
                # Add it only if it gets us closer to exact target
                remaining = target - class_instance_counts[cls]
                if remaining >= instance_count / 2:  # Threshold for inclusion
                    if img_name not in selected:
                        selected.add(img_name)
                        for c, cnt in image_to_class_counts[img_name].items():
                            class_instance_counts[c] += cnt
    
    # Second pass: try to reach exact targets by adding single-class images
    for cls in all_classes:
        target = samples_per_class.get(cls, available_instances[cls])
        
        if class_instance_counts[cls] < target:
            # Find single-class images not yet selected
            single_class_imgs = [
                (img, cnt) for img, cnt in class_to_images[cls]
                if img not in selected and len(image_to_class_counts[img]) == 1
            ]
            
            for img_name, instance_count in single_class_imgs:
                if class_instance_counts[cls] >= target:
                    break
                if class_instance_counts[cls] + instance_count <= target:
                    selected.add(img_name)
                    class_instance_counts[cls] += instance_count
    
    # Add background-only images
    if include_bg and background_only:
        bg_sample = random.sample(background_only, min(max_bg, len(background_only)))
        selected.update(bg_sample)
        print(f"  Added {len(bg_sample)} background-only images")
    
    print(f"  Selected instances per class: {dict(sorted(class_instance_counts.items()))}")
    print(f"  Target instances per class:   {dict(sorted(samples_per_class.items()))}")
    
    # Show differences
    differences = {cls: class_instance_counts[cls] - samples_per_class.get(cls, 0) 
                   for cls in samples_per_class.keys()}
    print(f"  Difference (actual - target):  {dict(sorted(differences.items()))}")
    
    return list(selected)


def reduce_split(split_name, samples_per_class, include_bg=False, max_bg=200, scale=1.0):
    """Reduce a single split with exact class-balanced sampling."""
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
    class_to_images, image_to_class_counts, background_only = analyze_split(src_images, src_labels)
    
    # Scale samples_per_class for val/test
    scaled_spc = {
        c: int(n * scale) if n is not None else None 
        for c, n in samples_per_class.items()
    }
    
    # Sample
    selected = exact_balanced_sample(
        class_to_images, image_to_class_counts, background_only,
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
    print("EXACT CLASS-BALANCED DATASET REDUCTION")
    print("=" * 50)
    
    # Print debug info first
    print("\n📍 PATH VERIFICATION:")
    print(f"   Script: {os.path.abspath(__file__)}")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Raw Dir: {RAW_DIR}")
    print(f"   Reduced Dir: {REDUCED_DIR}")
    
    # Check if raw directory exists
    if not os.path.exists(RAW_DIR):
        print(f"\n❌ ERROR: RAW_DIR does not exist!")
        print(f"   Expected: {RAW_DIR}")
        print(f"\n💡 SOLUTION:")
        print(f"   1. Check your dataset structure")
        print(f"   2. Modify RAW_DIR in the script to point to your actual data")
        print(f"   3. Expected structure:")
        print(f"      {RAW_DIR}/")
        print(f"      ├── train/")
        print(f"      │   ├── images/")
        print(f"      │   └── labels/")
        print(f"      ├── val/")
        print(f"      └── test/")
        return
    
    print(f"\n✅ Raw directory found!")
    print(f"\nTarget instances per class: {SAMPLES_PER_CLASS}")
    print(f"Include background-only: {INCLUDE_BACKGROUND_ONLY} (max {MAX_BACKGROUND_ONLY})")
    print(f"Output: {REDUCED_DIR}\n")
    
    # Clear existing
    if os.path.exists(REDUCED_DIR):
        shutil.rmtree(REDUCED_DIR)
    
    # Reduce each split
    print("\n--- TRAIN ---")
    reduce_split("train", SAMPLES_PER_CLASS, INCLUDE_BACKGROUND_ONLY, MAX_BACKGROUND_ONLY, scale=1.0)
    
    print("\n--- VAL ---")
    reduce_split("val", SAMPLES_PER_CLASS, INCLUDE_BACKGROUND_ONLY, MAX_BACKGROUND_ONLY, scale=0.2)
    
    print("\n--- TEST ---")
    reduce_split("test", SAMPLES_PER_CLASS, INCLUDE_BACKGROUND_ONLY, MAX_BACKGROUND_ONLY, scale=0.2)
    
    print("\n" + "=" * 50)
    print("✅ Done! Dataset balanced by exact class instances.")
    print("  Next: re-run preprocess_images.ipynb to regenerate masks")
    print("  Then: train with the balanced dataset")
    print("=" * 50)


if __name__ == "__main__":
    main()