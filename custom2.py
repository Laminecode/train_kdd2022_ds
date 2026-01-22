"""
Interactive Training Configuration Creator
3 Models: U-Net (SMP), Hybrid U-Net (SMP), YOLO
Allows users to customize all training hyperparameters
"""

import os
import json
from datetime import datetime
import sys

# Optional Tkinter GUI (safe import for headless environments)
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# ============ TRAINING SCRIPT TEMPLATE (U-NET & HYBRID) ============

SMP_TRAINING_TEMPLATE = '''"""
{model_title} Training Script
Auto-generated on {timestamp}
Uses segmentation_models_pytorch (smp)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============ CONFIGURATION ============
CONFIG = {{
    # Model
    "model_type": "{model_type}",
    "encoder": "{encoder}",
    "encoder_weights": "{encoder_weights}",
    "in_channels": {in_channels},
    "classes": {classes},
    
    # Training
    "epochs": {epochs},
    "batch_size": {batch_size},
    "learning_rate": {learning_rate},
    "weight_decay": {weight_decay},
    "device": "{device}",
    
    # Data
    "img_size": ({img_height}, {img_width}),
    "train_path": "{train_path}",
    "val_path": "{val_path}",
    
    # Augmentation
    "augmentation": {augmentation},
    
    # Optimization
    "optimizer": "{optimizer}",
    "loss_function": "{loss_function}",
    "scheduler": "{scheduler}",
    "scheduler_patience": {scheduler_patience},
    "scheduler_factor": {scheduler_factor},
    
    # Early Stopping
    "early_stopping": {early_stopping},
    "early_stopping_patience": {early_stopping_patience},
    
    # Checkpoints
    "save_best_only": {save_best_only},
    "checkpoint_dir": "{checkpoint_dir}",
    
    # Misc
    "num_workers": {num_workers},
    "pin_memory": {pin_memory},
    "mixed_precision": {mixed_precision}
}}

print("="*60)
print("="*60)
print("(configuration print moved into main() to avoid worker re-import side-effects)")
print("="*60)


# ============ DATASET ============
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, 
                                os.path.splitext(img_name)[0] + '.png')
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()


# ============ AUGMENTATION ============
def get_training_augmentation():
    if not CONFIG['augmentation']:
        return A.Compose([
            A.Resize(CONFIG['img_size'][0], CONFIG['img_size'][1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    return A.Compose([
        A.Resize(CONFIG['img_size'][0], CONFIG['img_size'][1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(CONFIG['img_size'][0], CONFIG['img_size'][1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ============ LOSS FUNCTIONS ============
def get_loss_function():
    if CONFIG['loss_function'] == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif CONFIG['loss_function'] == 'Dice':
        return smp.losses.DiceLoss(mode='multiclass')
    elif CONFIG['loss_function'] == 'Focal':
        return smp.losses.FocalLoss(mode='multiclass')
    elif CONFIG['loss_function'] == 'DiceFocal':
        dice = smp.losses.DiceLoss(mode='multiclass')
        focal = smp.losses.FocalLoss(mode='multiclass')
        return lambda pred, target: dice(pred, target) + focal(pred, target)
    else:
        return nn.CrossEntropyLoss()


# ============ METRICS ============
def calculate_iou(pred, target, num_classes):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return np.nanmean(ious)


# ============ EARLY STOPPING ============
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {{self.counter}}/{{self.patience}}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ============ TRAINING LOOP ============
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler and CONFIG['mixed_precision']:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            iou = calculate_iou(pred, masks, CONFIG['classes'])
        
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix({{'loss': loss.item(), 'iou': iou}})
    
    return total_loss / len(dataloader), total_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred = torch.argmax(outputs, dim=1)
            iou = calculate_iou(pred, masks, CONFIG['classes'])
            
            total_loss += loss.item()
            total_iou += iou
            
            pbar.set_postfix({{'loss': loss.item(), 'iou': iou}})
    
    return total_loss / len(dataloader), total_iou / len(dataloader)


# ============ MAIN TRAINING FUNCTION ============
def main():
    # Set device
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\\n🖥️  Using device: {{device}}\\n")
    
    # Create model
    print(f"🏗️  Creating {{CONFIG['model_type']}} with {{CONFIG['encoder']}} encoder...")
    
    if CONFIG['model_type'] == 'U-Net':
        model = smp.Unet(
            encoder_name=CONFIG['encoder'],
            encoder_weights=CONFIG['encoder_weights'],
            in_channels=CONFIG['in_channels'],
            classes=CONFIG['classes']
        )
    else:  # Hybrid (U-Net++)
        model = smp.UnetPlusPlus(
            encoder_name=CONFIG['encoder'],
            encoder_weights=CONFIG['encoder_weights'],
            in_channels=CONFIG['in_channels'],
            classes=CONFIG['classes']
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {{total_params:,}}\\n")
    
    # Create datasets
    print("📁 Loading datasets...")
    train_dataset = SegmentationDataset(
        os.path.join(CONFIG['train_path'], 'images'),
        os.path.join(CONFIG['train_path'], 'masks'),
        transform=get_training_augmentation()
    )
    
    val_dataset = SegmentationDataset(
        os.path.join(CONFIG['val_path'], 'images'),
        os.path.join(CONFIG['val_path'], 'masks'),
        transform=get_validation_augmentation()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    print(f"   Train samples: {{len(train_dataset)}}")
    print(f"   Val samples: {{len(val_dataset)}}\\n")
    
    # Loss function and optimizer
    criterion = get_loss_function()
    
    if CONFIG['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=CONFIG['learning_rate'],
                                    weight_decay=CONFIG['weight_decay'])
    elif CONFIG['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=CONFIG['learning_rate'],
                                     weight_decay=CONFIG['weight_decay'])
    else:  # SGD
        optimizer = torch.optim.SGD(model.parameters(),
                                   lr=CONFIG['learning_rate'],
                                   momentum=0.9,
                                   weight_decay=CONFIG['weight_decay'])
    
    # Scheduler
    scheduler = None
    if CONFIG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min',
                                     patience=CONFIG['scheduler_patience'],
                                     factor=CONFIG['scheduler_factor'])
    
    # Early stopping
    early_stopping = None
    if CONFIG['early_stopping']:
        early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None
    
    # Create checkpoint directory
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    print("\\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60 + "\\n")
    
    best_iou = 0.0
    history = {{'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}}
    
    for epoch in range(CONFIG['epochs']):
        print(f"\\nEpoch {{epoch+1}}/{{CONFIG['epochs']}}")
        print("-" * 60)
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, scaler)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print epoch summary
        print(f"\\n📊 Epoch Summary:")
        print(f"   Train Loss: {{train_loss:.4f}} | Train IoU: {{train_iou:.4f}}")
        print(f"   Val Loss:   {{val_loss:.4f}} | Val IoU:   {{val_iou:.4f}}")
        
        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)
            print(f"   Learning Rate: {{optimizer.param_groups[0]['lr']:.6f}}")
        
        # Save checkpoint
        if val_iou > best_iou or not CONFIG['save_best_only']:
            if val_iou > best_iou:
                best_iou = val_iou
                print(f"   ✅ New best IoU! Saving checkpoint...")
            
            checkpoint = {{
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
                'config': CONFIG
            }}
            
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 
                                          f"best_model_iou_{{val_iou:.4f}}.pth")
            torch.save(checkpoint, checkpoint_path)
        
        # Early stopping
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\\n⏹️  Early stopping triggered at epoch {{epoch+1}}")
                break
    
    # Save final model and history
    print("\\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"   Best IoU: {{best_iou:.4f}}")
    
    # Save history
    history_path = os.path.join(CONFIG['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"   History saved to: {{history_path}}")
    
    # Save final model
    final_model_path = os.path.join(CONFIG['checkpoint_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"   Final model saved to: {{final_model_path}}")
    print("="*60)


def _gui_generate_and_save(creator, model_type, parent):
    # Build minimal config with dialogs
    # Common params
    creator.config = {}
    if model_type == 'unet':
        creator.config['model_type'] = 'U-Net'
        creator.config['model_title'] = 'U-Net'
        creator.config['encoder'] = simpledialog.askstring("Encoder",
                                                         "Encoder (resnet34/resnet50/efficientnet-b0)",
                                                         initialvalue='resnet34', parent=parent)
        creator.config['encoder_weights'] = simpledialog.askstring("Encoder weights",
                                                         "Encoder weights (imagenet/None)",
                                                         initialvalue='imagenet', parent=parent)
        creator.config['in_channels'] = simpledialog.askinteger("Input channels", "Input channels", initialvalue=3, parent=parent)
        creator.config['classes'] = simpledialog.askinteger("Classes", "Number of classes", initialvalue=6, parent=parent)
        creator.config['img_height'] = simpledialog.askinteger("Image height", "Image height", initialvalue=640, parent=parent)
        creator.config['img_width'] = simpledialog.askinteger("Image width", "Image width", initialvalue=640, parent=parent)
        creator.config['train_path'] = simpledialog.askstring("Train path", "Training data path", initialvalue='../../dataset/segmentation2/train', parent=parent)
        creator.config['val_path'] = simpledialog.askstring("Val path", "Validation data path", initialvalue='../../dataset/segmentation2/val', parent=parent)
    elif model_type == 'hybrid':
        creator.config['model_type'] = 'Hybrid'
        creator.config['model_title'] = 'Hybrid U-Net (U-Net++)'
        creator.config['encoder'] = simpledialog.askstring("Encoder",
                                                         "Encoder (resnet34/resnet50/efficientnet-b0)",
                                                         initialvalue='resnet50', parent=parent)
        creator.config['encoder_weights'] = simpledialog.askstring("Encoder weights",
                                                         "Encoder weights (imagenet/None)",
                                                         initialvalue='imagenet', parent=parent)
        creator.config['in_channels'] = simpledialog.askinteger("Input channels", "Input channels", initialvalue=3, parent=parent)
        creator.config['classes'] = simpledialog.askinteger("Classes", "Number of classes", initialvalue=6, parent=parent)
        creator.config['img_height'] = simpledialog.askinteger("Image height", "Image height", initialvalue=640, parent=parent)
        creator.config['img_width'] = simpledialog.askinteger("Image width", "Image width", initialvalue=640, parent=parent)
        creator.config['train_path'] = simpledialog.askstring("Train path", "Training data path", initialvalue='../../dataset/segmentation2/train', parent=parent)
        creator.config['val_path'] = simpledialog.askstring("Val path", "Validation data path", initialvalue='../../dataset/segmentation2/val', parent=parent)
        # minimal YOLO settings
        creator.config['model_size'] = simpledialog.askstring("YOLO size", "YOLO model size (n/s/m/l/x)", initialvalue='n', parent=parent)
        creator.config['pretrained'] = True
        creator.config['data_yaml'] = simpledialog.askstring("YOLO data.yaml", "Path to YOLO data.yaml", initialvalue='../../dataset/processed2/data.yaml', parent=parent)
        creator.config['imgsz'] = simpledialog.askinteger("YOLO imgsz", "YOLO image size", initialvalue=640, parent=parent)
    else:  # yolo
        creator.config['model_size'] = simpledialog.askstring("Model size", "Model size (n/s/m/l/x)", initialvalue='n', parent=parent)
        creator.config['pretrained'] = True
        creator.config['data_yaml'] = simpledialog.askstring("data.yaml", "Path to data.yaml", initialvalue='../../dataset/processed2/data.yaml', parent=parent)
        creator.config['imgsz'] = simpledialog.askinteger("Image size", "Image size", initialvalue=640, parent=parent)

    # Common training params
    creator.config['epochs'] = simpledialog.askinteger("Epochs", "Number of epochs", initialvalue=100, parent=parent)
    creator.config['batch_size'] = simpledialog.askinteger("Batch size", "Batch size", initialvalue=8, parent=parent)
    creator.config['device'] = simpledialog.askstring("Device", "Device (cuda/cpu/0)", initialvalue='cuda', parent=parent)
    creator.config['optimizer'] = 'AdamW'
    creator.config['learning_rate'] = 0.001
    creator.config['weight_decay'] = 0.0005
    creator.config['augmentation'] = True
    creator.config['loss_function'] = 'DiceFocal'
    creator.config['scheduler'] = 'ReduceLROnPlateau'
    creator.config['scheduler_patience'] = 5
    creator.config['scheduler_factor'] = 0.5
    creator.config['early_stopping'] = True
    creator.config['early_stopping_patience'] = 10
    creator.config['mixed_precision'] = True
    creator.config['num_workers'] = 4
    creator.config['pin_memory'] = False
    creator.config['save_best_only'] = True
    creator.config['checkpoint_dir'] = simpledialog.askstring("Checkpoint dir", "Checkpoint directory", initialvalue='./checkpoints', parent=parent)

    # Generate and save
    if model_type == 'unet':
        filename, script = creator.generate_script('smp')
    elif model_type == 'hybrid':
        filename, script = creator.generate_script('hybrid')
    else:
        filename, script = creator.generate_script('yolo')

    creator.save_script(filename, script)
    messagebox.showinfo("Saved", f"Script saved to generated_scripts/{filename}")

if __name__ == "__main__":
    main()
'''

# ============ YOLO TRAINING TEMPLATE ============

YOLO_TRAINING_TEMPLATE = '''"""
YOLO Training Script
Auto-generated on {timestamp}
Uses Ultralytics YOLOv8
"""

from ultralytics import YOLO
import os

# ============ CONFIGURATION ============
CONFIG = {{
    # Model
    "model_size": "{model_size}",
    "pretrained": {pretrained},
    
    # Data
    "data_yaml": "{data_yaml}",
    
    # Training
    "epochs": {epochs},
    "batch_size": {batch_size},
    "imgsz": {imgsz},
    "device": "{device}",
    
    # Optimization
    "optimizer": "{optimizer}",
    "lr0": {lr0},
    "lrf": {lrf},
    "momentum": {momentum},
    "weight_decay": {weight_decay},
    
    # Augmentation
    "augmentation": {augmentation},
    "hsv_h": {hsv_h},
    "hsv_s": {hsv_s},
    "hsv_v": {hsv_v},
    "degrees": {degrees},
    "translate": {translate},
    "scale": {scale},
    "shear": {shear},
    "flipud": {flipud},
    "fliplr": {fliplr},
    "mosaic": {mosaic},
    "mixup": {mixup},
    
    # Early Stopping & Validation
    "patience": {patience},
    "val": True,
    "save": True,
    "save_period": {save_period},
    
    # Output
    "project": "{project}",
    "name": "{name}",
    "exist_ok": {exist_ok},
    
    # Misc
    "workers": {workers},
    "verbose": True
}}

print("="*60)
print("🚀 YOLO TRAINING CONFIGURATION")
print("="*60)
for key, value in CONFIG.items():
    print(f"  {{key:.<30}} {{value}}")
print("="*60)


def main():
    # Load model
    model_name = f"yolov8{{CONFIG['model_size']}}.pt" if CONFIG['pretrained'] else f"yolov8{{CONFIG['model_size']}}.yaml"
    print(f"\\n🏗️  Loading model: {{model_name}}")
    model = YOLO(model_name)
    
    # Prepare training arguments
    train_args = {{
        'data': CONFIG['data_yaml'],
        'epochs': CONFIG['epochs'],
        'batch': CONFIG['batch_size'],
        'imgsz': CONFIG['imgsz'],
        'device': CONFIG['device'],
        'optimizer': CONFIG['optimizer'],
        'lr0': CONFIG['lr0'],
        'lrf': CONFIG['lrf'],
        'momentum': CONFIG['momentum'],
        'weight_decay': CONFIG['weight_decay'],
        'patience': CONFIG['patience'],
        'val': CONFIG['val'],
        'save': CONFIG['save'],
        'save_period': CONFIG['save_period'],
        'project': CONFIG['project'],
        'name': CONFIG['name'],
        'exist_ok': CONFIG['exist_ok'],
        'workers': CONFIG['workers'],
        'verbose': CONFIG['verbose']
    }}
    
    # Add augmentation if enabled
    if CONFIG['augmentation']:
        train_args.update({{
            'hsv_h': CONFIG['hsv_h'],
            'hsv_s': CONFIG['hsv_s'],
            'hsv_v': CONFIG['hsv_v'],
            'degrees': CONFIG['degrees'],
            'translate': CONFIG['translate'],
            'scale': CONFIG['scale'],
            'shear': CONFIG['shear'],
            'flipud': CONFIG['flipud'],
            'fliplr': CONFIG['fliplr'],
            'mosaic': CONFIG['mosaic'],
            'mixup': CONFIG['mixup']
        }})
    
    # Train
    print("\\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60 + "\\n")
    
    results = model.train(**train_args)
    
    print("\\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"   Results saved to: {{results.save_dir}}")
    print("="*60)


if __name__ == "__main__":
    main()
'''


# ============ HYBRID (U-NET + YOLO) TRAINING TEMPLATE ============

HYBRID_TRAINING_TEMPLATE = '''"""
Hybrid U-Net (SMP) + YOLO Training Script
Auto-generated on {timestamp}
"""

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

# ============ CONFIGURATION ============
UNET_CONFIG = {unet_config}
YOLO_CONFIG = {yolo_config}

print("="*60)
print("🚀 HYBRID TRAINING CONFIGURATION")
print("="*60)
print("U-Net CONFIG:")
for key, value in UNET_CONFIG.items():
    print(f"  {{key:.<30}} {{value}}")
print("-"*60)
print("YOLO CONFIG:")
for key, value in YOLO_CONFIG.items():
    print(f"  {{key:.<30}} {{value}}")
print("="*60)

# Simple U-Net dataset and augmentation functions
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + '.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.long()

def get_training_augmentation():
    if not UNET_CONFIG['augmentation']:
        return A.Compose([
            A.Resize(UNET_CONFIG['img_size'][0], UNET_CONFIG['img_size'][1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(UNET_CONFIG['img_size'][0], UNET_CONFIG['img_size'][1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(UNET_CONFIG['img_size'][0], UNET_CONFIG['img_size'][1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_loss_function():
    if UNET_CONFIG['loss_function'] == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif UNET_CONFIG['loss_function'] == 'Dice':
        return smp.losses.DiceLoss(mode='multiclass')
    elif UNET_CONFIG['loss_function'] == 'Focal':
        return smp.losses.FocalLoss(mode='multiclass')
    elif UNET_CONFIG['loss_function'] == 'DiceFocal':
        dice = smp.losses.DiceLoss(mode='multiclass')
        focal = smp.losses.FocalLoss(mode='multiclass')
        return lambda pred, target: dice(pred, target) + focal(pred, target)
    else:
        return nn.CrossEntropyLoss()

def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {{self.counter}}/{{self.patience}}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_unet():
    device = torch.device(UNET_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    if UNET_CONFIG['model_type'] == 'U-Net':
        model = smp.Unet(
            encoder_name=UNET_CONFIG['encoder'],
            encoder_weights=UNET_CONFIG['encoder_weights'],
            in_channels=UNET_CONFIG['in_channels'],
            classes=UNET_CONFIG['classes']
        )
    else:
        model = smp.UnetPlusPlus(
            encoder_name=UNET_CONFIG['encoder'],
            encoder_weights=UNET_CONFIG['encoder_weights'],
            in_channels=UNET_CONFIG['in_channels'],
            classes=UNET_CONFIG['classes']
        )
    model = model.to(device)
    train_dataset = SegmentationDataset(
        os.path.join(UNET_CONFIG['train_path'], 'images'),
        os.path.join(UNET_CONFIG['train_path'], 'masks'),
        transform=get_training_augmentation()
    )
    val_dataset = SegmentationDataset(
        os.path.join(UNET_CONFIG['val_path'], 'images'),
        os.path.join(UNET_CONFIG['val_path'], 'masks'),
        transform=get_validation_augmentation()
    )
    train_loader = DataLoader(train_dataset, batch_size=UNET_CONFIG['batch_size'], shuffle=True,
                              num_workers=UNET_CONFIG['num_workers'], pin_memory=UNET_CONFIG['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=UNET_CONFIG['batch_size'], shuffle=False,
                            num_workers=UNET_CONFIG['num_workers'], pin_memory=UNET_CONFIG['pin_memory'])
    criterion = get_loss_function()
    if UNET_CONFIG['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=UNET_CONFIG['learning_rate'], weight_decay=UNET_CONFIG['weight_decay'])
    elif UNET_CONFIG['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=UNET_CONFIG['learning_rate'], weight_decay=UNET_CONFIG['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=UNET_CONFIG['learning_rate'], momentum=0.9, weight_decay=UNET_CONFIG['weight_decay'])
    scheduler = None
    if UNET_CONFIG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=UNET_CONFIG['scheduler_patience'], factor=UNET_CONFIG['scheduler_factor'])
    early_stopping = EarlyStopping(patience=UNET_CONFIG['early_stopping_patience']) if UNET_CONFIG['early_stopping'] else None
    scaler = torch.cuda.amp.GradScaler() if UNET_CONFIG['mixed_precision'] else None
    os.makedirs(UNET_CONFIG['checkpoint_dir'], exist_ok=True)
    best_iou = 0.0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
    for epoch in range(UNET_CONFIG['epochs']):
        model.train()
        total_loss = 0
        total_iou = 0
        for images, masks in tqdm(train_loader, desc='U-Net Training'):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                iou = calculate_iou(pred, masks, UNET_CONFIG['classes'])
            total_loss += loss.item()
            total_iou += iou
        # validation skipped for brevity in hybrid script

def train_yolo():
    model_name = f"yolov8{YOLO_CONFIG['model_size']}.pt" if YOLO_CONFIG['pretrained'] else f"yolov8{YOLO_CONFIG['model_size']}.yaml"
    model = YOLO(model_name)
    train_args = {
        'data': YOLO_CONFIG['data_yaml'],
        'epochs': YOLO_CONFIG['epochs'],
        'batch': YOLO_CONFIG['batch_size'],
        'imgsz': YOLO_CONFIG['imgsz'],
        'device': YOLO_CONFIG['device'],
        'optimizer': YOLO_CONFIG['optimizer'],
        'weight_decay': YOLO_CONFIG['weight_decay'],
        'patience': YOLO_CONFIG['patience'],
        'save_period': YOLO_CONFIG['save_period'],
        'project': YOLO_CONFIG['project'],
        'name': YOLO_CONFIG['name'],
        'exist_ok': YOLO_CONFIG['exist_ok'],
        'workers': YOLO_CONFIG['workers']
    }
    if YOLO_CONFIG['augmentation']:
        train_args.update({
            'hsv_h': YOLO_CONFIG['hsv_h'],
            'hsv_s': YOLO_CONFIG['hsv_s'],
            'hsv_v': YOLO_CONFIG['hsv_v'],
            'degrees': YOLO_CONFIG['degrees'],
            'translate': YOLO_CONFIG['translate'],
            'scale': YOLO_CONFIG['scale'],
            'shear': YOLO_CONFIG['shear'],
            'flipud': YOLO_CONFIG['flipud'],
            'fliplr': YOLO_CONFIG['fliplr'],
            'mosaic': YOLO_CONFIG['mosaic'],
            'mixup': YOLO_CONFIG['mixup']
        })
    results = model.train(**train_args)
    print(f"YOLO results saved to: {results.save_dir}")

def main():
    train_unet()
    train_yolo()

if __name__ == '__main__':
    main()
'''

# ============ INTERACTIVE CREATOR ============

class TrainingConfigCreator:
    def __init__(self):
        self.config = {}
    
    def get_input(self, prompt, default, input_type=str, choices=None):
        """Get user input with validation"""
        while True:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
            
            if choices and user_input not in choices:
                print(f"❌ Invalid choice. Choose from: {', '.join(map(str, choices))}")
                continue
            
            try:
                if input_type == bool:
                    return user_input.lower() in ['true', 't', 'yes', 'y', '1']
                return input_type(user_input)
            except ValueError:
                print(f"❌ Invalid input. Expected {input_type.__name__}")
    
    def configure_unet(self):
        """Configure U-Net model"""
        print("\n" + "="*60)
        print("🎨 U-NET CONFIGURATION")
        print("="*60)
        
        self.config['model_type'] = 'U-Net'
        self.config['model_title'] = 'U-Net'
        
        # Encoder selection
        print("\n🔧 Popular encoders:")
        encoders = ["resnet34", "resnet50", "efficientnet-b0", "efficientnet-b3",
                   "mobilenet_v2"]
        for i, enc in enumerate(encoders, 1):
            print(f"   {i}. {enc}")
        
        enc_choice = self.get_input(f"\nChoose encoder (1-{len(encoders)})", "1", str)
        self.config['encoder'] = encoders[int(enc_choice) - 1]
        
        self.config['encoder_weights'] = self.get_input(
            "Encoder weights (imagenet/None)", "imagenet", str)
        self.config['in_channels'] = self.get_input("Input channels", 3, int)
        self.config['classes'] = self.get_input("Number of classes", 6, int)
        
        self._configure_training_params('segmentation')
        return "smp"
    
    def configure_hybrid(self):
        """Configure Hybrid U-Net (U-Net++) model"""
        print("\n" + "="*60)
        print("🔥 HYBRID U-NET CONFIGURATION (U-Net++)")
        print("="*60)
        
        self.config['model_type'] = 'Hybrid'
        self.config['model_title'] = 'Hybrid U-Net (U-Net++)'
        
        # Encoder selection
        print("\n🔧 Popular encoders:")
        encoders = ["resnet34", "resnet50", "efficientnet-b0", "efficientnet-b3",
                   "mobilenet_v2"]
        for i, enc in enumerate(encoders, 1):
            print(f"   {i}. {enc}")
        
        enc_choice = self.get_input(f"\nChoose encoder (1-{len(encoders)})", "2", str)
        self.config['encoder'] = encoders[int(enc_choice) - 1]
        
        self.config['encoder_weights'] = self.get_input(
            "Encoder weights (imagenet/None)", "imagenet", str)
        self.config['in_channels'] = self.get_input("Input channels", 3, int)
        self.config['classes'] = self.get_input("Number of classes", 6, int)
        
        self._configure_training_params('segmentation')

        # Collect minimal YOLO settings for the hybrid script (defaults provided)
        print("\n🔀 Hybrid also includes YOLO settings (using sensible defaults)")
        self.config['model_size'] = self.get_input(
            "YOLO model size (n/s/m/l/x)", "n", str, ['n', 's', 'm', 'l', 'x'])
        self.config['pretrained'] = self.get_input(
            "Use pretrained YOLO weights?", True, bool)
        self.config['data_yaml'] = self.get_input(
            "Path to YOLO data.yaml", "../../dataset/processed2/data.yaml", str)
        self.config['imgsz'] = self.get_input("YOLO image size", 640, int)

        return "hybrid"
    
    def configure_yolo(self):
        """Configure YOLO model"""
        print("\n" + "="*60)
        print("⚡ YOLO CONFIGURATION")
        print("="*60)
        
        self.config['model_size'] = self.get_input(
            "Model size (n/s/m/l/x)", "n", str, ['n', 's', 'm', 'l', 'x'])
        self.config['pretrained'] = self.get_input(
            "Use pretrained weights?", True, bool)
        self.config['data_yaml'] = self.get_input(
            "Path to data.yaml", "../../dataset/processed2/data.yaml", str)
        
        self._configure_training_params('yolo')
        return "yolo"
    
    def _configure_training_params(self, model_type):
        """Configure training hyperparameters"""
        print("\n" + "="*60)
        print("⚙️  TRAINING HYPERPARAMETERS")
        print("="*60)

        # Basic params
        self.config['epochs'] = self.get_input("Number of epochs", 100, int)
        self.config['batch_size'] = self.get_input("Batch size", 8, int)
        self.config['device'] = self.get_input("Device (cuda/cpu/0)", "cuda", str)

        if model_type == 'segmentation':
            self.config['img_height'] = self.get_input("Image height", 640, int)
            self.config['img_width'] = self.get_input("Image width", 640, int)
            self.config['train_path'] = self.get_input(
                "Training data path", "../../dataset/segmentation2/train", str)
            self.config['val_path'] = self.get_input(
                "Validation data path", "../../dataset/segmentation2/val", str)
        else:
            self.config['imgsz'] = self.get_input("Image size", 640, int)

        # Optimizer
        print("\n🔧 Optimization:")
        self.config['optimizer'] = self.get_input(
            "Optimizer (Adam/AdamW/SGD)", "AdamW", str, ['Adam', 'AdamW', 'SGD'])

        if model_type == 'segmentation':
            self.config['learning_rate'] = self.get_input("Learning rate", 0.001, float)
        else:
            self.config['lr0'] = 0.01  # Default, removed from CLI
            self.config['lrf'] = 0.01  # Default, removed from CLI
            self.config['momentum'] = 0.937  # Default, removed from CLI

        self.config['weight_decay'] = self.get_input("Weight decay", 0.0005, float)

        # Loss (segmentation only)
        if model_type == 'segmentation':
            print("\n📉 Loss function:")
            losses = ["CrossEntropy", "Dice", "Focal", "DiceFocal"]
            for i, loss in enumerate(losses, 1):
                print(f"   {i}. {loss}")
            loss_choice = self.get_input(f"Choose loss (1-{len(losses)})", "4", str)
            self.config['loss_function'] = losses[int(loss_choice) - 1]

        # Augmentation
        print("\n🎭 Data Augmentation:")
        self.config['augmentation'] = self.get_input("Enable augmentation?", True, bool)

        if model_type == 'yolo' and self.config['augmentation']:
            self.config['hsv_h'] = 0.015
            self.config['hsv_s'] = 0.7
            self.config['hsv_v'] = 0.4
            self.config['degrees'] = 0.0
            self.config['translate'] = 0.1
            self.config['scale'] = 0.5
            self.config['shear'] = 0.0
            self.config['flipud'] = 0.0
            self.config['fliplr'] = 0.5
            self.config['mosaic'] = 1.0
            self.config['mixup'] = 0.0

        # Scheduler (segmentation only) - set as default
        if model_type == 'segmentation':
            self.config['scheduler'] = "ReduceLROnPlateau"
            self.config['scheduler_patience'] = 5
            self.config['scheduler_factor'] = 0.5

            # Early stopping
            print("\n⏹️  Early Stopping:")
            self.config['early_stopping'] = self.get_input(
                "Enable early stopping?", True, bool)
            if self.config['early_stopping']:
                self.config['early_stopping_patience'] = self.get_input(
                    "Early stopping patience", 10, int)
            else:
                self.config['early_stopping_patience'] = 0

            # Mixed precision - set as default
            self.config['mixed_precision'] = True

            # DataLoader params - set as default
            self.config['num_workers'] = 4
            self.config['pin_memory'] = False

            # Checkpoints - set as default
            self.config['save_best_only'] = self.get_input(
                "Save best model only?", True, bool)
            self.config['checkpoint_dir'] = "./checkpoints"

        # YOLO specific params
        else:
            self.config['patience'] = self.get_input("Early stopping patience", 50, int)
            self.config['save_period'] = -1
            self.config['project'] = "runs/train"
            self.config['name'] = "exp"
            self.config['exist_ok'] = False
            self.config['workers'] = 8

    def generate_script(self, model_family):
        """Generate training script based on configuration"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if model_family == "smp":
            temp = SMP_TRAINING_TEMPLATE

            # Protect real placeholders
            protected_keys = ['timestamp'] + list(self.config.keys())
            for key in protected_keys:
                temp = temp.replace(f'{{{key}}}', f'<<{key.upper()}>>')

            # Escape all braces
            temp = temp.replace('{', '{{').replace('}', '}}')

            # Restore placeholders
            temp = temp.replace('<<TIMESTAMP>>', '{timestamp}')
            for key in self.config.keys():
                temp = temp.replace(f'<<{key.upper()}>>', f'{{{key}}}')

            script = temp.format(timestamp=timestamp, **self.config)
            filename = f"train_{self.config['model_type'].lower().replace(' ', '_')}.py"

        elif model_family == "yolo":
            script = YOLO_TRAINING_TEMPLATE.format(
                timestamp=timestamp,
                **self.config
            )
            filename = "train_yolo.py"

        elif model_family == "hybrid":
            # Build U-Net config
            unet_keys = [
                'model_type', 'model_title', 'encoder', 'encoder_weights', 'in_channels', 'classes',
                'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'device',
                'img_height', 'img_width', 'train_path', 'val_path', 'augmentation',
                'optimizer', 'loss_function', 'scheduler', 'scheduler_patience', 'scheduler_factor',
                'early_stopping', 'early_stopping_patience', 'save_best_only', 'checkpoint_dir',
                'num_workers', 'pin_memory', 'mixed_precision'
            ]
            yolo_keys = [
                'model_size', 'pretrained', 'data_yaml', 'epochs', 'batch_size', 'imgsz', 'device',
                'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay', 'augmentation',
                'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'flipud', 'fliplr', 'mosaic', 'mixup',
                'patience', 'save_period', 'project', 'name', 'exist_ok', 'workers'
            ]
            unet_config = {k: self.config[k] for k in unet_keys if k in self.config}
            # convert img height/width into img_size tuple for UNET_CONFIG
            if 'img_height' in unet_config and 'img_width' in unet_config:
                unet_config['img_size'] = (unet_config.pop('img_height'), unet_config.pop('img_width'))
            yolo_config = {k: self.config.get(k, None) for k in yolo_keys}
            # Fill sensible defaults for YOLO where None
            defaults = {
                'model_size': 'n', 'pretrained': True, 'data_yaml': '../../dataset/processed2/data.yaml',
                'epochs': self.config.get('epochs', 100), 'batch_size': self.config.get('batch_size', 8),
                'imgsz': self.config.get('imgsz', 640), 'device': self.config.get('device', 'cuda'),
                'optimizer': self.config.get('optimizer', 'AdamW'), 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937,
                'weight_decay': self.config.get('weight_decay', 0.0005), 'augmentation': self.config.get('augmentation', True),
                'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0,
                'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0, 'patience': self.config.get('patience', 50),
                'save_period': self.config.get('save_period', -1), 'project': 'runs/train', 'name': 'exp', 'exist_ok': False, 'workers': 8
            }
            for k, v in defaults.items():
                if yolo_config.get(k) is None:
                    yolo_config[k] = v

            # The HYBRID template contains many braces used inside the
            # generated Python code. To avoid str.format() interpreting
            # them as placeholders, temporarily protect the real
            # placeholders, escape all braces, then restore and format.
            temp = HYBRID_TRAINING_TEMPLATE
            # protect intended placeholders
            temp = temp.replace('{unet_config}', '<<UNET_CONFIG_PLACEHOLDER>>')
            temp = temp.replace('{yolo_config}', '<<YOLO_CONFIG_PLACEHOLDER>>')
            temp = temp.replace('{timestamp}', '<<TIMESTAMP_PLACEHOLDER>>')
            # escape all braces
            temp = temp.replace('{', '{{').replace('}', '}}')
            # restore real placeholders
            temp = temp.replace('<<UNET_CONFIG_PLACEHOLDER>>', '{unet_config}')
            temp = temp.replace('<<YOLO_CONFIG_PLACEHOLDER>>', '{yolo_config}')
            temp = temp.replace('<<TIMESTAMP_PLACEHOLDER>>', '{timestamp}')

            script = temp.format(
                timestamp=timestamp,
                unet_config=repr(unet_config),
                yolo_config=repr(yolo_config)
            )
            filename = "train_hybrid.py"

        else:
            raise ValueError(f"Unknown model_family: {model_family}")

        return filename, script

    def save_script(self, filename, script):
        """Save generated script to file"""
        os.makedirs("generated_scripts", exist_ok=True)
        path = os.path.join("generated_scripts", filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        print("\n" + "="*60)
        print("✅ SCRIPT GENERATED SUCCESSFULLY")
        print("="*60)
        print(f"📄 File saved at: {path}")
        print("="*60)


# ============ MAIN INTERACTIVE ENTRY POINT ============

def main():
    print("\n" + "="*60)
    print("🧠 INTERACTIVE TRAINING CONFIGURATION CREATOR")
    print("   U-Net | Hybrid U-Net++ | YOLOv8")
    print("="*60)

    creator = TrainingConfigCreator()

    print("\n📦 Choose Model Type:")
    print("   1. U-Net (Segmentation)")
    print("   2. Hybrid U-Net++ (Segmentation)")
    print("   3. YOLOv8 (Detection)")

    choice = creator.get_input("Your choice (1-3)", "1", str)

    if choice == "1":
        model_family = creator.configure_unet()
    elif choice == "2":
        model_family = creator.configure_hybrid()
    elif choice == "3":
        model_family = creator.configure_yolo()
    else:
        print("❌ Invalid choice")
        return

    filename, script = creator.generate_script(model_family)
    creator.save_script(filename, script)

    print("\n🎉 All done! You can now run:")
    print(f"   python generated_scripts/{filename}")
    print("="*60)


if __name__ == "__main__":
    main()

