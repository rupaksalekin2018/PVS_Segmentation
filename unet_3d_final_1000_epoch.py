#!python -c "import monai" || pip install -q "monai-weekly[gdown, nibabel, tqdm, ignite]"
#!python -c "import matplotlib" || pip install -q matplotlib
#%matplotlib inline

#!pip install segmentation_models_pytorch_3d

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import nibabel as nib
from tqdm import tqdm

# MONAI imports
from monai.data import DataLoader, Dataset, decollate_batch
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    AsDiscrete, MapTransform, RandAffined,
    RandAdjustContrastd, 
    RandGaussianNoised,
    RandCoarseDropoutd
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# Import the 3D segmentation models library
import segmentation_models_pytorch_3d as smp

# --- CONFIGURATION ---
class Config:
    DATA_DIR = "/home/ubuntu/FMRI_Perivascular_Space_Dataset/ds005595-1.0.0"
    OUTPUT_DIR = "outputs_3d"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 123  # Using a smaller, safer seed value
    IMG_PATCH_SIZE = (96, 96, 96)
    BATCH_SIZE = 2
    EPOCHS = 1000  # Increased to 1000
    LR = 1e-3  # Increased learning rate from 1e-4
    EARLY_STOPPING_PATIENCE = 100  # Increased to 100
    A_MIN, A_MAX = -1000.0, 3000.0  # Updated intensity range for T2w images
    B_MIN, B_MAX = 0.0, 1.0
    PIX_DIM = (1.5, 1.5, 1.5)

cfg = Config()

def set_seed(seed_value):
    """Set seeds for reproducibility"""
    # Use a simple, safe seed value
    seed_value = int(seed_value)
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set MONAI's determinism
    set_determinism(seed=seed_value)


class BinarizeMaskd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0).float()
        return d


class BiasField3D(MapTransform):
    def __init__(self, keys, intensity=0.1, p=0.4):
        super().__init__(keys)
        self.intensity = intensity
        self.p = p

    def __call__(self, data):
        if np.random.rand() >= self.p:
            return data
        d = dict(data)
        for key in self.keys:
            img = d[key]
            _, h, w, z = img.shape
            y, x, v = np.ogrid[:h, :w, :z]
            center_y, center_x, center_z = h // 2, w // 2, z // 2
            dy, dx, dv = y - center_y, x - center_x, v - center_z
            max_dist = np.sqrt(center_x**2 + center_y**2 + center_z**2)
            if max_dist == 0: continue
            normalized_r = np.sqrt(dx**2 + dy**2 + dv**2) / max_dist
            coeff = np.random.uniform(-self.intensity, self.intensity)
            bias = 1.0 + coeff * (normalized_r ** 2)
            
            # Apply bias field and ensure tensor compatibility
            bias_tensor = torch.from_numpy(bias).to(img.device, dtype=img.dtype)
            augmented_img = img * bias_tensor
            d[key] = torch.clamp(augmented_img, img.min(), img.max())
        return d


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate comprehensive segmentation metrics for a single sample"""
    # Convert to binary
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate components
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # Avoid division by zero
    eps = 1e-8
    
    # Dice Score (F1)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    
    # IoU (Jaccard Index)
    iou = tp / (tp + fp + fn + eps)
    
    # Precision
    precision = tp / (tp + fp + eps)
    
    # Recall (Sensitivity)
    recall = tp / (tp + fn + eps)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def get_file_paths(data_dir):
    # Support both .nii and .nii.gz files
    image_paths = sorted(glob.glob(os.path.join(data_dir, "sub-*/anat/*_T2w.nii*")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "derivatives/mask/sub-*/sub-*_desc-mask_PVS.nii*")))
    
    # Filter out any .json files that might be caught by the wildcard
    image_paths = [p for p in image_paths if p.endswith('.nii') or p.endswith('.nii.gz')]
    mask_paths = [p for p in mask_paths if p.endswith('.nii') or p.endswith('.nii.gz')]
    
    image_dict = {os.path.basename(p).split('_')[0]: p for p in image_paths}
    mask_dict = {os.path.basename(p).split('_')[0]: p for p in mask_paths}
    common_subjects = sorted(list(set(image_dict.keys()) & set(mask_dict.keys())))
    initial_data_dicts = [{"image": image_dict[sub], "label": mask_dict[sub]} for sub in common_subjects]
    print(f"Found {len(initial_data_dicts)} total matching image-mask pairs.")

    final_data_dicts = []
    for data_dict in tqdm(initial_data_dicts, desc="Checking for non-empty masks"):
        try:
            mask_data = nib.load(data_dict["label"]).get_fdata()
            if np.any(mask_data): final_data_dicts.append(data_dict)
        except Exception as e:
            print(f"Warning: Could not process {data_dict['label']}. Skipping. Error: {e}")

    print(f"Found {len(final_data_dicts)} subjects with non-empty masks.")
    print(f"Using all {len(final_data_dicts)} subjects for training.")
    return final_data_dicts


def get_transforms():
    """Defines the MONAI transformation pipelines for training and validation."""
    
    # Basic transforms without random components
    basic_train_transforms = [
        LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]),
        BinarizeMaskd(keys=["label"]), 
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=cfg.PIX_DIM, mode=("bilinear", "nearest")),
        # Add intensity normalization
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=cfg.A_MIN, 
            a_max=cfg.A_MAX,
            b_min=cfg.B_MIN, 
            b_max=cfg.B_MAX,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ]
    
    # Random transforms with reduced augmentation initially
    random_transforms = [
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=cfg.IMG_PATCH_SIZE,
            pos=2,  # Increased positive samples
            neg=1,  # Reduced negative samples
            num_samples=4, image_key="image", image_threshold=0
        ),
        RandAffined(
            keys=["image", "label"], prob=0.3,  # Reduced probability
            rotate_range=(np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)),  # Reduced rotation
            scale_range=(0.95, 1.05),
            translate_range=[int(s * 0.03) for s in cfg.IMG_PATCH_SIZE],  # Reduced translation
            mode=("bilinear", "nearest")
        ),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.9, 1.1)),  # Reduced contrast adjustment
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.005),  # Reduced noise
    ]
    
    # Create train transform with controlled random state
    train_transforms_list = basic_train_transforms + random_transforms
    train_transform = Compose(train_transforms_list)
    
    # Manually set random state for each random transform
    for t in train_transform.transforms:
        if hasattr(t, 'set_random_state'):
            try:
                t.set_random_state(seed=cfg.SEED)
            except:
                pass  # Skip if there's an issue with this specific transform

    val_transform = Compose([
        LoadImaged(keys=["image", "label"]), 
        EnsureChannelFirstd(keys=["image", "label"]),
        BinarizeMaskd(keys=["label"]), 
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=cfg.PIX_DIM, mode=("bilinear", "nearest")),
        # Add intensity normalization for validation too
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=cfg.A_MIN, 
            a_max=cfg.A_MAX,
            b_min=cfg.B_MIN, 
            b_max=cfg.B_MAX,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ])
    
    return train_transform, val_transform


def get_dataloaders():
    data_dicts = get_file_paths(cfg.DATA_DIR)
    if not data_dicts:
        raise ValueError("No non-empty mask data found.")
    train_val_files, test_files = train_test_split(data_dicts, test_size=0.05, random_state=cfg.SEED)
    train_files, val_files = train_test_split(train_val_files, test_size=0.0526, random_state=cfg.SEED)
    train_transform, val_transform = get_transforms()
    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_transform)
    test_ds = Dataset(data=test_files, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Data split: {len(train_files)} Train || {len(val_files)} Validation || {len(test_files)} Test")
    return train_loader, val_loader, test_loader


def get_model():
    """Instantiates a UNet++ model with EfficientNet-B7 encoder."""
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=1, 
        classes=1,
    )
    return model.to(cfg.DEVICE)


def dice_bce_loss(preds, targets, smooth=1.0, bce_weight=0.5):
    """Combined Dice and BCE loss with adjustable weighting"""
    preds_sig = torch.sigmoid(preds)
    
    # BCE Loss
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    
    # Dice Loss
    preds_flat = preds_sig.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    
    dice_coeff = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    dice_loss = 1 - dice_coeff
    
    # Weighted combination
    total_loss = bce_weight * bce + (1 - bce_weight) * dice_loss
    
    return total_loss


def train_model(model, train_loader, val_loader):
    loss_function = dice_bce_loss
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)  # Reduced weight decay
    # Fixed: Removed verbose parameter which is not supported in newer PyTorch versions
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)

    best_val_dice = -1.0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(threshold=0.5)

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        
        for batch in train_pbar:
            images, masks = batch["image"].to(cfg.DEVICE), batch["label"].to(cfg.DEVICE)
            
            # Check if masks contain any positive samples
            if masks.sum() == 0:
                print(f"Warning: Batch contains no positive masks")
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate training dice for monitoring
            with torch.no_grad():
                train_pred = torch.sigmoid(outputs) > 0.5
                if masks.sum() > 0:  # Only calculate dice if there are positive masks
                    intersection = (train_pred * masks).sum()
                    dice = (2. * intersection) / (train_pred.sum() + masks.sum() + 1e-8)
                    train_dice += dice.item()
            
            train_pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_samples_with_masks = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch["image"].to(cfg.DEVICE), batch["label"].to(cfg.DEVICE)
                outputs = sliding_window_inference(images, cfg.IMG_PATCH_SIZE, 4, model, overlap=0.5)
                loss = loss_function(outputs, masks)
                val_loss += loss.item()
                
                # Only calculate dice for samples with positive masks
                if masks.sum() > 0:
                    val_samples_with_masks += 1
                    binary_outputs = [post_pred(torch.sigmoid(i)) for i in decollate_batch(outputs)]
                    dice_metric(y_pred=binary_outputs, y=decollate_batch(masks))

        avg_val_loss = val_loss / len(val_loader)
        
        # Only aggregate dice if there were samples with masks
        if val_samples_with_masks > 0:
            mean_val_dice = dice_metric.aggregate().item()
        else:
            mean_val_dice = 0.0
            print("Warning: No validation samples with positive masks!")
        
        dice_metric.reset()
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(mean_val_dice)
        scheduler.step(avg_val_loss)
        
        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {mean_val_dice:.4f} | LR: {current_lr:.6f}")

        if mean_val_dice > best_val_dice:
            print(f"Val Dice improved to {mean_val_dice:.4f}. Saving best model.")
            best_val_dice = mean_val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"))
        else:
            epochs_no_improve += 1
            print(f"Val Dice did not improve for {epochs_no_improve} epochs. Best: {best_val_dice:.4f}")
        
        # Additional debugging info every 10 epochs (reduced frequency for longer training)
        if (epoch + 1) % 10 == 0:
            print(f"Debug Info - Samples with masks in validation: {val_samples_with_masks}/{len(val_loader)}")
        
        if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    with open(os.path.join(cfg.LOG_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print("Training complete.")


def evaluate_and_visualize(model, loader, max_visualizations=10):
    """
    Evaluate the model and save comprehensive metrics and high-resolution visualizations.
    """
    model.load_state_dict(torch.load(os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")))
    model.eval()
    
    # Storage for all metrics
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    plot_data = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating and Visualizing Test Set")):
            image, label = batch["image"].to(cfg.DEVICE), batch["label"].to(cfg.DEVICE)
            prediction = sliding_window_inference(image, cfg.IMG_PATCH_SIZE, 4, model, overlap=0.5)
            prediction = torch.sigmoid(prediction)
            
            # Skip samples with no positive labels for metrics calculation
            if torch.sum(label) > 0:
                # Calculate comprehensive metrics for this sample
                sample_metrics = calculate_metrics(prediction, label)
                
                # Store metrics
                for metric_name in all_metrics.keys():
                    all_metrics[metric_name].append(sample_metrics[metric_name])

            # Collect visualization data
            if i < max_visualizations and torch.sum(label) > 0:
                img_np, lbl_np, pred_np = image.cpu().numpy()[0, 0], label.cpu().numpy()[0, 0], prediction.cpu().numpy()[0, 0]
                slice_idx = img_np.shape[2] // 2
                
                plot_data.append({
                    "subject_id": f"test_subject_{len(plot_data):02d}",
                    "slice_idx": slice_idx,
                    "img_slice": img_np[:, :, slice_idx],
                    "lbl_slice": lbl_np[:, :, slice_idx],
                    "pred_slice": (pred_np[:, :, slice_idx] > 0.5).astype(np.float32)
                })

    # Create visualizations
    if plot_data:
        num_subjects = len(plot_data)
        fig, axes = plt.subplots(num_subjects, 3, figsize=(15, 5 * num_subjects))
        if num_subjects == 1: 
            axes = axes.reshape(1, -1)

        for i, data in enumerate(plot_data):
            img_slice = data["img_slice"]
            lbl_slice = data["lbl_slice"]
            pred_slice = data["pred_slice"]

            # Normalize image for better visualization
            img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

            # Panel 1: Original Image
            axes[i, 0].imshow(img_normalized.T, cmap="gray", origin="lower")
            axes[i, 0].set_title(f"{data['subject_id']} - Image (Slice {data['slice_idx']})")
            axes[i, 0].axis('off')

            # Panel 2: Ground Truth Overlay
            axes[i, 1].imshow(img_normalized.T, cmap="gray", origin="lower", alpha=1.0)
            # Create colored overlay for ground truth mask
            mask_overlay = np.zeros((*lbl_slice.T.shape, 4))  # RGBA
            mask_overlay[..., 0] = 1.0  # Red channel
            mask_overlay[..., 3] = lbl_slice.T * 0.5  # Alpha channel based on mask
            axes[i, 1].imshow(mask_overlay, origin="lower")
            axes[i, 1].set_title("Ground Truth Overlay (Red)")
            axes[i, 1].axis('off')

            # Panel 3: Predicted Mask Overlay
            axes[i, 2].imshow(img_normalized.T, cmap="gray", origin="lower", alpha=1.0)
            # Create colored overlay for predicted mask
            pred_overlay = np.zeros((*pred_slice.T.shape, 4))  # RGBA
            pred_overlay[..., 1] = 1.0  # Green channel
            pred_overlay[..., 3] = pred_slice.T * 0.5  # Alpha channel based on prediction
            axes[i, 2].imshow(pred_overlay, origin="lower")
            axes[i, 2].set_title("Predicted Mask Overlay (Green)")
            axes[i, 2].axis('off')
        
        plt.tight_layout(pad=3.0)
        grid_save_path = os.path.join(cfg.OUTPUT_DIR, "test_set_visualization_grid_300dpi.jpg")
        plt.savefig(grid_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved high-resolution visualization grid (300 DPI) to: {grid_save_path}")

    # Calculate summary statistics
    results = {}
    for metric_name, values in all_metrics.items():
        if len(values) > 0:
            results[f"{metric_name}_mean"] = np.mean(values)
            results[f"{metric_name}_std"] = np.std(values)
        else:
            results[f"{metric_name}_mean"] = 0.0
            results[f"{metric_name}_std"] = 0.0

    # Print results in requested format
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    print(f"dice_mean{results['dice_mean']}")
    print(f"dice_std{results['dice_std']}")
    print(f"iou_mean{results['iou_mean']}")
    print(f"iou_std{results['iou_std']}")
    print(f"precision_mean{results['precision_mean']}")
    print(f"precision_std{results['precision_std']}")
    print(f"recall_mean{results['recall_mean']}")
    print(f"recall_std{results['recall_std']}")

    # Save comprehensive results
    with open(os.path.join(cfg.LOG_DIR, "comprehensive_test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(cfg.LOG_DIR, "individual_sample_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nResults saved to {cfg.LOG_DIR}/")

        
if __name__ == "__main__":
    set_seed(cfg.SEED)
    print(f"Using device: {cfg.DEVICE}")
    
    train_loader, val_loader, test_loader = get_dataloaders()
    
    model = get_model()
    
    if len(train_loader) > 0 and len(val_loader) > 0:
        print("Starting supervised training...")
        train_model(model, train_loader, val_loader)
        
        print("\nStarting comprehensive evaluation and visualization on the test set...")
        if len(test_loader) > 0:
            evaluate_and_visualize(model, test_loader)
        else:
            print("Skipping evaluation as the test loader is empty.")
    else:
        print("Cannot start training: The training or validation dataloader is empty.")

    print("\nScript finished. Check the 'outputs_3d' directory for results.")
