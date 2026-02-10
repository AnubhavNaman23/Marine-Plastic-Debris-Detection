"""
Comprehensive Marine Debris Segmentation Training

This script trains UNet, UNet++, and other segmentation models on combined datasets:
- MARIDA (Multi-class marine debris)
- FloatingObjects (Binary debris detection)
- RefinedFloatingObjects (High-quality labels)
- S2Ships (Hard negatives - ships)
- PLP (Plastic Litter Project validation)

Usage:
    python train_segmentation.py --data-path "C:/path/to/MarineDebrisData" --model unet++ --epochs 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    print("Warning: segmentation_models_pytorch not installed. Install with: pip install segmentation-models-pytorch")
    SMP_AVAILABLE = False

import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# CONFIGURATION
# ============================================================

# Sentinel-2 bands (12 bands for L2A)
BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
NUM_BANDS = 12

# Class mapping for MARIDA
MARIDA_CLASSES = {
    1: 'Marine Debris',
    2: 'Dense Sargassum',
    3: 'Sparse Sargassum', 
    4: 'Natural Organic Material',
    5: 'Ship',
    6: 'Clouds',
    7: 'Marine Water',
    8: 'Sediment-Laden Water',
    9: 'Foam',
    10: 'Turbid Water',
    11: 'Shallow Water',
    12: 'Waves',
    13: 'Cloud Shadows',
    14: 'Wakes',
    15: 'Mixed Water'
}


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transforms(image_size=128):
    """Training augmentations."""
    return A.Compose([
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussNoise(var_limit=(0.0001, 0.001), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ])

def get_val_transforms(image_size=128):
    """Validation transforms (center crop only)."""
    return A.Compose([
        A.CenterCrop(image_size, image_size),
    ])


# ============================================================
# SPECTRAL INDICES
# ============================================================

def compute_spectral_indices(bands_dict):
    """
    Compute spectral indices for marine debris detection.
    
    Args:
        bands_dict: Dictionary with band names as keys and arrays as values
    
    Returns:
        Dictionary of computed indices
    """
    indices = {}
    eps = 1e-6
    
    B2 = bands_dict.get('B02', bands_dict.get('B2', None))
    B3 = bands_dict.get('B03', bands_dict.get('B3', None))
    B4 = bands_dict.get('B04', bands_dict.get('B4', None))
    B5 = bands_dict.get('B05', bands_dict.get('B5', None))
    B6 = bands_dict.get('B06', bands_dict.get('B6', None))
    B8 = bands_dict.get('B08', bands_dict.get('B8', None))
    B11 = bands_dict.get('B11', None)
    B12 = bands_dict.get('B12', None)
    
    if B8 is not None and B4 is not None:
        # NDVI - Normalized Difference Vegetation Index
        indices['NDVI'] = np.nan_to_num((B8 - B4) / (B8 + B4 + eps), nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Plastic Index
        indices['PI'] = np.nan_to_num(B8 / (B8 + B4 + eps), nan=0.0, posinf=1.0, neginf=-1.0)
    
    if B3 is not None and B8 is not None:
        # NDWI - Normalized Difference Water Index
        indices['NDWI'] = np.nan_to_num((B3 - B8) / (B3 + B8 + eps), nan=0.0, posinf=1.0, neginf=-1.0)
    
    if B8 is not None and B11 is not None:
        # NDMI - Normalized Difference Moisture Index
        indices['NDMI'] = np.nan_to_num((B8 - B11) / (B8 + B11 + eps), nan=0.0, posinf=1.0, neginf=-1.0)
    
    if B6 is not None and B8 is not None and B11 is not None:
        # FDI - Floating Debris Index
        lambda_nir = 833
        lambda_re2 = 665
        lambda_swir1 = 1610.4
        indices['FDI'] = np.nan_to_num(B8 - (B6 + (B11 - B6) * ((lambda_nir - lambda_re2) / (lambda_swir1 - lambda_re2)) * 10), nan=0.0, posinf=1.0, neginf=-1.0)
    
    return indices


# ============================================================
# DATASETS
# ============================================================

class MARIDASegmentationDataset(Dataset):
    """MARIDA dataset for marine debris segmentation.
    
    Handles the official MARIDA structure where each scene directory
    (e.g. patches/S2_1-12-19_48MYU/) contains multi-band TIF files
    (S2_1-12-19_48MYU_0.tif) and class-label masks (*_cl.tif).
    
    Split files list entries like '1-12-19_48MYU_0' which map to:
      - Scene dir:  patches/S2_1-12-19_48MYU/
      - Image file: S2_1-12-19_48MYU_0.tif   (multi-band)
      - Mask file:  S2_1-12-19_48MYU_0_cl.tif (class labels)
    """
    
    def __init__(self, root, split='train', transform=None, binary=True, image_size=128):
        self.root = Path(root)
        self.patches_dir = self.root / 'patches'
        self.split = split
        self.transform = transform
        self.binary = binary
        self.image_size = image_size
        self.has_debris = []
        
        # Load split file
        splits_dir = self.root / 'splits'
        if splits_dir.exists():
            split_file = splits_dir / f'{split}_X.txt'
            if split_file.exists():
                with open(split_file) as f:
                    self.samples = [line.strip() for line in f.readlines() if line.strip()]
            else:
                self.samples = self._find_samples()
        else:
            self.samples = self._find_samples()
            
        filtered_samples = []
        for sample in self.samples:
            scene_dir, _, mask_path = self._sample_to_paths(sample)
            has_deb = False
            mask_sum = 0
            if mask_path.exists():
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                if self.binary:
                    mask = (mask == 1)
                mask_sum = mask.sum()
                has_deb = mask_sum > 0
                
            if self.split == 'train':
                if mask_sum > 10 or np.random.rand() > 0.5:
                    filtered_samples.append(sample)
                    self.has_debris.append(has_deb)
            else:
                filtered_samples.append(sample)
                self.has_debris.append(has_deb)
                
        self.samples = filtered_samples
        
        print(f"MARIDA {split}: {len(self.samples)} samples")
    
    def _sample_to_paths(self, sample_name):
        """Convert a sample name (e.g. '1-12-19_48MYU_0') to file paths.
        
        Returns (scene_dir, image_path, mask_path).
        """
        full_name = f'S2_{sample_name}'
        # Scene dir = everything before the last underscore + digit(s)
        # e.g. 'S2_1-12-19_48MYU_0' -> scene 'S2_1-12-19_48MYU'
        last_underscore = full_name.rfind('_')
        scene_name = full_name[:last_underscore]
        
        scene_dir = self.patches_dir / scene_name
        image_path = scene_dir / f'{full_name}.tif'
        mask_path = scene_dir / f'{full_name}_cl.tif'
        return scene_dir, image_path, mask_path
    
    def _find_samples(self):
        """Scan patch directories for available sub-patches (fallback when no split file)."""
        samples = []
        if self.patches_dir.exists():
            for patch_dir in self.patches_dir.iterdir():
                if patch_dir.is_dir() and patch_dir.name.startswith('S2_'):
                    for tif_file in patch_dir.glob('*.tif'):
                        name = tif_file.stem
                        # Skip mask and confidence files
                        if name.endswith('_cl') or name.endswith('_conf'):
                            continue
                        # Strip the 'S2_' prefix to match split file format
                        sample_id = name[3:]  # Remove 'S2_'
                        samples.append(sample_id)
        return sorted(samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        scene_dir, image_path, mask_path = self._sample_to_paths(sample_name)
        
        # Load multi-band image
        if image_path.exists():
            with rasterio.open(image_path) as src:
                image = src.read().astype(np.float32)
                if image.max() > 100:
                    image = image * 1e-4  # Scale DN to 0-1
        else:
            # Fallback: create empty image
            image = np.zeros((12, self.image_size, self.image_size), dtype=np.float32)
        
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)
        
        # Pad/truncate to exactly 12 bands
        n_bands = image.shape[0]
        if n_bands < 12:
            padding = np.zeros((12 - n_bands, image.shape[1], image.shape[2]), dtype=np.float32)
            image = np.concatenate([image, padding], axis=0)
        elif n_bands > 12:
            image = image[:12]
        
        # Load mask
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int64)
        else:
            mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.int64)
        
        # Binary classification: debris (class 1) vs non-debris
        if self.binary:
            mask = (mask == 1).astype(np.float32)  # Only Marine Debris class
        
        # Transpose C,H,W -> H,W,C for processing
        image = image.transpose(1, 2, 0)
        
        # Ensure image and mask have matching spatial dims, then resize to image_size
        from skimage.transform import resize as ski_resize
        img_h, img_w = image.shape[0], image.shape[1]
        mask_h, mask_w = mask.shape[0], mask.shape[1]
        
        # If mask and image sizes differ, resize mask to match image
        if (mask_h, mask_w) != (img_h, img_w):
            mask = ski_resize(mask, (img_h, img_w), preserve_range=True, order=0).astype(mask.dtype)
        
        # Resize both to target image_size
        if img_h != self.image_size or img_w != self.image_size:
            image = ski_resize(image, (self.image_size, self.image_size, image.shape[2]),
                               preserve_range=True).astype(np.float32)
            mask = ski_resize(mask, (self.image_size, self.image_size),
                              preserve_range=True, order=0).astype(mask.dtype)
        
        # Apply transforms (image and mask now guaranteed to have matching shapes)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to torch tensors
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))  # C, H, W
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask)
        
        return image, mask, sample_name



class FloatingObjectsDataset(Dataset):
    """FloatingObjects dataset for binary debris detection."""
    
    def __init__(self, root, split='train', transform=None, image_size=128):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.has_debris = []
        
        # Find scenes
        self.scenes = []
        scenes_dir = self.root / 'scenes'
        if scenes_dir.exists():
            for scene_file in scenes_dir.glob('*.tif'):
                self.scenes.append(scene_file)
        
        # Also check root directly
        for scene_file in self.root.glob('*.tif'):
            if 'mask' not in scene_file.name.lower():
                self.scenes.append(scene_file)
        
        # Generate patches from scenes
        self.patches = []
        for scene in self.scenes:
            self._extract_patches(scene)
        
        print(f"FloatingObjects {split}: {len(self.patches)} patches from {len(self.scenes)} scenes")
    
    def _extract_patches(self, scene_path):
        """Extract patch locations from a scene."""
        try:
            mask_path = str(scene_path).replace('.tif', '_mask.tif')
            if os.path.exists(mask_path):
                with rasterio.open(mask_path) as src_mask:
                    mask_full = src_mask.read(1)
            else:
                mask_full = None

            with rasterio.open(scene_path) as src:
                h, w = src.height, src.width
                
                # Grid sampling
                step = self.image_size
                for y in range(0, h - self.image_size, step):
                    for x in range(0, w - self.image_size, step):
                        has_deb = False
                        if mask_full is not None:
                            patch_mask = mask_full[y:y+self.image_size, x:x+self.image_size]
                            has_deb = (patch_mask > 0).sum() > 0
                        self.patches.append((scene_path, x, y))
                        self.has_debris.append(has_deb)
        except Exception as e:
            print(f"Warning: Could not process {scene_path}: {e}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        scene_path, x, y = self.patches[idx]
        
        # Read image patch
        with rasterio.open(scene_path) as src:
            window = Window(x, y, self.image_size, self.image_size)
            image = src.read(window=window).astype(np.float32)
            if image.max() > 100:
                image = image * 1e-4
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0.0, 1.0)
        
        # Try to find mask
        mask_path = str(scene_path).replace('.tif', '_mask.tif')
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                mask = src.read(1, window=window).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        mask = (mask > 0).astype(np.float32)
        
        # Transpose to H, W, C for albumentations
        image = image.transpose(1, 2, 0)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']   # ToTensorV2 returns (C,H,W) tensor
            mask = transformed['mask']
            # Ensure correct types (ToTensorV2 already gives tensors)
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask.astype(np.float32))
        else:
            # No transform applied — manually convert HWC numpy → CHW tensor
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
            mask = torch.from_numpy(mask.astype(np.float32))
        
        return image, mask, f"{scene_path.stem}_{x}_{y}"


class RefinedFloatingObjectsDataset(Dataset):
    """Refined FloatingObjects with high-quality labels."""
    
    def __init__(self, root, split='train', transform=None, image_size=128):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.has_debris = []
        
        self.scenes = []
        
        # Look for regions
        for tif_file in self.root.glob('*.tif'):
            if '_qualitative' not in tif_file.name and '_mask' not in tif_file.name and '_label' not in tif_file.name:
                self.scenes.append(tif_file)
        
        self.patches = []
        for scene in self.scenes:
            self._extract_patches(scene)
            
        print(f"RefinedFloatingObjects {split}: {len(self.patches)} patches from {len(self.scenes)} scenes")
        
    def _extract_patches(self, scene_path):
        try:
            mask_path = str(scene_path).replace('.tif', '_qualitative_poly.tif')
            if not os.path.exists(mask_path):
                mask_path = str(scene_path).replace('.tif', '_mask.tif')
                if not os.path.exists(mask_path):
                    mask_path = str(scene_path).replace('.tif', '_label.tif')

            if os.path.exists(mask_path):
                with rasterio.open(mask_path) as src_mask:
                    mask_full = src_mask.read(1)
            else:
                mask_full = None

            with rasterio.open(scene_path) as src:
                h, w = src.height, src.width
                step = self.image_size
                for y in range(0, h - self.image_size, step):
                    for x in range(0, w - self.image_size, step):
                        mask_sum = 0
                        if mask_full is not None:
                            patch_mask = mask_full[y:y+self.image_size, x:x+self.image_size]
                            mask_sum = (patch_mask > 0).sum()
                            
                        has_deb = mask_sum > 0
                        
                        if self.split == 'train':
                            if mask_sum > 10 or np.random.rand() > 0.5:
                                self.patches.append((scene_path, x, y))
                                self.has_debris.append(has_deb)
                        else:
                            self.patches.append((scene_path, x, y))
                            self.has_debris.append(has_deb)
        except Exception as e:
            print(f"Warning: Could not process {scene_path}: {e}")

    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        scene_path, x, y = self.patches[idx]
        
        # Read image patch
        with rasterio.open(scene_path) as src:
            window = Window(x, y, self.image_size, self.image_size)
            image = src.read(window=window).astype(np.float32)
            if image.max() > 100:
                image = image * 1e-4
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0.0, 1.0)
            
            # Ensure we have at least 12 bands
            if image.shape[0] < 12:
                padding = np.zeros((12 - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
                image = np.concatenate([image, padding], axis=0)
            elif image.shape[0] > 12:
                image = image[:12]
        
        # Try to find mask
        mask_path = str(scene_path).replace('.tif', '_qualitative_poly.tif')
        if not os.path.exists(mask_path):
            mask_path = str(scene_path).replace('.tif', '_mask.tif')
            if not os.path.exists(mask_path):
                mask_path = str(scene_path).replace('.tif', '_label.tif')
        
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                mask = src.read(1, window=window).astype(np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        mask = (mask > 0).astype(np.float32)
        
        # Transpose for albumentations
        image_hwc = image.transpose(1, 2, 0)
        
        if self.transform:
            transformed = self.transform(image=image_hwc, mask=mask)
            image = transformed['image']   # ToTensorV2 returns (C,H,W) tensor
            mask = transformed['mask']
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask.astype(np.float32))
        else:
            image = torch.from_numpy(image_hwc.transpose(2, 0, 1).astype(np.float32))
            mask = torch.from_numpy(mask.astype(np.float32))
        
        return image, mask, f"{scene_path.stem}_{x}_{y}"


class S2ShipsDataset(Dataset):
    """S2Ships dataset for hard negative mining (ships are not debris)."""
    
    def __init__(self, root, transform=None, image_size=128):
        self.root = Path(root)
        self.transform = transform
        self.image_size = image_size
        
        self.samples = []
        self.has_debris = []
        for npy_file in self.root.rglob('*.npy'):
            self.samples.append(npy_file)
            self.has_debris.append(False)
        
        print(f"S2Ships: {len(self.samples)} samples (hard negatives)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npy_path = self.samples[idx]
        
        data_dict = np.load(npy_path, allow_pickle=True).item()
        image = data_dict['data'].astype(np.float32)
        if image.max() > 100:
            image = image * 1e-4
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)
        
        h, w = image.shape[0], image.shape[1]
        
        # Center crop or resize
        if h >= self.image_size and w >= self.image_size:
            y = (h - self.image_size) // 2
            x = (w - self.image_size) // 2
            image = image[y:y+self.image_size, x:x+self.image_size, :]
        else:
            # Resize
            from skimage.transform import resize
            image = resize(image, (self.image_size, self.image_size, 12), preserve_range=True)
            
        # Hard negative -> mask is 0
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Transpose HWC -> CHW
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        
        return image, mask, npy_path.stem


class PLPDataset(Dataset):
    """Plastic Litter Project dataset for validation."""
    
    def __init__(self, root, year=2021, image_size=128):
        self.root = Path(root)
        self.year = year
        self.image_size = image_size
        
        self.samples = []
        self.has_debris = []
        year_dir = self.root / str(year)
        if year_dir.exists():
            for tif_file in year_dir.rglob('*.tif'):
                if 'mask' not in tif_file.name.lower():
                    self.samples.append(tif_file)
                    self.has_debris.append(False)
        
        print(f"PLP {year}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            if image.max() > 100:
                image = image * 1e-4
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0.0, 1.0)
        
        if image.shape[0] < 12:
            padding = np.zeros((12 - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
            image = np.concatenate([image, padding], axis=0)
        elif image.shape[0] > 12:
            image = image[:12]
        
        # Center crop
        h, w = image.shape[1], image.shape[2]
        y = max(0, (h - self.image_size) // 2)
        x = max(0, (w - self.image_size) // 2)
        image = image[:, y:y+self.image_size, x:x+self.image_size]
        
        # Pad if needed
        if image.shape[1] < self.image_size or image.shape[2] < self.image_size:
            pad_h = self.image_size - image.shape[1]
            pad_w = self.image_size - image.shape[2]
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        
        # PLP has plastic presence labels (used for validation)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        
        return image, mask, image_path.stem


# ============================================================
# MODELS
# ============================================================

class ConvBlock(nn.Module):
    """Double convolution block for UNet."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Custom UNet for satellite imagery."""
    
    def __init__(self, in_channels=12, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder.append(ConvBlock(feature * 2, feature))
        
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


class AttentionUNet(nn.Module):
    """UNet with attention gates."""
    
    def __init__(self, in_channels=12, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder with attention
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.attention.append(self._attention_gate(feature, feature, feature // 2))
            self.decoder.append(ConvBlock(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def _attention_gate(self, g_ch, x_ch, inter_ch):
        return nn.Sequential(
            nn.Conv2d(g_ch + x_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        skip_connections = []
        
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            # Attention
            attn_idx = idx // 2
            attn = self.attention[attn_idx](torch.cat([x, skip], dim=1))
            skip = skip * attn
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


def get_segmentation_model(model_name, in_channels=12, out_channels=1, encoder_name='resnet34'):
    """
    Get segmentation model by name.
    
    Supported models:
    - unet: Custom UNet
    - unet_attention: UNet with attention gates
    - unet++: UNet++ (nested UNet) from SMP
    - deeplabv3+: DeepLabV3+ from SMP
    - fpn: Feature Pyramid Network from SMP
    - manet: Multi-scale Attention Network from SMP
    """
    model_name = model_name.lower().replace(' ', '').replace('_', '')
    
    if model_name == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == 'unetattention':
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name in ['unet++', 'unetplusplus', 'unetpp']:
        if SMP_AVAILABLE:
            return smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,  # No pretrained for 12 channels
                in_channels=in_channels,
                classes=out_channels,
                activation=None
            )
        else:
            print("SMP not available, using custom UNet")
            return UNet(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name in ['deeplabv3+', 'deeplabv3plus', 'deeplabv3']:
        if SMP_AVAILABLE:
            return smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=out_channels,
                activation=None
            )
        else:
            return UNet(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == 'fpn':
        if SMP_AVAILABLE:
            return smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=out_channels,
                activation=None
            )
        else:
            return UNet(in_channels=in_channels, out_channels=out_channels)
    
    elif model_name == 'manet':
        if SMP_AVAILABLE:
            return smp.MAnet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=out_channels,
                activation=None
            )
        else:
            return UNet(in_channels=in_channels, out_channels=out_channels)
    
    else:
        print(f"Unknown model: {model_name}, using UNet")
        return UNet(in_channels=in_channels, out_channels=out_channels)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        pred = torch.clamp(pred, min=-20.0, max=20.0)
        if torch.isnan(pred).any():
            print("NaN detected in predictions after nan_to_num!")
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


# ============================================================
# LIGHTNING MODULE
# ============================================================

class SegmentationLightningModule(pl.LightningModule):
    """PyTorch Lightning module for segmentation training."""
    
    def __init__(
        self,
        model_name='unet++',
        in_channels=12,
        learning_rate=1e-5,
        weight_decay=1e-4,
        pos_weight=2.0,
        encoder_name='resnet34'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = get_segmentation_model(
            model_name, 
            in_channels=in_channels,
            encoder_name=encoder_name
        )
        
        self.criterion = CombinedLoss(pos_weight=pos_weight)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Optimal threshold (updated during validation)
        self.register_buffer('threshold', torch.tensor(0.5))
        
        # Metrics storage
        self.val_outputs = []
        self.test_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def predict_mask(self, x):
        """Get binary prediction mask."""
        logits = self(x)
        probs = torch.sigmoid(logits)
        return (probs > self.threshold).long()
    
    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        logits = self(images)
        loss = self.criterion(logits.squeeze(1), masks)
        
        if torch.isnan(loss):
            print("Warning: NaN loss detected, returning None to skip batch")
            return None
            
        # Metrics
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
        acc = (preds == masks).float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks, ids = batch
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        logits = self(images)
        loss = self.criterion(logits.squeeze(1), masks)
        
        if torch.isnan(loss):
            print("Warning: NaN validation loss detected")
            return None
            
        probs = torch.sigmoid(logits.squeeze(1))
        
        self.val_outputs.append({
            'loss': loss.item(),
            'probs': probs.cpu().numpy(),
            'masks': masks.cpu().numpy(),
            'ids': ids
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        all_probs = np.concatenate([o['probs'].flatten() for o in self.val_outputs])
        all_masks = np.concatenate([o['masks'].flatten() for o in self.val_outputs])
        all_masks = (all_masks > 0.5).astype(int)  # Ensure discrete binary labels
        avg_loss = np.mean([o['loss'] for o in self.val_outputs])
        
        # Find optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.8, 0.05):
            preds = (all_probs > thresh).astype(int)
            f1 = f1_score(all_masks, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        self.threshold = torch.tensor(best_thresh)
        
        # Calculate metrics at optimal threshold
        preds = (all_probs > best_thresh).astype(int)
        
        precision = precision_score(all_masks, preds, zero_division=0)
        recall = recall_score(all_masks, preds, zero_division=0)
        iou = jaccard_score(all_masks, preds, zero_division=0)
        
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/f1', best_f1, prog_bar=True)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/iou', iou)
        self.log('val/threshold', best_thresh)
        
        self.val_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        images, masks, ids = batch
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        logits = self(images)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=0.0)
        
        probs = torch.sigmoid(logits.squeeze(1))
        
        self.test_outputs.append({
            'probs': probs.cpu().numpy(),
            'masks': masks.cpu().numpy()
        })
    
    def on_test_epoch_end(self):
        all_probs = np.concatenate([o['probs'].flatten() for o in self.test_outputs])
        all_masks = np.concatenate([o['masks'].flatten() for o in self.test_outputs])
        all_masks = (all_masks > 0.5).astype(int)  # Ensure discrete binary labels        
        preds = (all_probs > self.threshold.item()).astype(int)
        
        f1 = f1_score(all_masks, preds, zero_division=0)
        precision = precision_score(all_masks, preds, zero_division=0)
        recall = recall_score(all_masks, preds, zero_division=0)
        iou = jaccard_score(all_masks, preds, zero_division=0)
        acc = (preds == all_masks).mean()
        
        self.log('test/accuracy', acc)
        self.log('test/f1', f1)
        self.log('test/precision', precision)
        self.log('test/recall', recall)
        self.log('test/iou', iou)
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  F1 Score:  {f1*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  IoU:       {iou*100:.2f}%")
        
        self.test_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/f1'
            }
        }


# ============================================================
# DATA MODULE
# ============================================================

class MarineDebrisDataModule(pl.LightningDataModule):
    """Combined data module for all marine debris datasets."""
    
    def __init__(
        self,
        data_root,
        image_size=128,
        batch_size=32,
        num_workers=4,
        include_s2ships=True,
        include_plp=True
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_s2ships = include_s2ships
        self.include_plp = include_plp
    
    def setup(self, stage=None):
        train_transform = get_train_transforms(self.image_size)
        val_transform = get_val_transforms(self.image_size)
        
        train_datasets = []
        val_datasets = []
        test_datasets = []
        
        # MARIDA
        marida_path = self.data_root / 'MARIDA'
        if marida_path.exists():
            train_datasets.append(MARIDASegmentationDataset(
                marida_path, split='train', transform=train_transform, image_size=self.image_size
            ))
            val_datasets.append(MARIDASegmentationDataset(
                marida_path, split='val', transform=val_transform, image_size=self.image_size
            ))
            test_datasets.append(MARIDASegmentationDataset(
                marida_path, split='test', transform=val_transform, image_size=self.image_size
            ))
        
        # FloatingObjects
        flobs_path = self.data_root / 'floatingobjects'
        if flobs_path.exists():
            flobs_ds = FloatingObjectsDataset(
                flobs_path, split='train', transform=train_transform, image_size=self.image_size
            )
            # Split into train/val
            n_val = len(flobs_ds) // 5
            n_train = len(flobs_ds) - n_val
            train_flobs, val_flobs = random_split(flobs_ds, [n_train, n_val])
            if len(train_flobs) > 5000:
                indices = np.random.choice(len(train_flobs), 5000, replace=False)
                train_flobs = torch.utils.data.Subset(train_flobs, indices)
                print(f"FloatingObjects: Limited training patches to 5000")
            train_datasets.append(train_flobs)
            val_datasets.append(val_flobs)
        
        # RefinedFloatingObjects
        refined_path = self.data_root / 'refinedfloatingobjects'
        if refined_path.exists():
            refined_ds = RefinedFloatingObjectsDataset(
                refined_path, split='train', transform=train_transform, image_size=self.image_size
            )
            n_val = len(refined_ds) // 5
            n_train = len(refined_ds) - n_val
            train_ref, val_ref = random_split(refined_ds, [n_train, n_val])
            if len(train_ref) > 5000:
                indices = np.random.choice(len(train_ref), 5000, replace=False)
                train_ref = torch.utils.data.Subset(train_ref, indices)
                print(f"RefinedFloatingObjects: Limited training patches to 5000")
            train_datasets.append(train_ref)
            val_datasets.append(val_ref)
        
        # S2Ships (hard negatives)
        if self.include_s2ships:
            s2ships_path = self.data_root / 'S2SHIPS'
            if s2ships_path.exists():
                s2ships_ds = S2ShipsDataset(s2ships_path, image_size=self.image_size)
                train_datasets.append(s2ships_ds)
        
        # Combine datasets
        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
            
            targets = []
            for ds in train_datasets:
                if isinstance(ds, torch.utils.data.Subset):
                    orig_ds = ds.dataset
                    indices = ds.indices
                    if hasattr(orig_ds, 'has_debris'):
                        targets.extend([orig_ds.has_debris[i] for i in indices])
                    else:
                        targets.extend([False] * len(indices))
                elif hasattr(ds, 'has_debris'):
                    targets.extend(ds.has_debris)
                else:
                    targets.extend([False] * len(ds))
                    
            num_debris = sum(targets)
            num_non_debris = len(targets) - num_debris
            
            weight_debris = len(targets) / max(num_debris, 1)
            weight_non_debris = len(targets) / max(num_non_debris, 1)
            
            self.train_weights = [weight_debris if t else weight_non_debris for t in targets]
        else:
            self.train_dataset = []
            self.train_weights = []
        
        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)
            if len(self.val_dataset) > 1000:
                indices = np.random.choice(len(self.val_dataset), 1000, replace=False)
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, indices)
                print(f"Validation dataset: Limited to 1000 patches to save memory")
        else:
            self.val_dataset = []
        
        if test_datasets:
            self.test_dataset = ConcatDataset(test_datasets)
        else:
            self.test_dataset = self.val_dataset
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Val:   {len(self.val_dataset)}")
        print(f"  Test:  {len(self.test_dataset)}")
    
    def train_dataloader(self):
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.train_weights,
            num_samples=len(self.train_dataset),
            replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Marine Debris Segmentation Models')
    
    # Data
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to MarineDebrisData folder')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image size for training')
    
    # Model
    parser.add_argument('--model', type=str, default='unet++',
                        choices=['unet', 'unet++', 'unet_attention', 'deeplabv3+', 'fpn', 'manet'],
                        help='Segmentation model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone for SMP models')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--pos-weight', type=float, default=2.0,
                        help='Positive class weight for handling imbalance')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./segmentation_output',
                        help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    # Flags
    parser.add_argument('--no-s2ships', action='store_true',
                        help='Exclude S2Ships dataset')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.name or f"{args.model}_{timestamp}"
    
    print("=" * 60)
    print("MARINE DEBRIS SEGMENTATION TRAINING")
    print("=" * 60)
    print(f"\nModel:      {args.model}")
    print(f"Encoder:    {args.encoder}")
    print(f"Data path:  {args.data_path}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs:     {args.epochs}")
    print(f"LR:         {args.lr}")
    print()
    
    # Data module
    datamodule = MarineDebrisDataModule(
        data_root=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        include_s2ships=not args.no_s2ships
    )
    
    # Model
    model = SegmentationLightningModule(
        model_name=args.model,
        in_channels=12,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        encoder_name=args.encoder
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints' / exp_name,
        filename='{epoch}-{val_f1:.4f}',
        save_top_k=3,
        monitor='val/f1',
        mode='max',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/f1',
        patience=15,
        mode='max'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / 'logs',
        name=exp_name
    )
    
    # Trainer
    if args.device == 'auto':
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        accelerator = 'gpu' if args.device == 'cuda' else 'cpu'
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Train
    print("Starting training...")
    print("=" * 60)
    
    trainer.fit(model, datamodule)
    
    # Test
    print("\n" + "=" * 60)
    print("Running test evaluation...")
    trainer.test(model, datamodule)
    
    # Save final model
    final_path = output_dir / 'checkpoints' / exp_name / 'final_model.pt'
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'threshold': model.threshold.item(),
        'model_name': args.model,
        'encoder': args.encoder,
        'image_size': args.image_size
    }, final_path)
    
    print(f"\nFinal model saved: {final_path}")
    print(f"Checkpoints: {output_dir / 'checkpoints' / exp_name}")
    print(f"Logs: {output_dir / 'logs' / exp_name}")
    
    # Save config
    config = vars(args)
    config['best_checkpoint'] = str(checkpoint_callback.best_model_path)
    with open(output_dir / 'checkpoints' / exp_name / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
