"""
Ensemble Marine Debris Detector

This system combines multiple models and techniques:
1. UNet++ Segmentation (marinedebrisdetector)
2. Classification models (MLP, SENet, ResNet, CNN1D, etc.)
3. Spectral indices analysis (NDVI, FAI, FDI, NDWI, etc.)
4. Multi-model ensemble with voting/averaging

Provides comprehensive comparison with metrics.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'marinedebrisdetector-main'))


# ============================================================================
# SPECTRAL INDICES (24 indices for debris detection)
# ============================================================================

class SpectralIndicesCalculator:
    """Calculate 24 spectral indices for marine debris detection."""
    
    def __init__(self):
        self.indices_names = [
            'NDVI', 'FAI', 'FDI', 'NDWI', 'NDMI', 'PI', 'SABI', 'SI', 'SRDI',
            'MNDWI', 'AWEI', 'WRI', 'NDTI', 'RI', 'GI', 'KIVU', 'NDPI',
            'RDVI', 'MSI', 'GNDVI', 'EVI', 'SAVI', 'OSAVI', 'NBI'
        ]
    
    def calculate_all(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate all spectral indices from band dictionary."""
        indices = {}
        
        # Extract bands (handle both 0-indexed and name-indexed)
        B2 = bands.get('B02', bands.get('B2', bands.get(0, np.zeros((1, 1)))))
        B3 = bands.get('B03', bands.get('B3', bands.get(1, np.zeros((1, 1)))))
        B4 = bands.get('B04', bands.get('B4', bands.get(2, np.zeros((1, 1)))))
        B5 = bands.get('B05', bands.get('B5', bands.get(3, np.zeros((1, 1)))))
        B6 = bands.get('B06', bands.get('B6', bands.get(4, np.zeros((1, 1)))))
        B7 = bands.get('B07', bands.get('B7', bands.get(5, np.zeros((1, 1)))))
        B8 = bands.get('B08', bands.get('B8', bands.get(6, np.zeros((1, 1)))))
        B8A = bands.get('B8A', bands.get(7, B8))
        B11 = bands.get('B11', bands.get(9, np.zeros((1, 1))))
        B12 = bands.get('B12', bands.get(10, np.zeros((1, 1))))
        
        eps = 1e-10
        
        # 1. NDVI - Normalized Difference Vegetation Index
        indices['NDVI'] = (B8 - B4) / (B8 + B4 + eps)
        
        # 2. FAI - Floating Algae Index
        indices['FAI'] = B8 - (B4 + (B11 - B4) * (832.8 - 664.6) / (1613.7 - 664.6))
        
        # 3. FDI - Floating Debris Index
        indices['FDI'] = B8 - (B6 + (B11 - B6) * ((832.8 - 740) / (1613.7 - 740)))
        
        # 4. NDWI - Normalized Difference Water Index
        indices['NDWI'] = (B3 - B8) / (B3 + B8 + eps)
        
        # 5. NDMI - Normalized Difference Moisture Index
        indices['NDMI'] = (B8 - B11) / (B8 + B11 + eps)
        
        # 6. PI - Plastic Index
        indices['PI'] = B8 / (B8 + B4 + eps)
        
        # 7. SABI - Surface Algal Bloom Index
        indices['SABI'] = (B8 - B4) / (B3 + B2 + eps)
        
        # 8. SI - Shadow Index
        indices['SI'] = np.sqrt((1 - B2) * (1 - B3))
        
        # 9. SRDI - Simple Ratio Debris Index
        indices['SRDI'] = B8 / (B4 + eps)
        
        # 10. MNDWI - Modified NDWI
        indices['MNDWI'] = (B3 - B11) / (B3 + B11 + eps)
        
        # 11. AWEI - Automated Water Extraction Index
        indices['AWEI'] = 4 * (B3 - B11) - (0.25 * B8 + 2.75 * B12)
        
        # 12. WRI - Water Ratio Index
        indices['WRI'] = (B3 + B4) / (B8 + B11 + eps)
        
        # 13. NDTI - Normalized Difference Turbidity Index
        indices['NDTI'] = (B4 - B3) / (B4 + B3 + eps)
        
        # 14. RI - Redness Index
        indices['RI'] = B4 / (B3 + eps)
        
        # 15. GI - Greenness Index
        indices['GI'] = B3 / (B4 + eps)
        
        # 16. KIVU - Lake Kivu Index
        indices['KIVU'] = (B2 - B4) / (B3 + eps)
        
        # 17. NDPI - Normalized Difference Pond Index
        indices['NDPI'] = (B11 - B3) / (B11 + B3 + eps)
        
        # 18. RDVI - Renormalized Difference Vegetation Index
        indices['RDVI'] = (B8 - B4) / np.sqrt(B8 + B4 + eps)
        
        # 19. MSI - Moisture Stress Index
        indices['MSI'] = B11 / (B8 + eps)
        
        # 20. GNDVI - Green NDVI
        indices['GNDVI'] = (B8 - B3) / (B8 + B3 + eps)
        
        # 21. EVI - Enhanced Vegetation Index
        indices['EVI'] = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)
        
        # 22. SAVI - Soil Adjusted Vegetation Index
        L = 0.5
        indices['SAVI'] = ((B8 - B4) / (B8 + B4 + L + eps)) * (1 + L)
        
        # 23. OSAVI - Optimized SAVI
        indices['OSAVI'] = (B8 - B4) / (B8 + B4 + 0.16 + eps)
        
        # 24. NBI - New Built-up Index (debris often looks like built-up)
        indices['NBI'] = (B4 * B11) / (B8 + eps)
        
        return indices


# ============================================================================
# ADVANCED LOSS FUNCTIONS
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined loss function with multiple components."""
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            'bce': 1.0,
            'dice': 1.0,
            'focal': 0.5,
            'tversky': 0.5,
            'lovasz': 0.3
        }
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        
        if self.weights.get('bce', 0) > 0:
            total_loss += self.weights['bce'] * self.bce_loss(pred, target)
        
        if self.weights.get('dice', 0) > 0:
            total_loss += self.weights['dice'] * self.dice_loss(pred, target)
        
        if self.weights.get('focal', 0) > 0:
            total_loss += self.weights['focal'] * self.focal_loss(pred, target)
        
        if self.weights.get('tversky', 0) > 0:
            total_loss += self.weights['tversky'] * self.tversky_loss(pred, target)
        
        return total_loss
    
    def bce_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def tversky_loss(self, pred, target, alpha=0.7, beta=0.3, smooth=1e-6):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - tversky


# ============================================================================
# CLASSIFICATION MODELS (9 models)
# ============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron - Best performing at 94.29%"""
    def __init__(self, input_dim=24, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)


class CNN1D(nn.Module):
    """1D Convolutional Neural Network - 93.21%"""
    def __init__(self, input_dim=24, channels=[64, 128, 256], dropout=0.3):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            in_ch = out_ch
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 24)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c = x.shape
        y = F.adaptive_avg_pool1d(x.unsqueeze(-1), 1).squeeze(-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class SENet(nn.Module):
    """Squeeze-and-Excitation Network - 94.21%"""
    def __init__(self, input_dim=24, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                SEBlock(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        x = self.features(x)
        return self.fc(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual Block for ResNet"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNet(nn.Module):
    """Residual Network - 94.14%"""
    def __init__(self, input_dim=24, hidden_dim=128, num_blocks=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.fc(x).squeeze(-1)


class AttentionBlock(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out).squeeze(1)


class TransformerClassifier(nn.Module):
    """Transformer-based Classifier"""
    def __init__(self, input_dim=24, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for attn in self.attention_layers:
            x = attn(x)
        return self.fc(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    """LSTM-based Classifier"""
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, 24, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


class GRUClassifier(nn.Module):
    """GRU-based Classifier"""
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        _, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze(-1)


# ============================================================================
# ENSEMBLE PREDICTOR
# ============================================================================

@dataclass
class ModelResult:
    """Result from a single model prediction."""
    name: str
    mask: np.ndarray
    probabilities: np.ndarray
    confidence: float
    time_taken: float
    debris_pixels: int
    debris_percentage: float


class EnsemblePredictor:
    """
    Ensemble Marine Debris Predictor
    
    Combines:
    1. UNet++ Segmentation model
    2. 9 Classification models (MLP, SENet, ResNet, CNN1D, Transformer, LSTM, GRU, etc.)
    3. 24 Spectral indices
    4. Voting/averaging ensemble
    """
    
    def __init__(
        self,
        segmentation_model_path: str = None,
        classifier_dir: str = None,
        device: str = 'cpu',
        threshold: float = 0.5
    ):
        self.device = torch.device(device)
        self.threshold = threshold
        self.spectral_calc = SpectralIndicesCalculator()
        
        self.models = {}
        self.model_weights = {}
        
        # Default paths
        if segmentation_model_path is None:
            segmentation_model_path = r'C:\Users\anubh\OneDrive\Desktop\Major\MarineDebrisProject\checkpoints\unetpp_hrbands.ckpt'
        if classifier_dir is None:
            classifier_dir = r'C:\Users\anubh\OneDrive\Desktop\Major\PlasticDebrisDetector\models'
        
        # Load segmentation model
        self._load_segmentation_model(segmentation_model_path)
        
        # Load/create classification models
        self._load_classification_models(classifier_dir)
        
        print(f"\n✅ Ensemble loaded with {len(self.models)} models")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def _load_segmentation_model(self, model_path: str):
        """Load UNet++ segmentation model."""
        if not os.path.exists(model_path):
            print(f"⚠️ Segmentation model not found: {model_path}")
            return
        
        try:
            from marinedebrisdetector.model import SegmentationModel
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            hparams = checkpoint.get('hyper_parameters', {})
            
            if 'args' in hparams:
                args = hparams['args']
            else:
                args = hparams
            
            model = SegmentationModel(
                learning_rate=getattr(args, 'learning_rate', 0.001),
                weight_decay=getattr(args, 'weight_decay', 0.0001),
                pos_weight=getattr(args, 'pos_weight', 1.0),
                model=getattr(args, 'model', 'unet++'),
                pretrained=False,
                num_classes=1,
                in_channels=getattr(args, 'in_channels', 4),
                loss=getattr(args, 'loss', 'focal'),
                hr_only=getattr(args, 'hr_only', True)
            )
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            model.to(self.device)
            
            self.models['UNet++'] = model
            self.model_weights['UNet++'] = 2.0  # Higher weight for segmentation
            
            # Get threshold from checkpoint
            if 'threshold' in checkpoint:
                self.threshold = checkpoint['threshold']
            
            print(f"✅ Loaded UNet++ segmentation model")
            
        except Exception as e:
            print(f"⚠️ Error loading segmentation model: {e}")
    
    def _load_classification_models(self, model_dir: str):
        """Load or create classification models."""
        model_classes = {
            'MLP': (MLP, {'input_dim': 24, 'hidden_dims': [512, 256, 128, 64]}, 1.0),
            'SENet': (SENet, {'input_dim': 24, 'hidden_dims': [256, 128, 64]}, 0.95),
            'ResNet': (ResNet, {'input_dim': 24, 'hidden_dim': 128, 'num_blocks': 4}, 0.9),
            'CNN1D': (CNN1D, {'input_dim': 24, 'channels': [64, 128, 256]}, 0.85),
            'Transformer': (TransformerClassifier, {'input_dim': 24, 'hidden_dim': 128}, 0.8),
            'LSTM': (LSTMClassifier, {'input_dim': 24, 'hidden_dim': 128}, 0.75),
            'GRU': (GRUClassifier, {'input_dim': 24, 'hidden_dim': 128}, 0.75),
        }
        
        os.makedirs(model_dir, exist_ok=True)
        
        for name, (model_class, kwargs, weight) in model_classes.items():
            model_path = os.path.join(model_dir, f'{name.lower()}_classifier.pth')
            
            try:
                model = model_class(**kwargs)
                
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(state)
                    print(f"✅ Loaded {name} classifier")
                else:
                    print(f"⚠️ {name} not trained, using random init")
                
                model.eval()
                model.to(self.device)
                self.models[name] = model
                self.model_weights[name] = weight
                
            except Exception as e:
                print(f"⚠️ Error with {name}: {e}")
    
    def _extract_bands(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """Extract bands from image file."""
        with rasterio.open(image_path) as src:
            data = src.read()
            profile = src.profile
        
        # Create band dictionary
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        bands = {}
        
        for i, name in enumerate(band_names[:data.shape[0]]):
            bands[name] = data[i].astype(np.float32)
        
        return data, bands, profile
    
    def _segment_with_unet(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run UNet++ segmentation."""
        if 'UNet++' not in self.models:
            return None, None
        
        model = self.models['UNet++']
        h, w = data.shape[1], data.shape[2]
        
        # Get HR bands (B2, B3, B4, B8 = indices 1, 2, 3, 7)
        if data.shape[0] >= 8:
            hr_data = data[[1, 2, 3, 7], :, :]
        else:
            hr_data = data[:4, :, :]
        
        # Normalize
        hr_data = hr_data.astype(np.float32) / 10000.0
        
        # Predict with tiling
        tile_size = 64
        stride = 48
        probs = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    
                    tile = hr_data[:, y:y_end, x:x_end]
                    
                    # Pad if needed
                    if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                        padded = np.zeros((4, tile_size, tile_size), dtype=np.float32)
                        padded[:, :tile.shape[1], :tile.shape[2]] = tile
                        tile = padded
                    
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                    pred = model(tile_tensor)
                    pred_np = torch.sigmoid(pred).cpu().numpy()[0, 0]
                    
                    th, tw = y_end - y, x_end - x
                    probs[y:y_end, x:x_end] += pred_np[:th, :tw]
                    counts[y:y_end, x:x_end] += 1
        
        probs = probs / np.maximum(counts, 1)
        mask = (probs > self.threshold).astype(np.uint8)
        
        return mask, probs
    
    def _classify_with_indices(self, bands: dict, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Classify using spectral indices and a classifier."""
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        
        # Calculate spectral indices
        indices = self.spectral_calc.calculate_all(bands)
        
        # Get image shape from first valid band
        shape = None
        for v in bands.values():
            if v is not None and v.size > 1:
                shape = v.shape
                break
        
        if shape is None:
            return None, None
        
        h, w = shape
        
        # Stack indices into feature array
        feature_names = list(indices.keys())
        features = np.stack([indices[name] for name in feature_names], axis=-1)  # (H, W, 24)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        features = np.clip(features, -10, 10)
        
        # Reshape for model
        flat_features = features.reshape(-1, len(feature_names))  # (H*W, 24)
        
        # Predict in batches
        batch_size = 10000
        probs = np.zeros(flat_features.shape[0], dtype=np.float32)
        
        with torch.no_grad():
            for i in range(0, len(flat_features), batch_size):
                batch = torch.from_numpy(flat_features[i:i+batch_size]).float().to(self.device)
                pred = model(batch)
                probs[i:i+batch_size] = torch.sigmoid(pred).cpu().numpy()
        
        probs = probs.reshape(h, w)
        mask = (probs > 0.5).astype(np.uint8)
        
        return mask, probs
    
    def predict(self, image_path: str, use_ensemble: bool = True) -> Dict[str, ModelResult]:
        """
        Run prediction with all models.
        
        Args:
            image_path: Path to input image
            use_ensemble: Whether to combine results
            
        Returns:
            Dictionary of results from each model
        """
        print(f"\n🔍 Processing: {image_path}")
        
        # Extract bands
        data, bands, profile = self._extract_bands(image_path)
        h, w = data.shape[1], data.shape[2]
        print(f"   Image size: {w} x {h}")
        
        results = {}
        
        # Run UNet++ segmentation
        if 'UNet++' in self.models:
            print("   Running UNet++ segmentation...")
            start = time.time()
            mask, probs = self._segment_with_unet(data)
            if mask is not None:
                results['UNet++'] = ModelResult(
                    name='UNet++',
                    mask=mask,
                    probabilities=probs,
                    confidence=float(probs.mean()),
                    time_taken=time.time() - start,
                    debris_pixels=int(mask.sum()),
                    debris_percentage=float(mask.mean() * 100)
                )
                print(f"      ✅ Debris: {mask.sum():,} pixels ({mask.mean()*100:.2f}%)")
        
        # Run classification models
        classifier_names = [n for n in self.models.keys() if n != 'UNet++']
        
        for name in classifier_names:
            print(f"   Running {name} classifier...")
            start = time.time()
            mask, probs = self._classify_with_indices(bands, name)
            if mask is not None:
                results[name] = ModelResult(
                    name=name,
                    mask=mask,
                    probabilities=probs,
                    confidence=float(probs.mean()),
                    time_taken=time.time() - start,
                    debris_pixels=int(mask.sum()),
                    debris_percentage=float(mask.mean() * 100)
                )
                print(f"      ✅ Debris: {mask.sum():,} pixels ({mask.mean()*100:.2f}%)")
        
        # Create ensemble prediction
        if use_ensemble and len(results) > 1:
            print("   Creating ensemble prediction...")
            ensemble_probs = np.zeros((h, w), dtype=np.float32)
            total_weight = 0
            
            for name, result in results.items():
                weight = self.model_weights.get(name, 1.0)
                ensemble_probs += result.probabilities * weight
                total_weight += weight
            
            ensemble_probs /= total_weight
            ensemble_mask = (ensemble_probs > 0.5).astype(np.uint8)
            
            results['ENSEMBLE'] = ModelResult(
                name='ENSEMBLE (Weighted Average)',
                mask=ensemble_mask,
                probabilities=ensemble_probs,
                confidence=float(ensemble_probs.mean()),
                time_taken=sum(r.time_taken for r in results.values()),
                debris_pixels=int(ensemble_mask.sum()),
                debris_percentage=float(ensemble_mask.mean() * 100)
            )
            print(f"      ✅ Ensemble Debris: {ensemble_mask.sum():,} pixels ({ensemble_mask.mean()*100:.2f}%)")
        
        return results
    
    def compare_models(
        self,
        results: Dict[str, ModelResult],
        ground_truth: np.ndarray = None,
        output_dir: str = 'comparison_output'
    ) -> Dict[str, dict]:
        """
        Compare model performance with metrics.
        
        If ground_truth is provided, calculates accuracy, precision, recall, F1.
        Otherwise, compares predictions across models.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = {}
        
        for name, result in results.items():
            metrics[name] = {
                'debris_pixels': result.debris_pixels,
                'debris_percentage': result.debris_percentage,
                'confidence': result.confidence,
                'time_seconds': result.time_taken
            }
            
            if ground_truth is not None:
                pred_flat = result.mask.flatten()
                gt_flat = ground_truth.flatten()
                
                metrics[name]['accuracy'] = accuracy_score(gt_flat, pred_flat)
                metrics[name]['precision'] = precision_score(gt_flat, pred_flat, zero_division=0)
                metrics[name]['recall'] = recall_score(gt_flat, pred_flat, zero_division=0)
                metrics[name]['f1_score'] = f1_score(gt_flat, pred_flat, zero_division=0)
        
        # Generate comparison visualization
        self._visualize_comparison(results, metrics, output_dir)
        
        # Print comparison table
        self._print_comparison_table(metrics, ground_truth is not None)
        
        return metrics
    
    def _visualize_comparison(self, results: Dict[str, ModelResult], metrics: dict, output_dir: str):
        """Generate visualization comparing all models."""
        n_models = len(results)
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, result) in enumerate(results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            im = ax.imshow(result.mask, cmap='Reds')
            ax.set_title(f'{name}\n{result.debris_pixels:,} px ({result.debris_percentage:.2f}%)', 
                        fontsize=10)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(results), rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Model Comparison - Debris Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Probability maps
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, result) in enumerate(results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            im = ax.imshow(result.probabilities, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'{name} - Probability', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        for idx in range(len(results), rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Model Comparison - Probability Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probability_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        names = list(metrics.keys())
        debris_pct = [metrics[n]['debris_percentage'] for n in names]
        times = [metrics[n]['time_seconds'] for n in names]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        
        ax = axes[0]
        bars = ax.bar(names, debris_pct, color=colors)
        ax.set_ylabel('Debris Percentage (%)')
        ax.set_title('Detection Rate by Model')
        ax.tick_params(axis='x', rotation=45)
        for bar, pct in zip(bars, debris_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{pct:.2f}%', ha='center', va='bottom', fontsize=9)
        
        ax = axes[1]
        bars = ax.bar(names, times, color=colors)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Processing Time by Model')
        ax.tick_params(axis='x', rotation=45)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Visualizations saved to: {output_dir}/")
    
    def _print_comparison_table(self, metrics: dict, has_ground_truth: bool):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("                    MODEL COMPARISON RESULTS")
        print("="*80)
        
        if has_ground_truth:
            print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>10}")
            print("-"*80)
            for name, m in metrics.items():
                print(f"{name:<20} {m['accuracy']*100:>9.2f}% {m['precision']*100:>9.2f}% "
                      f"{m['recall']*100:>9.2f}% {m['f1_score']*100:>9.2f}% {m['time_seconds']:>9.2f}s")
        else:
            print(f"{'Model':<20} {'Debris Pixels':>15} {'Percentage':>12} {'Confidence':>12} {'Time':>10}")
            print("-"*80)
            for name, m in metrics.items():
                print(f"{name:<20} {m['debris_pixels']:>15,} {m['debris_percentage']:>11.2f}% "
                      f"{m['confidence']:>11.4f} {m['time_seconds']:>9.2f}s")
        
        print("="*80)
        
        # Find best model
        if 'ENSEMBLE' in metrics:
            print(f"\n🏆 ENSEMBLE provides balanced prediction combining all models")
        
        best_detector = max(
            [(n, m['debris_pixels']) for n, m in metrics.items() if n != 'ENSEMBLE'],
            key=lambda x: x[1]
        )
        print(f"📈 Highest detection: {best_detector[0]} ({best_detector[1]:,} pixels)")


def main():
    """Main function to run ensemble prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Marine Debris Detector')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='comparison_output', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize ensemble
    predictor = EnsemblePredictor(
        threshold=args.threshold,
        device=args.device
    )
    
    # Run prediction
    results = predictor.predict(args.input)
    
    # Compare models
    metrics = predictor.compare_models(results, output_dir=args.output)
    
    # Save ensemble result
    if 'ENSEMBLE' in results:
        ensemble_result = results['ENSEMBLE']
        output_path = os.path.join(args.output, 'ensemble_debris_mask.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(ensemble_result.mask, cmap='Reds')
        plt.title(f'Ensemble Detection\n{ensemble_result.debris_pixels:,} pixels ({ensemble_result.debris_percentage:.2f}%)')
        plt.colorbar(label='Debris')
        plt.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Ensemble result saved: {output_path}")


if __name__ == '__main__':
    main()
