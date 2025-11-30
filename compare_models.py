"""
Standalone Ensemble Marine Debris Detector

Runs ALL models simultaneously and compares results:
1. UNet++ Segmentation (direct architecture)
2. 7 Classification models using 24 spectral indices
3. 3 Rule-based spectral threshold models
4. Weighted ensemble combination

Outputs comprehensive comparison with metrics.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


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
        """Calculate all spectral indices."""
        indices = {}
        
        B2 = bands.get('B02', bands.get(1, np.zeros((1, 1))))
        B3 = bands.get('B03', bands.get(2, np.zeros((1, 1))))
        B4 = bands.get('B04', bands.get(3, np.zeros((1, 1))))
        B6 = bands.get('B06', bands.get(5, np.zeros((1, 1))))
        B8 = bands.get('B08', bands.get(7, np.zeros((1, 1))))
        B11 = bands.get('B11', bands.get(10, np.zeros((1, 1))))
        B12 = bands.get('B12', bands.get(11, np.zeros((1, 1))))
        
        eps = 1e-10
        
        # Normalize if DN values
        if B2.max() > 100:
            B2, B3, B4, B6, B8, B11, B12 = [x.astype(np.float32) / 10000.0 
                                             for x in [B2, B3, B4, B6, B8, B11, B12]]
        
        indices['NDVI'] = (B8 - B4) / (B8 + B4 + eps)
        indices['FAI'] = B8 - (B4 + (B11 - B4) * 0.45)
        indices['FDI'] = B8 - (B6 + (B11 - B6) * 0.1)
        indices['NDWI'] = (B3 - B8) / (B3 + B8 + eps)
        indices['NDMI'] = (B8 - B11) / (B8 + B11 + eps)
        indices['PI'] = B8 / (B8 + B4 + eps)
        indices['SABI'] = (B8 - B4) / (B3 + B2 + eps)
        indices['SI'] = np.sqrt(np.maximum(1 - B2, 0) * np.maximum(1 - B3, 0))
        indices['SRDI'] = B8 / (B4 + eps)
        indices['MNDWI'] = (B3 - B11) / (B3 + B11 + eps)
        indices['AWEI'] = 4 * (B3 - B11) - (0.25 * B8 + 2.75 * B12)
        indices['WRI'] = (B3 + B4) / (B8 + B11 + eps)
        indices['NDTI'] = (B4 - B3) / (B4 + B3 + eps)
        indices['RI'] = B4 / (B3 + eps)
        indices['GI'] = B3 / (B4 + eps)
        indices['KIVU'] = (B2 - B4) / (B3 + eps)
        indices['NDPI'] = (B11 - B3) / (B11 + B3 + eps)
        indices['RDVI'] = (B8 - B4) / np.sqrt(B8 + B4 + eps)
        indices['MSI'] = B11 / (B8 + eps)
        indices['GNDVI'] = (B8 - B3) / (B8 + B3 + eps)
        indices['EVI'] = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)
        L = 0.5
        indices['SAVI'] = ((B8 - B4) / (B8 + B4 + L + eps)) * (1 + L)
        indices['OSAVI'] = (B8 - B4) / (B8 + B4 + 0.16 + eps)
        indices['NBI'] = (B4 * B11) / (B8 + eps)
        
        return indices


# ============================================================================
# UNET++ ARCHITECTURE
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """UNet++ for semantic segmentation."""
    
    def __init__(self, in_channels=4, num_classes=1, features=[32, 64, 128, 256, 512]):
        super().__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = ConvBlock(in_channels, features[0])
        self.conv1_0 = ConvBlock(features[0], features[1])
        self.conv2_0 = ConvBlock(features[1], features[2])
        self.conv3_0 = ConvBlock(features[2], features[3])
        self.conv4_0 = ConvBlock(features[3], features[4])
        
        self.conv0_1 = ConvBlock(features[0] + features[1], features[0])
        self.conv1_1 = ConvBlock(features[1] + features[2], features[1])
        self.conv2_1 = ConvBlock(features[2] + features[3], features[2])
        self.conv3_1 = ConvBlock(features[3] + features[4], features[3])
        
        self.conv0_2 = ConvBlock(features[0]*2 + features[1], features[0])
        self.conv1_2 = ConvBlock(features[1]*2 + features[2], features[1])
        self.conv2_2 = ConvBlock(features[2]*2 + features[3], features[2])
        
        self.conv0_3 = ConvBlock(features[0]*3 + features[1], features[0])
        self.conv1_3 = ConvBlock(features[1]*3 + features[2], features[1])
        
        self.conv0_4 = ConvBlock(features[0]*4 + features[1], features[0])
        
        self.final = nn.Conv2d(features[0], num_classes, 1)
    
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        return self.final(x0_4)


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

class MLP(nn.Module):
    def __init__(self, input_dim=24, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, input_dim=24, channels=[64, 128, 256], dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout)
            ))
            in_ch = out_ch
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
        return self.fc(self.pool(x).squeeze(-1)).squeeze(-1)


class SENet(nn.Module):
    def __init__(self, input_dim=24, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)])
            prev = hd
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, 1)
    
    def forward(self, x):
        return self.fc(self.features(x)).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return F.relu(x + self.block(x))


class ResNet(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_blocks=4, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        return self.fc(self.blocks(self.proj(x))).squeeze(-1)


class Transformer(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc = nn.TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
    
    def forward(self, x):
        return self.fc(self.transformer(self.proj(x).unsqueeze(1)).squeeze(1)).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return self.fc(h[-1]).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, h = self.gru(x.unsqueeze(-1))
        return self.fc(h[-1]).squeeze(-1)


# ============================================================================
# SPECTRAL THRESHOLD MODELS
# ============================================================================

class SpectralThresholdModel:
    def __init__(self, name, rules):
        self.name = name
        self.rules = rules
    
    def predict(self, indices):
        result = None
        for idx_name, (op, thresh) in self.rules.items():
            if idx_name not in indices:
                continue
            if op == '>':
                mask = indices[idx_name] > thresh
            elif op == '<':
                mask = indices[idx_name] < thresh
            else:
                continue
            result = mask if result is None else (result & mask)
        return result.astype(np.float32) if result is not None else np.zeros((1, 1))


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class ModelResult:
    name: str
    mask: np.ndarray
    probabilities: np.ndarray
    confidence: float
    time_taken: float
    debris_pixels: int
    debris_percentage: float


# ============================================================================
# ENSEMBLE PREDICTOR
# ============================================================================

class EnsemblePredictor:
    """Ensemble Marine Debris Predictor - Runs ALL models simultaneously."""
    
    def __init__(self, seg_model_path=None, device='cpu', threshold=0.5):
        self.device = torch.device(device)
        self.threshold = threshold
        self.spectral_calc = SpectralIndicesCalculator()
        self.models = {}
        self.weights = {}
        
        if seg_model_path is None:
            seg_model_path = r'C:\Users\anubh\OneDrive\Desktop\Major\MarineDebrisProject\checkpoints\unetpp_hrbands.ckpt'
        
        print("="*60)
        print("  🌊 ENSEMBLE MARINE DEBRIS DETECTOR")
        print("="*60)
        
        self._load_segmentation(seg_model_path)
        self._init_classifiers()
        self._init_spectral_models()
        
        print(f"\n✅ {len(self.models)} models loaded")
        print("="*60)
    
    def _load_segmentation(self, path):
        """Load UNet++ segmentation model."""
        model = UNetPlusPlus(in_channels=4, num_classes=1)
        
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                if 'threshold' in ckpt:
                    self.threshold = ckpt['threshold']
                print(f"✅ Loaded checkpoint (threshold={self.threshold:.3f})")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}")
        
        model.eval()
        model.to(self.device)
        self.models['UNet++'] = model
        self.weights['UNet++'] = 2.0
    
    def _init_classifiers(self):
        """Initialize classification models."""
        configs = [
            ('MLP', MLP, 1.0), ('SENet', SENet, 0.95), ('ResNet', ResNet, 0.9),
            ('CNN1D', CNN1D, 0.85), ('Transformer', Transformer, 0.8),
            ('LSTM', LSTM, 0.75), ('GRU', GRU, 0.75)
        ]
        for name, cls, w in configs:
            m = cls()
            m.eval()
            m.to(self.device)
            self.models[name] = m
            self.weights[name] = w
    
    def _init_spectral_models(self):
        """Initialize rule-based spectral models."""
        self.models['FDI_Rule'] = SpectralThresholdModel('FDI_Rule', {'FDI': ('>', 0.02), 'NDWI': ('<', 0.3)})
        self.weights['FDI_Rule'] = 0.6
        
        self.models['FAI_Rule'] = SpectralThresholdModel('FAI_Rule', {'FAI': ('>', 0.01), 'NDVI': ('<', 0.4)})
        self.weights['FAI_Rule'] = 0.5
        
        self.models['PI_Rule'] = SpectralThresholdModel('PI_Rule', {'PI': ('>', 0.4), 'NDWI': ('<', 0.5)})
        self.weights['PI_Rule'] = 0.5
    
    def _extract_bands(self, path):
        with rasterio.open(path) as src:
            data = src.read()
            profile = src.profile
        
        bands = {}
        names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        for i in range(min(data.shape[0], len(names))):
            bands[names[i]] = data[i].astype(np.float32)
            bands[i] = data[i].astype(np.float32)
        
        return data, bands, profile
    
    def _predict_unet(self, data, name):
        if name not in self.models:
            return None, None
        
        model = self.models[name]
        h, w = data.shape[1], data.shape[2]
        
        # Get HR bands
        if data.shape[0] >= 8:
            hr = data[[1, 2, 3, 7], :, :].astype(np.float32)
        else:
            hr = data[:4, :, :].astype(np.float32)
        
        if hr.max() > 100:
            hr = hr / 10000.0
        
        tile_size, stride = 64, 48
        probs = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    ye, xe = min(y + tile_size, h), min(x + tile_size, w)
                    tile = hr[:, y:ye, x:xe]
                    
                    if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                        padded = np.zeros((4, tile_size, tile_size), dtype=np.float32)
                        padded[:, :tile.shape[1], :tile.shape[2]] = tile
                        tile = padded
                    
                    t = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                    pred = torch.sigmoid(model(t)).cpu().numpy()[0, 0]
                    
                    th, tw = ye - y, xe - x
                    probs[y:ye, x:xe] += pred[:th, :tw]
                    counts[y:ye, x:xe] += 1
        
        probs = probs / np.maximum(counts, 1)
        mask = (probs > self.threshold).astype(np.uint8)
        return mask, probs
    
    def _predict_classifier(self, indices, name):
        if name not in self.models:
            return None, None
        
        model = self.models[name]
        
        if isinstance(model, SpectralThresholdModel):
            mask = model.predict(indices)
            shape = list(indices.values())[0].shape
            mask = mask.reshape(shape) if mask.size > 1 else np.zeros(shape)
            return mask.astype(np.uint8), mask.astype(np.float32)
        
        shape = list(indices.values())[0].shape
        h, w = shape
        
        feats = []
        for n in self.spectral_calc.indices_names:
            feats.append(indices.get(n, np.zeros(shape)))
        
        feats = np.stack(feats, axis=-1)
        feats = np.nan_to_num(feats, 0, 0, 0)
        feats = np.clip(feats, -10, 10)
        flat = feats.reshape(-1, len(self.spectral_calc.indices_names))
        
        batch = 10000
        probs = np.zeros(flat.shape[0], dtype=np.float32)
        
        with torch.no_grad():
            for i in range(0, len(flat), batch):
                b = torch.from_numpy(flat[i:i+batch]).float().to(self.device)
                try:
                    probs[i:i+batch] = torch.sigmoid(model(b)).cpu().numpy()
                except:
                    probs[i:i+batch] = 0.5
        
        probs = probs.reshape(h, w)
        mask = (probs > 0.5).astype(np.uint8)
        return mask, probs
    
    def predict(self, image_path):
        """Run ALL models and return results."""
        print(f"\n🔍 Processing: {os.path.basename(image_path)}")
        
        data, bands, profile = self._extract_bands(image_path)
        h, w = data.shape[1], data.shape[2]
        print(f"   Image: {w} × {h} pixels, {data.shape[0]} bands")
        
        print("   📊 Calculating 24 spectral indices...")
        indices = self.spectral_calc.calculate_all(bands)
        
        results = {}
        print("\n   🚀 Running all models...")
        
        for name in tqdm(self.models.keys(), desc="   Models"):
            start = time.time()
            
            if 'UNet' in name:
                mask, probs = self._predict_unet(data, name)
            else:
                mask, probs = self._predict_classifier(indices, name)
            
            if mask is not None:
                results[name] = ModelResult(
                    name=name, mask=mask, probabilities=probs,
                    confidence=float(probs.mean()) if probs is not None else 0,
                    time_taken=time.time() - start,
                    debris_pixels=int(mask.sum()),
                    debris_percentage=float(mask.mean() * 100)
                )
        
        # Ensemble
        print("\n   🎯 Creating weighted ensemble...")
        ens_probs = np.zeros((h, w), dtype=np.float32)
        total_w = 0
        
        for name, r in results.items():
            if r.probabilities is not None and r.probabilities.shape == (h, w):
                w = self.weights.get(name, 1.0)
                ens_probs += r.probabilities * w
                total_w += w
        
        if total_w > 0:
            ens_probs /= total_w
        
        ens_mask = (ens_probs > 0.5).astype(np.uint8)
        results['ENSEMBLE'] = ModelResult(
            name='ENSEMBLE', mask=ens_mask, probabilities=ens_probs,
            confidence=float(ens_probs.mean()),
            time_taken=sum(r.time_taken for r in results.values()),
            debris_pixels=int(ens_mask.sum()),
            debris_percentage=float(ens_mask.mean() * 100)
        )
        
        return results
    
    def compare(self, results, output_dir='comparison_output'):
        """Generate comparison visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("                    📊 MODEL COMPARISON RESULTS")
        print("="*70)
        print(f"{'Model':<20} {'Debris Pixels':>15} {'Percentage':>12} {'Time (s)':>10}")
        print("-"*70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].debris_pixels, reverse=True)
        for name, r in sorted_results:
            print(f"{name:<20} {r.debris_pixels:>15,} {r.debris_percentage:>11.2f}% {r.time_taken:>9.2f}s")
        
        print("="*70)
        
        # Best models
        non_ens = [(n, r) for n, r in sorted_results if n != 'ENSEMBLE']
        if non_ens:
            print(f"\n🏆 Most Sensitive: {non_ens[0][0]} ({non_ens[0][1].debris_pixels:,} pixels)")
            print(f"📉 Most Conservative: {non_ens[-1][0]} ({non_ens[-1][1].debris_pixels:,} pixels)")
        
        if 'ENSEMBLE' in results:
            print(f"⚖️ ENSEMBLE (Balanced): {results['ENSEMBLE'].debris_pixels:,} pixels")
        
        # Visualizations
        self._visualize(results, output_dir)
        
        return output_dir
    
    def _visualize(self, results, output_dir):
        """Save visualizations."""
        n = len(results)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        # Masks
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.array(axes).flatten()
        
        for i, (name, r) in enumerate(results.items()):
            axes[i].imshow(r.mask, cmap='Reds')
            axes[i].set_title(f'{name}\n{r.debris_pixels:,} px ({r.debris_percentage:.2f}%)', fontsize=9)
            axes[i].axis('off')
        
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('🔍 All Models - Debris Masks', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_masks.png'), dpi=150)
        plt.close()
        
        # Probabilities
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.array(axes).flatten()
        
        for i, (name, r) in enumerate(results.items()):
            im = axes[i].imshow(r.probabilities, cmap='hot', vmin=0, vmax=1)
            axes[i].set_title(name, fontsize=9)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('🌡️ All Models - Probability Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_probabilities.png'), dpi=150)
        plt.close()
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(results.keys())
        pcts = [results[n].debris_percentage for n in names]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
        
        bars = ax.barh(names, pcts, color=colors)
        ax.set_xlabel('Debris Percentage (%)')
        ax.set_title('📊 Detection Rate by Model', fontsize=14)
        
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_chart.png'), dpi=150)
        plt.close()
        
        # Ensemble standalone
        if 'ENSEMBLE' in results:
            r = results['ENSEMBLE']
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            axes[0].imshow(r.mask, cmap='Reds')
            axes[0].set_title(f'ENSEMBLE Mask\n{r.debris_pixels:,} px ({r.debris_percentage:.2f}%)')
            axes[0].axis('off')
            
            im = axes[1].imshow(r.probabilities, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title('ENSEMBLE Probability')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            plt.suptitle('🎯 ENSEMBLE PREDICTION', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ensemble_result.png'), dpi=150)
            plt.close()
        
        print(f"\n📁 Saved to: {output_dir}/")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Marine Debris Detector')
    parser.add_argument('--input', type=str, required=True, help='Input image')
    parser.add_argument('--output', type=str, default='comparison_output', help='Output dir')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
    
    args = parser.parse_args()
    
    predictor = EnsemblePredictor(threshold=args.threshold)
    results = predictor.predict(args.input)
    predictor.compare(results, args.output)
    
    print("\n✅ Complete!")


if __name__ == '__main__':
    main()
