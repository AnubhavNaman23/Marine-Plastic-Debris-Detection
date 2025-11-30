"""
Marine Debris Prediction Tool

Easy-to-use prediction interface for detecting marine debris in satellite imagery.

Supports multiple input types:
1. Sentinel-2 .SAFE folder (raw download)
2. Stacked GeoTIFF file (12 bands)
3. Individual band files
4. NumPy array

Usage:
    # From command line:
    python predict.py --input path/to/image.tif --model checkpoints/best_model.pt --output results/
    
    # In Python:
    from predict import MarineDebrisPredictor
    predictor = MarineDebrisPredictor('checkpoints/best_model.pt')
    mask = predictor.predict('image.tif')
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Union, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. GeoTIFF support disabled.")

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_segmentation import get_segmentation_model, UNet


class MarineDebrisPredictor:
    """
    Marine Debris Detection Predictor
    
    Easy-to-use interface for predicting marine debris in satellite imagery.
    
    Example:
        predictor = MarineDebrisPredictor('checkpoints/best_model.pt')
        
        # Predict on a GeoTIFF
        mask, probs = predictor.predict('sentinel2_image.tif')
        
        # Visualize results
        predictor.visualize('sentinel2_image.tif', mask, save_path='result.png')
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        threshold: float = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            threshold: Classification threshold (if None, uses model's trained threshold)
        """
        self.device = self._get_device(device)
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        
        self.threshold = threshold or self.config.get('threshold', 0.5)
        self.image_size = self.config.get('image_size', 128)
        
        print(f"Model loaded: {self.config.get('model_name', 'unknown')}")
        print(f"Device: {self.device}")
        print(f"Threshold: {self.threshold:.3f}")
    
    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, dict]:
        """Load model from checkpoint."""
        import argparse
        
        # Use weights_only=False for compatibility with older checkpoints
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get config
        config = {
            'model_name': checkpoint.get('model_name', 'unet'),
            'encoder': checkpoint.get('encoder', 'resnet34'),
            'image_size': checkpoint.get('image_size', 128),
            'threshold': checkpoint.get('threshold', 0.5),
            'hr_only': False
        }
        
        # Check if this is a PyTorch Lightning checkpoint
        if 'state_dict' in checkpoint and 'hyper_parameters' in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint['state_dict']
            hparams = checkpoint.get('hyper_parameters', {})
            
            # Handle nested args in hyper_parameters
            if 'args' in hparams:
                args = hparams['args']
                config['model_name'] = getattr(args, 'model', 'unet++')
                config['hr_only'] = getattr(args, 'hr_only', False)
                config['image_size'] = getattr(args, 'image_size', 128)
            else:
                config['model_name'] = hparams.get('model', 'unet++')
                config['hr_only'] = hparams.get('hr_only', False)
            
            # Try to load using marinedebrisdetector's SegmentationModel
            try:
                import sys
                mdd_path = Path(__file__).parent.parent / "marinedebrisdetector-main"
                if mdd_path.exists():
                    sys.path.insert(0, str(mdd_path))
                
                from marinedebrisdetector.model.segmentation_model import SegmentationModel
                
                # Get the args from hyperparameters
                if 'args' in hparams:
                    args = hparams['args']
                else:
                    args = argparse.Namespace(**hparams)
                
                # Create model with args
                model = SegmentationModel(args)
                
                # Load state dict
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                
                # Get threshold from model buffer if available
                if hasattr(model, 'threshold'):
                    config['threshold'] = model.threshold.item()
                
                print(f"Loaded marinedebrisdetector model: {config['model_name']}, hr_only={config['hr_only']}")
                return model, config
                
            except Exception as e:
                print(f"Could not load as Lightning checkpoint: {e}")
                # Fall back to manual loading
                
                # Extract just the model state dict (remove 'model.' prefix)
                model_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        model_state_dict[k[6:]] = v  # Remove 'model.' prefix
                
                if model_state_dict:
                    in_channels = 4 if config.get('hr_only', False) else 12
                    model = get_segmentation_model(
                        config['model_name'],
                        in_channels=in_channels,
                        encoder_name=config['encoder']
                    )
                    model.load_state_dict(model_state_dict)
                    model = model.to(self.device)
                    model.eval()
                    return model, config
        
        # Standard checkpoint format
        model = get_segmentation_model(
            config['model_name'],
            in_channels=12,
            encoder_name=config['encoder']
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model, config
    
    def predict(
        self,
        input_data: Union[str, np.ndarray, torch.Tensor],
        return_probs: bool = True,
        tile_size: int = None,
        overlap: int = 32
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict marine debris in satellite imagery.
        
        Args:
            input_data: Path to image file, numpy array (C,H,W) or (H,W,C), or torch tensor
            return_probs: If True, also return probability map
            tile_size: Size of tiles for large images (None = auto)
            overlap: Overlap between tiles to reduce edge effects
        
        Returns:
            Binary mask (and probability map if return_probs=True)
        """
        # Load image
        image = self._load_image(input_data)
        
        # Check if we need tiling
        _, h, w = image.shape
        tile_size = tile_size or self.image_size
        
        if h > tile_size * 2 or w > tile_size * 2:
            # Use tiled prediction for large images
            probs = self._predict_tiled(image, tile_size, overlap)
        else:
            # Direct prediction
            probs = self._predict_single(image)
        
        # Threshold to get binary mask
        mask = (probs > self.threshold).astype(np.uint8)
        
        if return_probs:
            return mask, probs
        return mask
    
    def _load_image(self, input_data) -> np.ndarray:
        """Load and normalize image data."""
        if isinstance(input_data, str):
            # Load from file
            path = Path(input_data)
            
            if path.suffix.lower() in ['.tif', '.tiff']:
                if not RASTERIO_AVAILABLE:
                    raise ImportError("rasterio is required for GeoTIFF files")
                
                with rasterio.open(path) as src:
                    image = src.read().astype(np.float32)
                    
                    # Normalize (Sentinel-2 scale)
                    if image.max() > 100:
                        image = image * 1e-4
            
            elif path.suffix.lower() == '.npy':
                image = np.load(path).astype(np.float32)
                if image.max() > 100:
                    image = image * 1e-4
            
            elif path.suffix.lower() == '.npz':
                data = np.load(path)
                if 'image' in data:
                    image = data['image'].astype(np.float32)
                else:
                    image = data[list(data.keys())[0]].astype(np.float32)
                if image.max() > 100:
                    image = image * 1e-4
            
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        elif isinstance(input_data, np.ndarray):
            image = input_data.astype(np.float32)
            if image.max() > 100:
                image = image * 1e-4
        
        elif isinstance(input_data, torch.Tensor):
            image = input_data.cpu().numpy().astype(np.float32)
            if image.max() > 100:
                image = image * 1e-4
        
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")
        
        # Ensure C, H, W format
        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        elif image.ndim == 3 and image.shape[-1] <= 12:
            # Likely H, W, C format
            image = image.transpose(2, 0, 1)
        
        # Ensure 12 bands
        if image.shape[0] < 12:
            padding = np.zeros((12 - image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
            image = np.concatenate([image, padding], axis=0)
        elif image.shape[0] > 12:
            image = image[:12]
        
        return image
    
    def _predict_single(self, image: np.ndarray) -> np.ndarray:
        """Predict on a single image (may be padded/resized)."""
        c, h, w = image.shape
        
        # Pad to multiple of image_size
        pad_h = (self.image_size - h % self.image_size) % self.image_size
        pad_w = (self.image_size - w % self.image_size) % self.image_size
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        
        # Convert to tensor
        x = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Remove padding
        probs = probs[:h, :w]
        
        return probs
    
    def _predict_tiled(self, image: np.ndarray, tile_size: int, overlap: int) -> np.ndarray:
        """Predict on large images using tiling."""
        c, h, w = image.shape
        stride = tile_size - overlap
        
        # Initialize output
        probs = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        # Generate tiles
        y_positions = list(range(0, h - tile_size + 1, stride)) + [max(0, h - tile_size)]
        x_positions = list(range(0, w - tile_size + 1, stride)) + [max(0, w - tile_size)]
        
        total_tiles = len(y_positions) * len(x_positions)
        
        with tqdm(total=total_tiles, desc="Predicting tiles") as pbar:
            for y in y_positions:
                for x in x_positions:
                    # Extract tile
                    tile = image[:, y:y+tile_size, x:x+tile_size]
                    
                    # Predict
                    tile_probs = self._predict_single(tile)
                    
                    # Accumulate (with blending)
                    probs[y:y+tile_size, x:x+tile_size] += tile_probs
                    counts[y:y+tile_size, x:x+tile_size] += 1
                    
                    pbar.update(1)
        
        # Average overlapping regions
        probs = probs / np.maximum(counts, 1)
        
        return probs
    
    def visualize(
        self,
        input_data: Union[str, np.ndarray],
        mask: np.ndarray = None,
        probs: np.ndarray = None,
        save_path: str = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Visualize prediction results.
        
        Args:
            input_data: Original image
            mask: Binary prediction mask
            probs: Probability map
            save_path: Path to save figure
            figsize: Figure size
        """
        image = self._load_image(input_data)
        
        if mask is None and probs is None:
            mask, probs = self.predict(input_data)
        
        # Create RGB composite (bands 4, 3, 2 = R, G, B)
        if image.shape[0] >= 4:
            rgb = np.stack([image[3], image[2], image[1]], axis=-1)
        else:
            rgb = np.stack([image[0], image[0], image[0]], axis=-1)
        
        # Normalize RGB for display
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = np.clip(rgb * 2.5, 0, 1)  # Enhance contrast
        
        # Create figure
        n_plots = 2 + (1 if probs is not None else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title('Satellite Image (RGB)')
        axes[0].axis('off')
        
        # Mask overlay
        overlay = rgb.copy()
        debris_color = np.array([1, 0, 0])  # Red for debris
        if mask is not None:
            for c in range(3):
                overlay[:, :, c] = np.where(mask > 0, 
                                           0.5 * rgb[:, :, c] + 0.5 * debris_color[c],
                                           rgb[:, :, c])
        axes[1].imshow(overlay)
        axes[1].set_title('Debris Detection (Red = Debris)')
        axes[1].axis('off')
        
        # Probability map
        if probs is not None and n_plots > 2:
            im = axes[2].imshow(probs, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Debris Probability')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        return fig
    
    def save_prediction(
        self,
        input_path: str,
        mask: np.ndarray,
        output_path: str,
        probs: np.ndarray = None
    ):
        """
        Save prediction as GeoTIFF with same georeferencing as input.
        
        Args:
            input_path: Path to input image (for copying georeference)
            mask: Binary prediction mask
            output_path: Path to save prediction
            probs: Optional probability map to save
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for saving GeoTIFF")
        
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
        
        # Save mask
        meta['count'] = 1
        meta['dtype'] = 'uint8'
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask.astype(np.uint8), 1)
        
        print(f"Saved mask: {output_path}")
        
        # Save probabilities
        if probs is not None:
            prob_path = output_path.replace('.tif', '_probs.tif')
            meta['dtype'] = 'float32'
            
            with rasterio.open(prob_path, 'w', **meta) as dst:
                dst.write(probs.astype(np.float32), 1)
            
            print(f"Saved probabilities: {prob_path}")


def predict_from_safe_folder(
    safe_folder: str,
    model_path: str,
    output_dir: str,
    resolution: str = '10m'
) -> Tuple[np.ndarray, str]:
    """
    Predict from Sentinel-2 .SAFE folder.
    
    Args:
        safe_folder: Path to .SAFE folder
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        resolution: '10m', '20m', or '60m'
    
    Returns:
        Prediction mask and path to saved result
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required")
    
    safe_path = Path(safe_folder)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find band files
    granule_dir = safe_path / 'GRANULE'
    if not granule_dir.exists():
        raise FileNotFoundError(f"GRANULE folder not found in {safe_folder}")
    
    # Get the first (usually only) granule
    granule = list(granule_dir.iterdir())[0]
    img_data_dir = granule / 'IMG_DATA'
    
    # Find resolution folder
    if resolution == '10m':
        res_dir = img_data_dir / 'R10m'
    elif resolution == '20m':
        res_dir = img_data_dir / 'R20m'
    else:
        res_dir = img_data_dir / 'R60m'
    
    if not res_dir.exists():
        # Try without resolution subfolder (L1C data)
        res_dir = img_data_dir
    
    # Band mapping
    band_files = {}
    for band_name in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
        pattern = f'*{band_name}*.jp2'
        matches = list(res_dir.glob(pattern))
        if not matches:
            matches = list(img_data_dir.rglob(pattern))
        if matches:
            band_files[band_name] = matches[0]
    
    if len(band_files) < 4:
        raise FileNotFoundError(f"Could not find enough bands in {safe_folder}")
    
    print(f"Found {len(band_files)} bands")
    
    # Stack bands
    bands = []
    ref_shape = None
    ref_transform = None
    ref_crs = None
    
    for band_name in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']:
        if band_name in band_files:
            with rasterio.open(band_files[band_name]) as src:
                data = src.read(1).astype(np.float32)
                if ref_shape is None:
                    ref_shape = data.shape
                    ref_transform = src.transform
                    ref_crs = src.crs
                elif data.shape != ref_shape:
                    # Resample to reference shape
                    from skimage.transform import resize
                    data = resize(data, ref_shape, preserve_range=True)
                bands.append(data)
        else:
            # Add zero band
            if ref_shape:
                bands.append(np.zeros(ref_shape, dtype=np.float32))
    
    image = np.stack(bands, axis=0) * 1e-4  # Scale
    
    # Predict
    predictor = MarineDebrisPredictor(model_path)
    mask, probs = predictor.predict(image)
    
    # Save results
    scene_name = safe_path.stem.replace('.SAFE', '')
    result_path = output_path / f'{scene_name}_debris_mask.tif'
    
    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': ref_shape[1],
        'height': ref_shape[0],
        'count': 1,
        'crs': ref_crs,
        'transform': ref_transform
    }
    
    with rasterio.open(result_path, 'w', **meta) as dst:
        dst.write(mask.astype(np.uint8), 1)
    
    # Save visualization
    predictor.visualize(image, mask, probs, save_path=output_path / f'{scene_name}_visualization.png')
    
    print(f"\nResults saved to: {output_path}")
    print(f"  - {result_path.name}")
    print(f"  - {scene_name}_visualization.png")
    
    return mask, str(result_path)


def main():
    parser = argparse.ArgumentParser(description='Marine Debris Prediction')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path (.tif, .npy, .npz, or .SAFE folder)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./predictions',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization')
    parser.add_argument('--save-geotiff', action='store_true',
                        help='Save result as GeoTIFF')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    
    # Check if input is a .SAFE folder
    if input_path.suffix == '.SAFE' or (input_path.is_dir() and 'GRANULE' in [d.name for d in input_path.iterdir() if d.is_dir()]):
        print("Detected Sentinel-2 .SAFE folder")
        mask, result_path = predict_from_safe_folder(
            str(input_path),
            args.model,
            str(output_dir)
        )
    else:
        # Regular file prediction
        predictor = MarineDebrisPredictor(
            args.model,
            device=args.device,
            threshold=args.threshold
        )
        
        print(f"\nProcessing: {input_path}")
        mask, probs = predictor.predict(str(input_path))
        
        print(f"Debris pixels detected: {mask.sum():,} ({mask.mean()*100:.2f}% of image)")
        
        # Save results
        result_name = input_path.stem + '_debris_mask'
        
        if args.save_geotiff and RASTERIO_AVAILABLE and input_path.suffix.lower() in ['.tif', '.tiff']:
            predictor.save_prediction(
                str(input_path),
                mask,
                str(output_dir / f'{result_name}.tif'),
                probs
            )
        else:
            # Save as numpy
            np.savez(
                output_dir / f'{result_name}.npz',
                mask=mask,
                probs=probs,
                threshold=predictor.threshold
            )
            print(f"Saved: {output_dir / f'{result_name}.npz'}")
        
        # Visualization
        if args.visualize:
            predictor.visualize(
                str(input_path),
                mask,
                probs,
                save_path=str(output_dir / f'{result_name}.png')
            )
        else:
            # Always save visualization
            predictor.visualize(
                str(input_path),
                mask,
                probs,
                save_path=str(output_dir / f'{result_name}.png')
            )
            plt.close()
    
    print("\nPrediction complete!")


if __name__ == '__main__':
    main()
