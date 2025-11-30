"""
Batch Process Multiple Sentinel-2 Images

This script:
1. Finds all sets of Sentinel-2 band files
2. Stacks them into multi-band TIFFs
3. Runs debris detection on each
4. Combines results for multi-temporal analysis

Usage:
    python batch_process.py --input-dir Browser_images --output-dir batch_output
"""

import os
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import rasterio
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent))


def find_band_sets(input_dir: Path) -> dict:
    """
    Find sets of Sentinel-2 band files that belong together.
    Groups by date/timestamp in filename.
    """
    band_sets = defaultdict(dict)
    
    # Pattern to match Sentinel-2 band files
    # Example: 2025-11-29-00_00_2025-11-29-23_59_Sentinel-2_L2A_B02_(Raw).tiff
    band_pattern = re.compile(r'(.+)_Sentinel-2_L2A_(B\d+[A]?)_\(Raw\)\.tiff?', re.IGNORECASE)
    
    for file in input_dir.glob('*.tiff'):
        match = band_pattern.match(file.name)
        if match:
            date_prefix = match.group(1)
            band_name = match.group(2).upper()
            band_sets[date_prefix][band_name] = file
    
    # Also check for files in subdirectories
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.glob('*.tiff'):
                match = band_pattern.match(file.name)
                if match:
                    date_prefix = f"{subdir.name}_{match.group(1)}"
                    band_name = match.group(2).upper()
                    band_sets[date_prefix][band_name] = file
    
    return dict(band_sets)


def stack_bands(band_files: dict, output_path: Path) -> bool:
    """
    Stack individual band files into a single multi-band TIFF.
    
    Band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    """
    # Required bands in order
    band_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    
    # Check which bands are available
    available = [b for b in band_order if b in band_files]
    
    if len(available) < 4:
        print(f"  Warning: Only {len(available)} bands found, need at least 4 (HR bands)")
        return False
    
    # Read and stack bands
    bands = []
    profile = None
    
    for band_name in band_order:
        if band_name in band_files:
            with rasterio.open(band_files[band_name]) as src:
                band_data = src.read(1)
                bands.append(band_data)
                if profile is None:
                    profile = src.profile.copy()
        else:
            # Fill missing bands with zeros (will be handled by model)
            if bands:
                bands.append(np.zeros_like(bands[0]))
            else:
                print(f"  Error: Cannot determine image size, no bands loaded yet")
                return False
    
    # Stack and save
    stacked = np.stack(bands, axis=0)
    profile.update(count=len(bands), dtype=stacked.dtype)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(stacked)
    
    return True


def run_prediction(image_path: Path, model_path: Path, output_dir: Path, threshold: float = 0.5):
    """Run debris prediction on a stacked image."""
    from predict import MarineDebrisPredictor
    
    predictor = MarineDebrisPredictor(str(model_path), threshold=threshold)
    
    mask, probs = predictor.predict(str(image_path))
    
    # Save results
    output_name = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_dir / f'{output_name}_result.npz',
        mask=mask,
        probs=probs
    )
    
    # Save visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(mask, cmap='Reds')
    axes[0].set_title(f'Debris Mask\n{mask.sum():,} pixels ({mask.mean()*100:.2f}%)')
    axes[0].axis('off')
    
    axes[1].imshow(probs, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{output_name}_visualization.png', dpi=150)
    plt.close()
    
    return mask, probs


def combine_temporal_results(results: list, output_dir: Path):
    """
    Combine results from multiple dates for temporal analysis.
    
    Creates:
    - Frequency map: How often each pixel is detected as debris
    - Persistent debris: Pixels detected in majority of images
    - Maximum probability: Highest probability across all dates
    """
    if not results:
        return
    
    masks = [r['mask'] for r in results]
    probs = [r['probs'] for r in results]
    
    # Ensure all arrays have same shape (use minimum)
    min_h = min(m.shape[0] for m in masks)
    min_w = min(m.shape[1] for m in masks)
    
    masks = [m[:min_h, :min_w] for m in masks]
    probs = [p[:min_h, :min_w] for p in probs]
    
    # Stack arrays
    mask_stack = np.stack(masks, axis=0)
    prob_stack = np.stack(probs, axis=0)
    
    # Compute temporal statistics
    frequency_map = mask_stack.mean(axis=0)  # Fraction of times detected
    persistent_debris = frequency_map > 0.5  # Detected in majority of images
    max_probability = prob_stack.max(axis=0)  # Maximum probability
    mean_probability = prob_stack.mean(axis=0)  # Mean probability
    
    # Save combined results
    np.savez_compressed(
        output_dir / 'temporal_analysis.npz',
        frequency_map=frequency_map,
        persistent_debris=persistent_debris,
        max_probability=max_probability,
        mean_probability=mean_probability,
        num_images=len(masks)
    )
    
    # Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    im1 = axes[0, 0].imshow(frequency_map, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Detection Frequency\n(across {len(masks)} images)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='Frequency')
    
    axes[0, 1].imshow(persistent_debris, cmap='Reds')
    axes[0, 1].set_title(f'Persistent Debris\n({persistent_debris.sum():,} pixels)')
    axes[0, 1].axis('off')
    
    im3 = axes[1, 0].imshow(max_probability, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Maximum Probability')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], label='Probability')
    
    im4 = axes[1, 1].imshow(mean_probability, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Mean Probability')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], label='Probability')
    
    plt.suptitle('Multi-Temporal Debris Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_analysis.png', dpi=150)
    plt.close()
    
    print(f"\n📊 Temporal Analysis Summary:")
    print(f"   Images analyzed: {len(masks)}")
    print(f"   Persistent debris pixels: {persistent_debris.sum():,}")
    print(f"   Max frequency: {frequency_map.max()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Batch process Sentinel-2 images for debris detection')
    parser.add_argument('--input-dir', type=str, default='Browser_images',
                        help='Directory containing Sentinel-2 band files')
    parser.add_argument('--output-dir', type=str, default='batch_output',
                        help='Output directory for results')
    parser.add_argument('--model', type=str, 
                        default=r'C:\Users\anubh\OneDrive\Desktop\Major\MarineDebrisProject\checkpoints\unetpp_hrbands.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--stack-only', action='store_true',
                        help='Only stack bands, do not run prediction')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Scanning for Sentinel-2 band files...")
    band_sets = find_band_sets(input_dir)
    
    if not band_sets:
        print("❌ No Sentinel-2 band files found!")
        print("   Expected format: *_Sentinel-2_L2A_B##_(Raw).tiff")
        return
    
    print(f"✅ Found {len(band_sets)} image set(s)\n")
    
    # Process each set
    stacked_images = []
    results = []
    
    for i, (date_prefix, bands) in enumerate(band_sets.items(), 1):
        print(f"📅 Processing set {i}/{len(band_sets)}: {date_prefix[:50]}...")
        print(f"   Bands found: {sorted(bands.keys())}")
        
        # Stack bands
        stacked_path = output_dir / 'stacked' / f'{date_prefix[:30]}_stacked.tif'
        
        if stack_bands(bands, stacked_path):
            print(f"   ✅ Stacked: {stacked_path.name}")
            stacked_images.append(stacked_path)
            
            if not args.stack_only:
                # Run prediction
                print(f"   🔮 Running prediction...")
                try:
                    mask, probs = run_prediction(
                        stacked_path, model_path, 
                        output_dir / 'predictions',
                        threshold=args.threshold
                    )
                    results.append({
                        'date': date_prefix,
                        'mask': mask,
                        'probs': probs
                    })
                    print(f"   ✅ Debris detected: {mask.sum():,} pixels ({mask.mean()*100:.2f}%)")
                except Exception as e:
                    print(f"   ❌ Prediction failed: {e}")
        else:
            print(f"   ❌ Stacking failed")
        
        print()
    
    # Temporal analysis if multiple images
    if len(results) > 1:
        print("📊 Running multi-temporal analysis...")
        combine_temporal_results(results, output_dir)
        print(f"   ✅ Saved to: {output_dir / 'temporal_analysis.png'}")
    
    print("\n" + "="*50)
    print("✅ Batch processing complete!")
    print(f"   Stacked images: {len(stacked_images)}")
    print(f"   Predictions: {len(results)}")
    print(f"   Output directory: {output_dir}")


if __name__ == '__main__':
    main()
