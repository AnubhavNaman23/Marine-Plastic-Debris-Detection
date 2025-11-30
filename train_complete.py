"""
Complete Training Pipeline for Marine Debris Detection

This is the MAIN training script that combines:
1. Image Segmentation Models (UNet, UNet++, etc.) trained on MARIDA/FloatingObjects
2. Pixel Classification Models (Transformer, CNN, etc.) trained on spectral CSV data
3. Ensemble Methods for combining predictions

Designed for local GPU training with high accuracy and F1 score optimization.

Usage:
    # Train both segmentation and classification models
    python train_complete.py --marida-path path/to/MARIDA --csv-path datasets/csv_data
    
    # Train only classification models
    python train_complete.py --csv-path datasets/csv_data --skip-segmentation
    
    # Train only segmentation models
    python train_complete.py --marida-path path/to/MARIDA --skip-classification
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import local modules
from train_ensemble import MultiModelTrainer


def print_header():
    """Print training header."""
    print("\n" + "="*70)
    print("🌊 MARINE DEBRIS DETECTOR - COMPLETE TRAINING PIPELINE 🌊")
    print("="*70)
    print("""
    This training pipeline combines multiple approaches:
    
    1. SEGMENTATION MODELS (for satellite image analysis)
       - UNet++, UNet, DeepLabV3+, FPN
       - Trained on MARIDA and FloatingObjects datasets
       - Outputs debris location masks
    
    2. CLASSIFICATION MODELS (for pixel-level detection)
       - Transformer, Attention, CNN, MLP, ResNet, Hybrid
       - Trained on spectral CSV data (11 bands + 24 indices)
       - Classifies: Water, Plastic, Driftwood, Seaweed, Pumice, Sea Snot, Sea Foam
    
    3. ENSEMBLE METHODS (for maximum accuracy)
       - Weighted voting from top-performing models
       - Fine-tuned ensemble weights
       - Optimized for F1 score
    """)
    print("="*70)


def check_gpu():
    """Check and print GPU information."""
    print("\n📊 HARDWARE CONFIGURATION")
    print("-"*40)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
        
        # Recommend batch size based on GPU memory
        if gpu_memory >= 16:
            recommended_batch = 32
        elif gpu_memory >= 8:
            recommended_batch = 16
        elif gpu_memory >= 4:
            recommended_batch = 8
        else:
            recommended_batch = 4
        print(f"  Recommended batch size: {recommended_batch}")
        
        return True, recommended_batch
    else:
        print("  GPU: Not available")
        print("  Using: CPU (training will be slower)")
        return False, 16


def train_segmentation_models(
    marida_path: str,
    output_dir: Path,
    architectures: list = ['unetpp', 'unet'],
    encoders: list = ['resnet34'],
    epochs: int = 50,
    batch_size: int = 8,
    device: str = 'auto',
    precision: str = '16-mixed'
):
    """Train segmentation models on MARIDA dataset."""
    
    print("\n" + "="*70)
    print("🎯 SEGMENTATION MODEL TRAINING")
    print("="*70)
    
    from datasets.marida_dataset import create_marida_dataloaders
    from training.lightning_module import MarineDebrisModule
    
    marida_path = Path(marida_path)
    splits_dir = marida_path.parent / 'splits'
    
    if not marida_path.exists():
        print(f"⚠️  MARIDA path not found: {marida_path}")
        print("   Skipping segmentation training...")
        return {}
    
    # Create dataloaders
    print(f"\nLoading MARIDA dataset from: {marida_path}")
    train_loader, val_loader, test_loader = create_marida_dataloaders(
        data_root=str(marida_path),
        splits_dir=str(splits_dir),
        batch_size=batch_size,
        num_workers=4,
        binary=True,
        patch_size=256
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    results = {}
    
    for arch in architectures:
        for encoder in encoders:
            model_name = f"{arch}_{encoder}"
            print(f"\n{'='*50}")
            print(f"Training: {model_name}")
            print(f"{'='*50}")
            
            # Create model
            arch_map = {
                'unet': smp.Unet,
                'unetpp': smp.UnetPlusPlus,
                'deeplabv3plus': smp.DeepLabV3Plus,
                'fpn': smp.FPN
            }
            
            model = arch_map[arch](
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=12,
                classes=2
            )
            
            params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {params:,}")
            
            # Create Lightning module
            module = MarineDebrisModule(
                model=model,
                learning_rate=1e-4,
                num_epochs=epochs
            )
            
            # Callbacks
            checkpoint_dir = output_dir / 'segmentation' / model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks = [
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename='best-{epoch:02d}-{val/f1:.4f}',
                    monitor='val/f1',
                    mode='max',
                    save_top_k=1
                ),
                EarlyStopping(
                    monitor='val/f1',
                    patience=10,
                    mode='max'
                ),
                LearningRateMonitor()
            ]
            
            logger = TensorBoardLogger(str(output_dir / 'logs'), name=f'seg_{model_name}')
            
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator=device,
                precision=precision,
                callbacks=callbacks,
                logger=logger,
                gradient_clip_val=1.0
            )
            
            # Train
            trainer.fit(module, train_loader, val_loader)
            
            # Get best metrics
            best_f1 = callbacks[0].best_model_score
            results[model_name] = {
                'architecture': arch,
                'encoder': encoder,
                'best_f1': float(best_f1) if best_f1 else 0,
                'checkpoint': str(checkpoint_dir)
            }
            
            # Save model
            torch.save(model.state_dict(), checkpoint_dir / 'final_model.pt')
    
    return results


def train_classification_models(
    csv_path: str,
    output_dir: Path,
    models: list = ['transformer', 'attention', 'cnn1d', 'resnet', 'hybrid'],
    epochs: int = 50,
    batch_size: int = 256,
    device: str = 'auto',
    precision: str = '16-mixed'
):
    """Train classification models on CSV spectral data."""
    
    print("\n" + "="*70)
    print("🔬 CLASSIFICATION MODEL TRAINING")
    print("="*70)
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"⚠️  CSV data path not found: {csv_path}")
        print("   Skipping classification training...")
        return {}
    
    # Initialize multi-model trainer
    trainer = MultiModelTrainer(
        data_dir=str(csv_path),
        output_dir=str(output_dir / 'classification'),
        models_to_train=models,
        device=device,
        precision=precision
    )
    
    # Load data
    trainer.load_data(add_indices=True)
    trainer.create_dataloaders(batch_size=batch_size)
    
    # Train all models
    trainer.train_all_models(epochs=epochs)
    
    # Create ensemble
    ensemble = trainer.create_ensemble(top_k=3)
    final_metrics = trainer.fine_tune_ensemble(ensemble)
    
    # Evaluate on test
    test_metrics = trainer.evaluate_on_test(ensemble)
    
    # Save results
    trainer.save_results(ensemble, final_metrics)
    
    return {
        'individual_results': trainer.model_results,
        'ensemble_metrics': final_metrics,
        'test_metrics': test_metrics
    }


def print_final_summary(seg_results: dict, cls_results: dict, output_dir: Path):
    """Print final training summary."""
    
    print("\n" + "="*70)
    print("📊 FINAL TRAINING SUMMARY")
    print("="*70)
    
    # Segmentation results
    if seg_results:
        print("\n🎯 SEGMENTATION MODELS:")
        print("-"*50)
        print(f"{'Model':<25} {'Best F1':>10}")
        print("-"*50)
        for name, results in sorted(seg_results.items(), key=lambda x: x[1]['best_f1'], reverse=True):
            print(f"{name:<25} {results['best_f1']:>10.4f}")
    
    # Classification results  
    if cls_results and 'individual_results' in cls_results:
        print("\n🔬 CLASSIFICATION MODELS:")
        print("-"*50)
        print(f"{'Model':<15} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}")
        print("-"*50)
        
        for name, metrics in sorted(
            cls_results['individual_results'].items(), 
            key=lambda x: x[1].get('f1_macro', 0), 
            reverse=True
        ):
            print(f"{name:<15} {metrics.get('accuracy', 0):>10.4f} {metrics.get('f1_macro', 0):>10.4f} "
                  f"{metrics.get('precision', 0):>10.4f} {metrics.get('recall', 0):>10.4f}")
        
        if 'ensemble_metrics' in cls_results:
            em = cls_results['ensemble_metrics']
            print("-"*50)
            print(f"{'ENSEMBLE':<15} {em.get('accuracy', 0):>10.4f} {em.get('f1_macro', 0):>10.4f} "
                  f"{em.get('precision', 0):>10.4f} {em.get('recall', 0):>10.4f}")
    
    print("\n" + "="*70)
    print("📁 OUTPUT FILES:")
    print("-"*50)
    print(f"  Checkpoints: {output_dir / 'checkpoints'}")
    print(f"  Logs:        {output_dir / 'logs'}")
    print(f"  Results:     {output_dir / 'training_results.json'}")
    print("\n  To view TensorBoard logs:")
    print(f"    tensorboard --logdir {output_dir / 'logs'}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Complete Marine Debris Detection Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both segmentation and classification
  python train_complete.py --marida-path ../MARIDA/patches --csv-path datasets/csv_data
  
  # Train only classification models
  python train_complete.py --csv-path datasets/csv_data --skip-segmentation
  
  # Train with specific models
  python train_complete.py --csv-path datasets/csv_data --cls-models transformer attention hybrid
        """
    )
    
    # Data paths
    parser.add_argument('--marida-path', default=None,
                       help='Path to MARIDA/patches directory')
    parser.add_argument('--csv-path', default='datasets/csv_data',
                       help='Path to CSV data directory')
    
    # Skip options
    parser.add_argument('--skip-segmentation', action='store_true',
                       help='Skip segmentation model training')
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip classification model training')
    
    # Model selection
    parser.add_argument('--seg-archs', nargs='+', default=['unetpp'],
                       help='Segmentation architectures')
    parser.add_argument('--seg-encoders', nargs='+', default=['resnet34'],
                       help='Segmentation encoders')
    parser.add_argument('--cls-models', nargs='+', 
                       default=['transformer', 'attention', 'cnn1d', 'resnet', 'hybrid'],
                       help='Classification models')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--seg-batch-size', type=int, default=8, help='Segmentation batch size')
    parser.add_argument('--cls-batch-size', type=int, default=256, help='Classification batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Output
    parser.add_argument('--output-dir', default='output',
                       help='Output directory')
    
    # Hardware
    parser.add_argument('--device', default='auto', help='Device (auto/gpu/cpu)')
    parser.add_argument('--precision', default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Check GPU
    has_gpu, recommended_batch = check_gpu()
    
    if not has_gpu:
        args.precision = '32'
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Training results
    seg_results = {}
    cls_results = {}
    
    # Train segmentation models
    if not args.skip_segmentation and args.marida_path:
        seg_results = train_segmentation_models(
            marida_path=args.marida_path,
            output_dir=output_dir,
            architectures=args.seg_archs,
            encoders=args.seg_encoders,
            epochs=args.epochs,
            batch_size=args.seg_batch_size,
            device=args.device,
            precision=args.precision
        )
    
    # Train classification models
    if not args.skip_classification:
        cls_results = train_classification_models(
            csv_path=args.csv_path,
            output_dir=output_dir,
            models=args.cls_models,
            epochs=args.epochs,
            batch_size=args.cls_batch_size,
            device=args.device,
            precision=args.precision
        )
    
    # Save combined results
    all_results = {
        'segmentation': seg_results,
        'classification': cls_results,
        'timestamp': timestamp,
        'config': vars(args)
    }
    
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print_final_summary(seg_results, cls_results, output_dir)
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"   Results saved to: {results_path}")


if __name__ == '__main__':
    main()
