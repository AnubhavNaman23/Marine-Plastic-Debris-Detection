"""
Local Training Script for Marine Debris Detection

For quick local training with MARIDA dataset.
For full training, use the Colab notebook.

Usage:
    python train.py --data-root path/to/MARIDA --epochs 50 --batch-size 8
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.marida_dataset import MARIDADataset, create_marida_dataloaders
from training.lightning_module import MarineDebrisModule, create_trainer


def main():
    parser = argparse.ArgumentParser(description='Train Marine Debris Detector')
    
    # Data arguments
    parser.add_argument('--data-root', required=True, help='Path to MARIDA/patches')
    parser.add_argument('--splits-dir', help='Path to MARIDA/splits (default: data-root/../splits)')
    
    # Model arguments
    parser.add_argument('--architecture', default='unetpp', 
                       choices=['unet', 'unetpp', 'deeplabv3plus', 'fpn'],
                       help='Model architecture')
    parser.add_argument('--encoder', default='resnet34',
                       help='Encoder backbone')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use ImageNet pretrained encoder')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patch-size', type=int, default=256, help='Patch size')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    
    # Output arguments
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--name', default='marine_debris', help='Experiment name')
    
    # Hardware arguments
    parser.add_argument('--device', default='auto', help='Device (auto/gpu/cpu)')
    parser.add_argument('--precision', default='16-mixed', 
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    
    args = parser.parse_args()
    
    # Set paths
    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir) if args.splits_dir else data_root.parent / 'splits'
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints' / args.name
    log_dir = output_dir / 'logs'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MARINE DEBRIS DETECTOR - LOCAL TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data root:     {data_root}")
    print(f"  Splits dir:    {splits_dir}")
    print(f"  Architecture:  {args.architecture}")
    print(f"  Encoder:       {args.encoder}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output:        {output_dir}")
    print()
    
    # Check data exists
    if not data_root.exists():
        print(f"ERROR: Data root not found: {data_root}")
        sys.exit(1)
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_marida_dataloaders(
        data_root=str(data_root),
        splits_dir=str(splits_dir),
        batch_size=args.batch_size,
        num_workers=args.workers,
        binary=True,
        patch_size=args.patch_size
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    arch_map = {
        'unet': smp.Unet,
        'unetpp': smp.UnetPlusPlus,
        'deeplabv3plus': smp.DeepLabV3Plus,
        'fpn': smp.FPN
    }
    
    model = arch_map[args.architecture](
        encoder_name=args.encoder,
        encoder_weights='imagenet' if args.pretrained else None,
        in_channels=12,
        classes=2
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture:  {args.architecture}")
    print(f"  Encoder:       {args.encoder}")
    print(f"  Parameters:    {params:,}")
    print()
    
    # Create Lightning module
    module = MarineDebrisModule(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='best-{epoch:02d}-{val/f1:.4f}',
            monitor='val/f1',
            mode='max',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val/f1',
            patience=10,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Logger
    logger = TensorBoardLogger(str(log_dir), name=args.name)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    # Train
    print("Starting training...")
    print("="*60)
    trainer.fit(module, train_loader, val_loader)
    print("="*60)
    print("Training complete!")
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save config
    import json
    config = {
        'architecture': args.architecture,
        'encoder': args.encoder,
        'in_channels': 12,
        'classes': 2,
        'patch_size': args.patch_size
    }
    config_path = checkpoint_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")
    
    # Test if test set available
    if test_loader and len(test_loader) > 0:
        print("\nRunning test evaluation...")
        trainer.test(module, test_loader)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs:     {log_dir}/{args.name}")
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")
    print(f"To predict:   python predict.py --image <path> --model {final_path}")


if __name__ == '__main__':
    main()
