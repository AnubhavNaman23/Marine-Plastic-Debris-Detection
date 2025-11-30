"""
Training Script for Pixel-Level Classification on CSV Data

This script trains classification models (MLP, CNN, Transformer, Attention, etc.)
on the Floating-Marine-Debris-Data CSV dataset.

Models classify individual pixels based on spectral bands and indices.

Usage:
    python train_classifier.py --data-dir path/to/csv/data --model transformer --epochs 100
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
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.csv_dataset import FloatingDebrisCSVDataset, load_csv_datasets, create_csv_dataloaders
from models.classification_models import get_classification_model


class PixelClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning module for pixel-level classification.
    """
    def __init__(
        self,
        model,
        num_classes=7,
        learning_rate=1e-3,
        weight_decay=1e-4,
        class_weights=None,
        scheduler='cosine'
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.num_classes = num_classes
        
        # Loss with optional class weights
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # For metrics
        self.save_hyperparameters(ignore=['model'])
        
        # Store predictions for epoch-level metrics
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())
        
        self.log('val/loss', loss, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            preds = torch.cat(self.val_preds).numpy()
            targets = torch.cat(self.val_targets).numpy()
            
            acc = accuracy_score(targets, preds)
            f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
            f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)
            
            self.log('val/acc', acc, prog_bar=True)
            self.log('val/f1_macro', f1_macro, prog_bar=True)
            self.log('val/f1_weighted', f1_weighted)
            
        self.val_preds.clear()
        self.val_targets.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.test_preds.append(preds.cpu())
        self.test_targets.append(y.cpu())
        
        return loss
    
    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()
            
            acc = accuracy_score(targets, preds)
            f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
            f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)
            
            self.log('test/acc', acc)
            self.log('test/f1_macro', f1_macro)
            self.log('test/f1_weighted', f1_weighted)
            
            # Print classification report
            print("\n" + "="*60)
            print("TEST RESULTS")
            print("="*60)
            print(f"\nAccuracy: {acc:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
            print(f"F1 (weighted): {f1_weighted:.4f}")
            print("\nClassification Report:")
            class_names = ['Water', 'Plastic', 'Driftwood', 'Seaweed', 'Pumice', 'Sea Snot', 'Sea Foam']
            print(classification_report(targets, preds, target_names=class_names[:max(targets)+1], zero_division=0))
            
        self.test_preds.clear()
        self.test_targets.clear()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=1e-6
            )
        elif self.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/f1_macro'
                }
            }
        else:
            return optimizer
        
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='Train Pixel Classification Model')
    
    # Data arguments
    parser.add_argument('--data-dir', required=True, 
                       help='Path to CSV data directory (containing train.csv, test.csv)')
    parser.add_argument('--train-file', default='train.csv', help='Training CSV file')
    parser.add_argument('--test-file', default='test.csv', help='Test CSV file')
    parser.add_argument('--use-all-data', action='store_true',
                       help='Use all_data.csv instead of train/test split')
    parser.add_argument('--use-balanced', action='store_true',
                       help='Use balanced_data.csv')
    
    # Model arguments
    parser.add_argument('--model', default='transformer',
                       choices=['mlp', 'cnn1d', 'cnn2d', 'transformer', 'attention', 
                               'senet', 'resnet', 'hybrid', 'ensemble'],
                       help='Model architecture')
    parser.add_argument('--add-indices', action='store_true', default=True,
                       help='Add spectral indices to features')
    parser.add_argument('--no-indices', action='store_true',
                       help='Do not add spectral indices')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--val-split', type=float, default=0.2, 
                       help='Validation split ratio (if using all_data)')
    
    # Output arguments
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--name', default='classifier', help='Experiment name')
    
    # Hardware arguments
    parser.add_argument('--device', default='auto', help='Device (auto/gpu/cpu)')
    parser.add_argument('--precision', default='32', 
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    
    args = parser.parse_args()
    
    # Set paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.name}_{args.model}_{timestamp}"
    checkpoint_dir = output_dir / 'checkpoints' / exp_name
    log_dir = output_dir / 'logs'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    add_indices = args.add_indices and not args.no_indices
    
    print("="*60)
    print("PIXEL CLASSIFICATION TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data dir:      {data_dir}")
    print(f"  Model:         {args.model}")
    print(f"  Add indices:   {add_indices}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler:     {args.scheduler}")
    print(f"  Output:        {checkpoint_dir}")
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU: Not available, using CPU")
    print()
    
    # Load data
    print("Loading datasets...")
    
    if args.use_all_data or args.use_balanced:
        # Use single file with train/val split
        csv_file = 'balanced_data.csv' if args.use_balanced else 'all_data.csv'
        csv_path = data_dir / csv_file
        
        if not csv_path.exists():
            print(f"ERROR: Data file not found: {csv_path}")
            sys.exit(1)
        
        full_dataset = FloatingDebrisCSVDataset(str(csv_path), add_indices=add_indices)
        
        # Split into train/val/test
        n_total = len(full_dataset)
        n_test = int(n_total * 0.15)
        n_val = int(n_total * 0.15)
        n_train = n_total - n_val - n_test
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        # Use separate train/test files
        train_path = data_dir / args.train_file
        test_path = data_dir / args.test_file
        
        if not train_path.exists():
            print(f"ERROR: Training file not found: {train_path}")
            sys.exit(1)
        
        full_train = FloatingDebrisCSVDataset(str(train_path), add_indices=add_indices)
        
        # Split train into train/val
        n_total = len(full_train)
        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val
        
        train_dataset, val_dataset = random_split(
            full_train,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        if test_path.exists():
            test_dataset = FloatingDebrisCSVDataset(str(test_path), add_indices=add_indices)
        else:
            test_dataset = None
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")
    if test_dataset:
        print(f"  Test samples:  {len(test_dataset):,}")
    
    # Get number of features
    sample_x, _ = train_dataset[0]
    n_features = sample_x.shape[0]
    print(f"  Features:      {n_features}")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    
    # Create model
    print("Creating model...")
    n_classes = 7  # Water, Plastic, Driftwood, Seaweed, Pumice, Sea Snot, Sea Foam
    
    model = get_classification_model(args.model, n_features, n_classes)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model:      {args.model}")
    print(f"  Parameters: {params:,}")
    print()
    
    # Create Lightning module
    module = PixelClassificationModule(
        model=model,
        num_classes=n_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='best-{epoch:02d}-{val/f1_macro:.4f}',
            monitor='val/f1_macro',
            mode='max',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val/f1_macro',
            patience=15,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Logger
    logger = TensorBoardLogger(str(log_dir), name=exp_name)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0
    )
    
    # Train
    print("Starting training...")
    print("="*60)
    trainer.fit(module, train_loader, val_loader)
    print("="*60)
    
    # Test
    if test_loader:
        print("\nRunning test evaluation...")
        trainer.test(module, test_loader)
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': args.model,
        'n_features': n_features,
        'n_classes': n_classes,
        'add_indices': add_indices
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save config
    config = {
        'model': args.model,
        'n_features': n_features,
        'n_classes': n_classes,
        'add_indices': add_indices,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs:        {log_dir}/{exp_name}")
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")


if __name__ == '__main__':
    main()
