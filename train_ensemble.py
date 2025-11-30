"""
Multi-Model Ensemble Training for Marine Debris Detection

This script trains multiple deep learning models and combines them for best results:
1. Train individual models (MLP, CNN, Transformer, Attention, etc.)
2. Evaluate each model on validation set
3. Create weighted ensemble based on performance
4. Fine-tune ensemble for maximum F1 score

Designed for local GPU training with automatic mixed precision.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.csv_dataset import CSVPixelDataset
from models.classification_models import get_classification_model, EnsembleModel


# Class names for reporting
CLASS_NAMES = ['Water', 'Plastic', 'Driftwood', 'Seaweed', 'Pumice', 'Sea Snot', 'Sea Foam']


class MultiModelTrainer:
    """
    Train multiple models and create an optimized ensemble.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        models_to_train: List[str] = None,
        device: str = 'auto',
        precision: str = '16-mixed'
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.precision = precision
        
        # Default models to train
        if models_to_train is None:
            self.models_to_train = [
                'mlp', 'cnn1d', 'transformer', 'attention', 
                'senet', 'resnet', 'hybrid'
            ]
        else:
            self.models_to_train = models_to_train
        
        # Store results
        self.model_results: Dict[str, dict] = {}
        self.trained_models: Dict[str, nn.Module] = {}
        
    def load_data(self, add_indices: bool = True, val_split: float = 0.15, test_split: float = 0.15):
        """Load and split the CSV dataset."""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Try to load train.csv first
        train_path = self.data_dir / 'train.csv'
        test_path = self.data_dir / 'test.csv'
        
        if train_path.exists():
            print(f"Loading training data from: {train_path}")
            full_dataset = CSVPixelDataset(str(train_path), add_indices=add_indices)
            
            if test_path.exists():
                print(f"Loading test data from: {test_path}")
                self.test_dataset = CSVPixelDataset(str(test_path), add_indices=add_indices)
            else:
                self.test_dataset = None
        else:
            # Use all_data.csv or balanced_data.csv
            data_path = self.data_dir / 'balanced_data.csv'
            if not data_path.exists():
                data_path = self.data_dir / 'all_data.csv'
            
            print(f"Loading data from: {data_path}")
            full_dataset = CSVPixelDataset(str(data_path), add_indices=add_indices)
            self.test_dataset = None
        
        # Get number of features
        sample_x, _ = full_dataset[0]
        self.n_features = sample_x.shape[0]
        self.n_classes = 7
        
        print(f"Total samples: {len(full_dataset):,}")
        print(f"Features: {self.n_features} (11 bands + {self.n_features - 11} indices)")
        
        # Split data
        n_total = len(full_dataset)
        n_test = int(n_total * test_split) if self.test_dataset is None else 0
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test
        
        if n_test > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
        
        print(f"Train: {len(self.train_dataset):,} | Val: {len(self.val_dataset):,} | Test: {len(self.test_dataset) if self.test_dataset else 0:,}")
        
        return self
    
    def create_dataloaders(self, batch_size: int = 256, num_workers: int = 4):
        """Create data loaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.test_loader = None
        
        return self
    
    def train_single_model(
        self,
        model_name: str,
        epochs: int = 50,
        lr: float = 1e-3,
        early_stopping_patience: int = 10
    ) -> Tuple[nn.Module, dict]:
        """Train a single model and return it with results."""
        
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Create model
        model = get_classification_model(model_name, self.n_features, self.n_classes)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        
        # Create Lightning module
        module = PixelClassificationModule(
            model=model,
            num_classes=self.n_classes,
            learning_rate=lr
        )
        
        # Callbacks
        checkpoint_dir = self.output_dir / 'checkpoints' / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename='best-{epoch:02d}-{val_f1_macro:.4f}',
                monitor='val_f1_macro',
                mode='max',
                save_top_k=1
            ),
            EarlyStopping(
                monitor='val_f1_macro',
                patience=early_stopping_patience,
                mode='max',
                verbose=False
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Logger
        logger = TensorBoardLogger(
            str(self.output_dir / 'logs'),
            name=model_name
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.device,
            precision=self.precision,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=20,
            gradient_clip_val=1.0
        )
        
        # Train
        trainer.fit(module, self.train_loader, self.val_loader)
        
        # Load best checkpoint
        best_model_path = callbacks[0].best_model_path
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            module.load_state_dict(checkpoint['state_dict'])
        
        # Evaluate on validation set
        val_metrics = self.evaluate_model(module.model, self.val_loader)
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
        print(f"  F1 (macro):  {val_metrics['f1_macro']:.4f}")
        print(f"  F1 (weight): {val_metrics['f1_weighted']:.4f}")
        print(f"  Precision:   {val_metrics['precision']:.4f}")
        print(f"  Recall:      {val_metrics['recall']:.4f}")
        
        return module.model, val_metrics
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> dict:
        """Evaluate a model on a dataloader."""
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        return {
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train_all_models(self, epochs: int = 50, lr: float = 1e-3):
        """Train all specified models."""
        
        print("\n" + "="*60)
        print("MULTI-MODEL TRAINING")
        print("="*60)
        print(f"Models to train: {', '.join(self.models_to_train)}")
        
        for model_name in self.models_to_train:
            model, metrics = self.train_single_model(
                model_name, 
                epochs=epochs, 
                lr=lr
            )
            
            self.trained_models[model_name] = model
            self.model_results[model_name] = metrics
        
        return self
    
    def create_ensemble(self, top_k: int = 3) -> nn.Module:
        """Create ensemble from top-k performing models."""
        
        print("\n" + "="*60)
        print("CREATING ENSEMBLE")
        print("="*60)
        
        # Sort models by F1 score
        sorted_models = sorted(
            self.model_results.items(),
            key=lambda x: x[1]['f1_macro'],
            reverse=True
        )
        
        print("\nModel Rankings (by F1 macro):")
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"  {i}. {name}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Select top-k models
        top_models = sorted_models[:top_k]
        print(f"\nSelected top {top_k} models for ensemble:")
        
        ensemble_models = []
        weights = []
        for name, metrics in top_models:
            print(f"  - {name} (F1={metrics['f1_macro']:.4f})")
            ensemble_models.append(self.trained_models[name])
            weights.append(metrics['f1_macro'])  # Use F1 as weight
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_models, self.n_classes, learnable_weights=True)
        
        # Initialize weights based on performance
        with torch.no_grad():
            weights_tensor = torch.tensor(weights, dtype=torch.float)
            weights_tensor = weights_tensor / weights_tensor.sum()
            ensemble.weights.data = weights_tensor
        
        return ensemble
    
    def fine_tune_ensemble(self, ensemble: nn.Module, epochs: int = 20, lr: float = 1e-4) -> dict:
        """Fine-tune the ensemble on training data."""
        
        print("\n" + "="*60)
        print("FINE-TUNING ENSEMBLE")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble = ensemble.to(device)
        
        # Only train ensemble weights, freeze base models
        for model in ensemble.models:
            for param in model.parameters():
                param.requires_grad = False
        
        optimizer = optim.Adam([ensemble.weights], lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        best_weights = ensemble.weights.data.clone()
        
        for epoch in range(epochs):
            ensemble.train()
            total_loss = 0
            
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                logits = ensemble(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluate
            val_metrics = self.evaluate_model(ensemble, self.val_loader)
            
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                best_weights = ensemble.weights.data.clone()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(self.train_loader):.4f}, "
                      f"Val F1={val_metrics['f1_macro']:.4f}, Best F1={best_f1:.4f}")
        
        # Restore best weights
        ensemble.weights.data = best_weights
        
        # Final evaluation
        final_metrics = self.evaluate_model(ensemble, self.val_loader)
        
        print(f"\nFinal Ensemble Performance:")
        print(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
        print(f"  F1 (macro):  {final_metrics['f1_macro']:.4f}")
        print(f"  F1 (weight): {final_metrics['f1_weighted']:.4f}")
        print(f"  Precision:   {final_metrics['precision']:.4f}")
        print(f"  Recall:      {final_metrics['recall']:.4f}")
        
        return final_metrics
    
    def evaluate_on_test(self, model: nn.Module) -> dict:
        """Evaluate model on test set."""
        if self.test_loader is None:
            print("No test set available")
            return {}
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        
        metrics = self.evaluate_model(model, self.test_loader)
        
        print(f"\nTest Results:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
        print(f"  F1 (weight): {metrics['f1_weighted']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            metrics['targets'], 
            metrics['predictions'],
            target_names=CLASS_NAMES[:max(metrics['targets'])+1],
            zero_division=0
        ))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(metrics['targets'], metrics['predictions'])
        print(cm)
        
        return metrics
    
    def save_results(self, ensemble: nn.Module, final_metrics: dict):
        """Save all results and models."""
        
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save ensemble model
        model_path = self.output_dir / 'ensemble_model.pt'
        torch.save({
            'model_state_dict': ensemble.state_dict(),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'model_names': list(self.trained_models.keys())
        }, model_path)
        print(f"Ensemble saved: {model_path}")
        
        # Save individual model results
        results = {
            'individual_models': {
                name: {k: v for k, v in metrics.items() if k not in ['predictions', 'targets']}
                for name, metrics in self.model_results.items()
            },
            'ensemble_metrics': {k: v for k, v in final_metrics.items() if k not in ['predictions', 'targets']},
            'ensemble_weights': ensemble.weights.data.tolist(),
            'n_features': self.n_features,
            'n_classes': self.n_classes
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {results_path}")
        
        # Print summary table
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"\n{'Model':<15} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weight':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 70)
        
        for name, metrics in sorted(self.model_results.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
            print(f"{name:<15} {metrics['accuracy']:>10.4f} {metrics['f1_macro']:>10.4f} "
                  f"{metrics['f1_weighted']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f}")
        
        print("-" * 70)
        print(f"{'ENSEMBLE':<15} {final_metrics['accuracy']:>10.4f} {final_metrics['f1_macro']:>10.4f} "
              f"{final_metrics['f1_weighted']:>10.4f} {final_metrics['precision']:>10.4f} {final_metrics['recall']:>10.4f}")


class PixelClassificationModule(pl.LightningModule):
    """PyTorch Lightning module for pixel classification."""
    
    def __init__(self, model, num_classes=7, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        
        self.val_preds = []
        self.val_targets = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            preds = torch.cat(self.val_preds).numpy()
            targets = torch.cat(self.val_targets).numpy()
            
            acc = accuracy_score(targets, preds)
            f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
            
            self.log('val_acc', acc, prog_bar=True)
            self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        self.val_preds.clear()
        self.val_targets.clear()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 50
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble Training')
    
    parser.add_argument('--data-dir', default='datasets/csv_data',
                       help='Path to CSV data directory')
    parser.add_argument('--output-dir', default='output/ensemble',
                       help='Output directory')
    parser.add_argument('--models', nargs='+', 
                       default=['mlp', 'cnn1d', 'transformer', 'attention', 'senet', 'resnet', 'hybrid'],
                       help='Models to train')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--top-k', type=int, default=3, help='Top-k models for ensemble')
    parser.add_argument('--device', default='auto', help='Device')
    parser.add_argument('--precision', default='16-mixed', help='Training precision')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    
    args = parser.parse_args()
    
    # Check GPU
    print("="*60)
    print("MARINE DEBRIS MULTI-MODEL TRAINING")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nNo GPU available, using CPU")
        args.precision = '32'
    
    # Initialize trainer
    trainer = MultiModelTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_to_train=args.models,
        device=args.device,
        precision=args.precision
    )
    
    # Load data
    trainer.load_data(add_indices=True)
    trainer.create_dataloaders(batch_size=args.batch_size, num_workers=args.workers)
    
    # Train all models
    trainer.train_all_models(epochs=args.epochs, lr=args.lr)
    
    # Create and fine-tune ensemble
    ensemble = trainer.create_ensemble(top_k=args.top_k)
    final_metrics = trainer.fine_tune_ensemble(ensemble, epochs=20)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_on_test(ensemble)
    
    # Save everything
    trainer.save_results(ensemble, final_metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
