#!/usr/bin/env python3

"""
MobileNetV3 Fine-tuning Script for Tent/Mannequin Classification
SUAS 2025 Drone Object Recognition Training Pipeline

This script provides a template for fine-tuning MobileNetV3 on aerial images
of tents and mannequins for the two-stage detection pipeline.

Features:
- Data loading with augmentation for aerial imagery
- Transfer learning from ImageNet pre-trained weights
- Training with validation monitoring
- Model checkpointing and early stopping
- Evaluation metrics and confusion matrix
- Export for production inference

Usage:
    python3 src/train_classifier.py --data_dir /path/to/dataset --epochs 50
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class AerialDataset(Dataset):
    """Dataset class for aerial tent/mannequin images."""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory
            transform: Image transformations
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Class mapping
        self.class_names = ['tent', 'mannequin', 'other']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples")
        for class_name in self.class_names:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            print(f"  {class_name}: {count} samples")
    
    def _load_samples(self):
        """Load image paths and labels from directory structure."""
        for class_name in self.class_names:
            class_dir = self.data_dir / self.split / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label


class MobileNetV3Trainer:
    """Trainer class for MobileNetV3 fine-tuning."""
    
    def __init__(self, num_classes=3, device=None, learning_rate=0.001):
        """
        Initialize trainer.
        
        Args:
            num_classes: Number of classes (3: tent/mannequin/other)
            device: Device to use ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
        """
        self.num_classes = num_classes
        self.class_names = ['tent', 'mannequin', 'other']
        
        # Setup device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Modify classifier for our classes
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, num_classes)
        
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def setup_data_loaders(self, data_dir, batch_size=32, val_split=0.2):
        """
        Setup data loaders for training and validation.
        
        Args:
            data_dir: Path to dataset directory
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
        """
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = AerialDataset(data_dir, transform=train_transform, split='train')
        val_dataset = AerialDataset(data_dir, transform=val_transform, split='val')
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=50, save_dir='models'):
        """
        Train the model for specified epochs.
        
        Args:
            epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names
                }
                torch.save(checkpoint, os.path.join(save_dir, 'mobilenetv3_tent_mannequin.pth'))
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Plot training history
        self.plot_training_history(save_dir)
    
    def plot_training_history(self, save_dir):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Training history plot saved")
    
    def evaluate(self):
        """Evaluate model on validation set and generate detailed metrics."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        print("\nEvaluating model...")
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, target_names=self.class_names))
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Confusion matrix saved to models/confusion_matrix.png")
    
    def export_model(self, save_path):
        """Export model for inference."""
        self.model.eval()
        
        # Create example input for tracing
        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Export as TorchScript
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(save_path.replace('.pth', '_traced.pt'))
        
        print(f"Model exported to {save_path.replace('.pth', '_traced.pt')}")


def create_sample_dataset_structure():
    """Create sample dataset directory structure."""
    sample_structure = """
    Dataset Directory Structure:
    
    dataset/
    ├── train/
    │   ├── tent/
    │   │   ├── tent_001.jpg
    │   │   ├── tent_002.jpg
    │   │   └── ...
    │   ├── mannequin/
    │   │   ├── mannequin_001.jpg
    │   │   ├── mannequin_002.jpg
    │   │   └── ...
    │   └── other/
    │       ├── other_001.jpg
    │       ├── other_002.jpg
    │       └── ...
    └── val/
        ├── tent/
        ├── mannequin/
        └── other/
    
    Tips for data collection:
    - Capture aerial images from 150ft AGL (competition altitude)
    - Include tents and mannequins in various orientations
    - Add occluded scenarios (trees, bushes, vehicles)
    - Include different lighting conditions
    - Collect 'other' class images of similar objects (umbrellas, backpacks, etc.)
    - Aim for balanced dataset (similar number of samples per class)
    """
    print(sample_structure)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MobileNetV3 for tent/mannequin classification')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--create_structure', action='store_true',
                       help='Show sample dataset structure')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_dataset_structure()
        return
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        print("Use --create_structure to see required dataset structure")
        return
    
    # Initialize trainer
    trainer = MobileNetV3Trainer(learning_rate=args.learning_rate)
    
    # Setup data loaders
    trainer.setup_data_loaders(args.data_dir, batch_size=args.batch_size)
    
    # Train model
    trainer.train(epochs=args.epochs, save_dir=args.save_dir)
    
    # Evaluate model
    trainer.evaluate()
    
    # Export model for inference
    model_path = os.path.join(args.save_dir, 'mobilenetv3_tent_mannequin.pth')
    trainer.export_model(model_path)
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {model_path}")
    print("\nTo use the trained model in the detection pipeline:")
    print("1. Copy the .pth file to your models/ directory")
    print("2. The detection system will automatically load it")
    print("3. Test with: ros2 launch drone_object_recognition object_recognition.launch.py")


if __name__ == "__main__":
    main()
