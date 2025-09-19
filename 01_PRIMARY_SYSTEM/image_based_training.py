#!/usr/bin/env python3
"""
Enhanced Image-Based SASL Training System
GPU-Optimized Static Image Classification for South African Sign Language

Features:
- Efficient CNN architecture for static images
- Advanced data augmentation pipeline
- Mixed precision training for A10-12Q
- Comprehensive visualization system
- Memory-optimized data loading
- Class balancing and sampling strategies
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Training Configuration - Optimized for A10-12Q
BATCH_SIZE = 32  # Default - will be customizable
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # Default - will be customizable
IMAGE_SIZE = 224  # Standard ImageNet size
ENABLE_VALIDATION = True
VALIDATION_SPLIT = 0.2
MIXED_PRECISION = True  # Enable for faster training
NUM_WORKERS = 4  # Parallel data loading

# Device will be initialized in the main function

class SASLImageDataset(Dataset):
    """
    Optimized dataset loader for SASL image data
    Supports data augmentation and class balancing
    """
    def __init__(self, data_dir, class_names, transform=None, is_training=True):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.is_training = is_training
        
        # Build file lists and labels
        self.image_paths = []
        self.labels = []
        self.class_counts = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load all images and create balanced dataset"""
        print(f"Loading images from {self.data_dir}")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Class directory '{class_name}' not found, skipping...")
                continue
                
            # Find all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend([f for f in os.listdir(class_path) if f.lower().endswith(ext)])
            
            if len(image_files) == 0:
                print(f"Warning: No images found for class '{class_name}'")
                continue
                
            # Add to dataset
            for img_file in image_files:
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(class_idx)
                
            self.class_counts[class_name] = len(image_files)
            print(f"Loaded {len(image_files)} images for class '{class_name}'")
        
        print(f"Total dataset: {len(self.image_paths)} images across {len(self.class_counts)} classes")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image as fallback
            if self.transform:
                black_image = self.transform(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), 0))
            else:
                black_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return black_image, self.labels[idx]

class SASLImageCNN(nn.Module):
    """
    Efficient CNN architecture for SASL image classification
    Based on ResNet with custom head for sign language recognition
    """
    def __init__(self, num_classes, pretrained=True):
        super(SASLImageCNN, self).__init__()
        
        # Use ResNet18 as backbone (smaller and faster than ResNet50)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace classifier with custom head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Custom classification head optimized for sign language
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize custom layers with proper weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Extract features with backbone
        features = self.backbone(x)
        
        # Classify with custom head
        output = self.classifier(features)
        
        return output

def get_data_transforms(is_training=True):
    """
    Advanced data augmentation pipeline for sign language images
    """
    if is_training:
        # Training transforms with heavy augmentation
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms - minimal processing
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_weighted_sampler(dataset):
    """
    Create weighted sampler for balanced training across classes
    """
    # Handle both regular Dataset and Subset objects
    if hasattr(dataset, 'labels'):
        # Regular dataset
        labels = dataset.labels
    else:
        # Subset dataset - extract labels from the underlying dataset
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate weights inversely proportional to class frequency
    total_samples = len(labels)
    class_weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))

def plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir):
    """Generate comprehensive training history plots"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    if val_accs:
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"image_training_history_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved: {plot_path}")

def plot_accuracy_chart(train_accs, val_accs, output_dir):
    """Create a dedicated accuracy chart showing progression over epochs"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_accs) + 1)
    
    # Plot training accuracy
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=3, marker='o', markersize=4)
    
    # Plot validation accuracy if available
    if val_accs:
        plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=3, marker='s', markersize=4)
    
    # Customize the plot
    plt.title('Training Accuracy Progress', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add some statistics as text
    max_train_acc = max(train_accs)
    final_train_acc = train_accs[-1]
    
    stats_text = f'Final Training Accuracy: {final_train_acc:.2f}%\nBest Training Accuracy: {max_train_acc:.2f}%'
    
    if val_accs:
        max_val_acc = max(val_accs)
        final_val_acc = val_accs[-1]
        stats_text += f'\nFinal Validation Accuracy: {final_val_acc:.2f}%\nBest Validation Accuracy: {max_val_acc:.2f}%'
    
    # Position text box in the lower right
    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom', horizontalalignment='right')
    
    # Set y-axis to start from 0 and go to 100 for percentage
    plt.ylim(0, 100)
    
    # Add horizontal lines for reference
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Reference')
    plt.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% Reference')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Reference')
    
    plt.tight_layout()
    
    # Save the accuracy chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_chart_path = os.path.join(output_dir, f"accuracy_progress_{timestamp}.png")
    plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy progress chart saved: {accuracy_chart_path}")
    
    return accuracy_chart_path

def save_training_metrics(train_losses, train_accs, val_losses, val_accs, output_dir):
    """Save training metrics to CSV file for further analysis"""
    import csv
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"training_metrics_{timestamp}.csv")
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            if val_losses and val_accs:
                writer.writerow(['Epoch', 'Training_Loss', 'Training_Accuracy', 'Validation_Loss', 'Validation_Accuracy'])
            else:
                writer.writerow(['Epoch', 'Training_Loss', 'Training_Accuracy'])
            
            # Write data
            for i in range(len(train_losses)):
                epoch = i + 1
                row = [epoch, f"{train_losses[i]:.6f}", f"{train_accs[i]:.4f}"]
                
                if val_losses and val_accs and i < len(val_losses):
                    row.extend([f"{val_losses[i]:.6f}", f"{val_accs[i]:.4f}"])
                
                writer.writerow(row)
        
        print(f"Training metrics saved to CSV: {csv_path}")
        return csv_path
        
    except Exception as e:
        print(f"Error saving training metrics to CSV: {e}")
        return None

def create_confusion_matrix(model, test_loader, class_names, output_dir, device):
    """Generate detailed confusion matrix analysis"""
    try:
        model.eval()
        all_preds = []
        all_labels = []
        
        print("Computing predictions for confusion matrix...")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast() if MIXED_PRECISION and torch.cuda.is_available() else torch.no_grad():
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix heatmap
        plt.subplot(2, 1, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Image Classification', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Calculate per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100
        
        # Plot per-class accuracy bars
        plt.subplot(2, 1, 2)
        bars = plt.bar(class_names, class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Per-Class Accuracy (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        
        # Add percentage labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save confusion matrix plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_plot_path = os.path.join(output_dir, f"image_confusion_matrix_{timestamp}.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed classification report
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        report_path = os.path.join(output_dir, f"image_classification_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("SASL Image Classification - Detailed Performance Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        # Save accuracy data as JSON
        accuracy_data = {
            'overall_accuracy': np.mean(class_accuracies),
            'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(class_accuracies)},
            'confusion_matrix': cm.tolist(),
            'timestamp': timestamp
        }
        
        json_path = os.path.join(output_dir, f"image_confusion_matrix_data_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(accuracy_data, f, indent=2)
        
        print(f"Confusion matrix saved: {cm_plot_path}")
        print(f"Classification report saved: {report_path}")
        print(f"Accuracy data saved: {json_path}")
        
        return cm, class_accuracies
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        return None, None

def train_image_model(custom_batch_size=None, custom_epochs=None):
    """
    Main training function for image-based SASL classification
    Args:
        custom_batch_size (int): Override default batch size
        custom_epochs (int): Override default number of epochs
    """
    print("Starting Image-Based SASL Training")
    print("=" * 50)
    
    # Initialize GPU device and mixed precision (print once only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION and torch.cuda.is_available() else None
    
    # Use custom parameters if provided
    batch_size = custom_batch_size if custom_batch_size is not None else BATCH_SIZE
    num_epochs = custom_epochs if custom_epochs is not None else NUM_EPOCHS
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Mixed Precision: {MIXED_PRECISION and torch.cuda.is_available()}")
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(workspace_dir, "dataset_images")
    config_dir = os.path.join(workspace_dir, "03_DATA_CONFIG")
    output_dir = os.path.join(workspace_dir, "05_OUTPUT_GENERATED")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load class names - auto-detect from dataset_images directory
    class_names_path = os.path.join(config_dir, "class_names.json")
    
    # First, detect actual classes from dataset_images directory
    detected_classes = []
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if this directory has images
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        image_files.extend([f for f in os.listdir(item_path) if f.lower().endswith(ext)])
                    except:
                        continue
                if len(image_files) > 0:
                    detected_classes.append(item)
        
        detected_classes.sort()  # Sort alphabetically for consistency
    
    if detected_classes:
        print(f"Auto-detected classes from dataset_images: {detected_classes}")
        class_names = detected_classes
        
        # Update class_names.json with detected classes
        try:
            with open(class_names_path, 'w') as f:
                json.dump(class_names, f, indent=2)
            print(f"Updated class_names.json with detected classes")
        except Exception as e:
            print(f"Could not update class_names.json: {e}")
    else:
        # Fallback to existing class_names.json or default
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
            print(f"Using existing class_names.json: {class_names}")
        else:
            print("Using default focused classes...")
            class_names = ["cousin", "before", "cool", "thin", "drink", "go"]
    
    print(f"Training classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Check if dataset_images has data
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        print("Please add images to dataset_images/ organized by class folders")
        print("Example structure:")
        print("   dataset_images/")
        print("   ├── cousin/")
        print("   ├── before/")
        print("   ├── cool/")
        print("   └── ...")
        return
    
    # Create datasets with transforms
    train_transform = get_data_transforms(is_training=True)
    val_transform = get_data_transforms(is_training=False)
    
    # Create full dataset first
    full_dataset = SASLImageDataset(data_dir, class_names, train_transform)
    
    if len(full_dataset) == 0:
        print("No images found in dataset_images directory!")
        print("Please add images organized by class folders")
        return
    
    # Split dataset
    if ENABLE_VALIDATION:
        dataset_size = len(full_dataset)
        val_size = int(VALIDATION_SPLIT * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Apply validation transforms to validation set
        val_dataset.dataset.transform = val_transform
        
        print(f"Dataset split: {train_size} training, {val_size} validation")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"Training on full dataset: {len(train_dataset)} images")
    
    # Create data loaders with weighted sampling
    weighted_sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=weighted_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
    
    # Initialize model
    print(f"Building SASLImageCNN model...")
    model = SASLImageCNN(num_classes=len(class_names), pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training tracking
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if MIXED_PRECISION and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (preds == labels).sum().item()
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = (correct_preds / total_preds) * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        val_loss = 0.0
        val_acc = 0.0
        
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    if MIXED_PRECISION and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
            
            val_loss = val_running_loss / len(val_loader)
            val_acc = (val_correct / val_total) * 100
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(output_dir, "best_image_sasl_model.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'class_names': class_names
                }, best_model_path)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} Complete ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loader:
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_image_sasl_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': NUM_EPOCHS,
        'class_names': class_names,
        'train_acc': train_accs[-1] if train_accs else 0,
        'val_acc': val_accs[-1] if val_accs else 0
    }, final_model_path)
    
    print(f"\nFinal model saved: {final_model_path}")
    if val_loader:
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best model saved: {best_model_path}")
    
    # Generate visualizations
    print(f"\nGenerating training visualizations...")
    
    # Training history plot (combined loss and accuracy)
    if train_losses and train_accs:
        plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir)
    
    # Dedicated accuracy progress chart
    if train_accs:
        print("Creating accuracy progress chart...")
        plot_accuracy_chart(train_accs, val_accs, output_dir)
    
    # Save training metrics to CSV
    if train_losses and train_accs:
        print("Saving training metrics to CSV...")
        save_training_metrics(train_losses, train_accs, val_losses, val_accs, output_dir)
    
    # Confusion matrix (only with validation data)
    if val_loader:
        print("Creating confusion matrix...")
        # Load best model for confusion matrix
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        cm, class_accuracies = create_confusion_matrix(model, val_loader, class_names, output_dir, device)
        if cm is not None:
            print("Confusion matrix analysis completed!")
    else:
        print("Skipping confusion matrix - no validation data available")
    
    print(f"\nImage-Based Training Completed Successfully!")
    print(f"Final training accuracy: {train_accs[-1]:.2f}%" if train_accs else "N/A")
    if val_accs:
        print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    try:
        train_image_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()