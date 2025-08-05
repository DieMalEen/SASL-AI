#!/usr/bin/env python3
"""
GPU-Optimized SASL Training Script - Enhanced for Maximum GPU Utilization
Optimized specifically for GTX 1650 and similar GPUs
"""

# Suppress verbose logging
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import numpy as np
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Import hand detection
try:
    from hand_detection import HandDetector, extract_hand_focused_frames
except ImportError:
    print("! Hand detection module not found, using fallback")
    HandDetector = None

def setup_gpu_optimization():
    """Configure optimal GPU settings for maximum utilization"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f">> GPU Acceleration Enabled!")
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        
        return device
    else:
        print("! CUDA not available, falling back to CPU")
        return torch.device("cpu")

def get_optimal_batch_size(device):
    """Get aggressive batch size for maximum GPU utilization"""
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # More aggressive batch sizes for better GPU utilization
        if gpu_memory_gb >= 8:
            return 16  # High-end GPU
        elif gpu_memory_gb >= 6:
            return 12  # Mid-range GPU
        elif gpu_memory_gb >= 4:
            return 8   # GTX 1650 - increased from 4 to 8
        else:
            return 4   # Low VRAM - increased from 2 to 4
    else:
        return 2  # CPU fallback

# Optimized transforms for speed
fast_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Minimal augmentation for speed
minimal_augmentation = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

class OptimizedVideoDataset(Dataset):
    """GPU-optimized dataset with pre-processed frames"""
    
    def __init__(self, videos, labels, transform=None, sequence_length=16, enable_augmentation=True):
        self.videos = videos
        self.labels = labels
        self.transform = transform or fast_transform
        self.sequence_length = sequence_length
        self.enable_augmentation = enable_augmentation
        
        # Pre-process all videos during initialization for speed
        print("| Pre-processing videos for GPU optimization...")
        self.processed_data = []
        self._preprocess_all_videos()
        
    def _preprocess_all_videos(self):
        """Pre-process all videos to eliminate runtime bottlenecks"""
        
        for video_path, label in zip(self.videos, self.labels):
            try:
                # Extract frames without hand detection for speed
                frames = self._extract_frames_fast(video_path)
                
                if len(frames) >= self.sequence_length:
                    # Create multiple sequences from each video
                    num_sequences = max(1, len(frames) // self.sequence_length)
                    
                    for i in range(num_sequences):
                        start_idx = i * (len(frames) // num_sequences)
                        end_idx = start_idx + self.sequence_length
                        
                        if end_idx <= len(frames):
                            sequence_frames = frames[start_idx:end_idx]
                            self.processed_data.append((sequence_frames, label))
                            
                            # Add augmented version if enabled
                            if self.enable_augmentation and len(minimal_augmentation) > 1:
                                self.processed_data.append((sequence_frames, label))
                                
            except Exception as e:
                print(f"! Error processing {video_path}: {e}")
                continue
                
        print(f"+ Processed {len(self.processed_data)} sequences for training")
    
    def _extract_frames_fast(self, video_path):
        """Extract frames quickly without hand detection"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample frames evenly for speed
        frame_step = max(1, total_frames // (self.sequence_length * 2))
        
        frame_count = 0
        while cap.isOpened() and len(frames) < self.sequence_length * 3:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_step == 0:
                # Resize immediately for memory efficiency
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        frames, label = self.processed_data[idx]
        
        # Apply random augmentation
        if self.enable_augmentation and len(minimal_augmentation) > 1:
            transform = np.random.choice(minimal_augmentation)
        else:
            transform = self.transform
            
        # Convert frames to tensors quickly
        tensor_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            tensor_frame = transform(frame)
            tensor_frames.append(tensor_frame)
            
        # Stack into sequence tensor
        sequence_tensor = torch.stack(tensor_frames)
        
        return sequence_tensor, torch.tensor(label, dtype=torch.long)

class FastCNNLSTM(nn.Module):
    """Optimized CNN-LSTM for maximum GPU throughput"""
    
    def __init__(self, num_classes, hidden_size=256, num_layers=1):
        super(FastCNNLSTM, self).__init__()
        
        # Use lightweight CNN backbone
        resnet = models.resnet18(weights='DEFAULT')
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Efficient feature reduction
        self.feature_reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce spatial dimensions
            nn.Flatten(start_dim=2)
        )
        
        # Streamlined LSTM
        self.lstm = nn.LSTM(
            input_size=512 * 16,  # 512 channels * 4*4 spatial
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Process all frames in parallel
        x = x.view(batch_size * seq_len, C, H, W)
        
        # CNN feature extraction
        features = self.cnn(x)
        features = self.feature_reduction(features)
        
        # Reshape for LSTM
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Use last time step
        output = self.classifier(lstm_out[:, -1, :])
        
        return output

def create_optimized_dataloaders(dataset_path, batch_size, sequence_length=16):
    """Create optimized dataloaders for maximum GPU utilization"""
    
    print("| Creating optimized dataset for GPU training...")
    
    # Collect video data
    video_paths = []
    labels = []
    class_names = []
    
    for class_idx, class_name in enumerate(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_paths.append(os.path.join(class_path, video_file))
                    labels.append(class_idx)
    
    print(f"Found {len(video_paths)} videos across {len(class_names)} classes")
    
    # Save class names
    os.makedirs("03_DATA_CONFIG", exist_ok=True)
    with open("03_DATA_CONFIG/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    # Split data intelligently based on class distribution
    from collections import Counter
    label_counts = Counter(labels)
    
    # Check if we can use stratified split (need at least 2 samples per class)
    min_samples = min(label_counts.values())
    can_stratify = min_samples >= 2 and len(video_paths) > 10
    
    if can_stratify:
        print("| Using stratified train-test split")
        train_videos, test_videos, train_labels, test_labels = train_test_split(
            video_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
    else:
        # Handle single-video classes like hand_focused_CNN_LSTM.py
        unique_labels, label_counts_array = np.unique(labels, return_counts=True)
        single_video_classes = unique_labels[label_counts_array == 1]
        
        if len(single_video_classes) > 0:
            print(f"!!! Found {len(single_video_classes)} classes with only 1 video each")
            print(f"Classes: {[class_names[i] for i in single_video_classes]}")
            print("These will be used for training only (no test split)")
            
            # Separate single-video and multi-video classes
            train_videos = []
            test_videos = []
            train_labels = []
            test_labels = []
            
            # Collect multi-video class data
            multi_video_paths = []
            multi_video_labels = []
            for path, label in zip(video_paths, labels):
                if label not in single_video_classes:
                    multi_video_paths.append(path)
                    multi_video_labels.append(label)
            
            # Split multi-video classes if possible
            if multi_video_paths and len(multi_video_paths) >= 6:
                train_multi, test_multi, train_multi_labels, test_multi_labels = train_test_split(
                    multi_video_paths, multi_video_labels, test_size=0.25, random_state=42
                )
                
                # Add single-video classes to training
                single_video_paths = [path for path, label in zip(video_paths, labels) if label in single_video_classes]
                single_video_labels = [label for label in labels if label in single_video_classes]
                
                train_videos = train_multi + single_video_paths
                train_labels = train_multi_labels + single_video_labels
                test_videos = test_multi
                test_labels = test_multi_labels
                print("Using non-stratified split due to limited multi-video data")
            else:
                # Use all data for training
                train_videos = video_paths
                train_labels = labels
                test_videos = video_paths[:min(2, len(video_paths))]
                test_labels = labels[:min(2, len(labels))]
                print("Insufficient data for validation - using minimal test set")
        else:
            # Use random split without stratification
            train_videos, test_videos, train_labels, test_labels = train_test_split(
                video_paths, labels, test_size=0.2, random_state=42
            )
    
    print(f"Training videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    # Create optimized datasets
    train_dataset = OptimizedVideoDataset(
        train_videos, train_labels, 
        sequence_length=sequence_length,
        enable_augmentation=True
    )
    
    test_dataset = OptimizedVideoDataset(
        test_videos, test_labels,
        sequence_length=sequence_length, 
        enable_augmentation=False
    )
    
    # Create dataloaders with optimal settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, mp.cpu_count()),  # Optimize worker count
        pin_memory=True,  # Speed up GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader, len(class_names)

def train_optimized_model(batch_size=None, epochs=None):
    """Main training function optimized for maximum GPU utilization"""
    
    print(">> Starting GPU-Optimized SASL Training")
    print("=" * 60)
    
    # Setup
    device = setup_gpu_optimization()
    
    # Use provided batch_size or get optimal one
    if batch_size is None:
        batch_size = get_optimal_batch_size(device)
    print(f"> Using batch size: {batch_size}")
    
    # Set default epochs if not provided
    if epochs is None:
        epochs = 15
    print(f"> Training for {epochs} epochs")
    
    # Create datasets
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"X Dataset not found at {dataset_path}")
        return
    
    train_loader, test_loader, num_classes = create_optimized_dataloaders(
        dataset_path, batch_size
    )
    
    # Create model
    model = FastCNNLSTM(num_classes=num_classes).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003,  # Reduced max_lr to prevent NaN loss
        steps_per_epoch=len(train_loader), epochs=epochs
    )
    
    # Mixed precision for speed
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print(f"+ Model loaded on {device}")
    print(f"| Training batches: {len(train_loader)}")
    print(f">> Starting training with mixed precision: {scaler is not None}")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                # Gradient clipping to prevent NaN
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                # Gradient clipping to prevent NaN
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            running_loss += loss.item()
            
            # Check for NaN loss and handle it
            if torch.isnan(loss):
                print(f"⚠️  Warning: NaN loss detected at batch {batch_idx}")
                print("   Skipping this batch and continuing...")
                continue
            
            # Progress updates
            if batch_idx % 10 == 0:
                batch_time = time.time() - epoch_start
                gpu_memory = torch.cuda.memory_allocated() / 1024**3 if device.type == 'cuda' else 0
                
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Time: {batch_time:.1f}s, "
                      f"GPU Memory: {gpu_memory:.2f}GB")
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        
        print(f"+ Epoch {epoch+1} completed in {epoch_time:.1f}s, Avg Loss: {avg_loss:.4f}")
        
        # Checkpoint saving disabled for GPU-optimized training
        # if (epoch + 1) % 5 == 0:
        #     os.makedirs("05_OUTPUT_GENERATED", exist_ok=True)
        #     checkpoint_path = f"05_OUTPUT_GENERATED/gpu_optimized_checkpoint_epoch_{epoch+1}.pth"
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'epoch': epoch,
        #         'loss': avg_loss
        #     }, checkpoint_path)
        #     print(f">> Checkpoint saved: {checkpoint_path}")
    
    # Final save with unique GPU-optimized name
    total_time = time.time() - start_time
    model_path = "05_OUTPUT_GENERATED/gpu_sasl_model.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"\n+ Training completed!")
    print(f"* Total time: {total_time/60:.1f} minutes")
    print(f">> Model saved: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU-Optimized SASL Training')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    train_optimized_model(batch_size=args.batch_size, epochs=args.epochs)
