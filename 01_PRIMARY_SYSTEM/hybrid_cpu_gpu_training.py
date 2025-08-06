# Hybrid CPU+GPU SASL Training System
# Optimizes workload distribution between CPU and GPU for maximum performance

# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import numpy as np
import argparse
import json
import time
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from hand_detection import HandDetector, extract_hand_focused_frames

class HybridCPUGPUTrainingSystem:
    """Hybrid system that optimally distributes work between CPU and GPU"""
    
    def __init__(self, enable_gpu=True, cpu_workers=None):
        """Initialize hybrid CPU+GPU system"""
        
        # Device setup
        self.gpu_available = torch.cuda.is_available() and enable_gpu
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        
        # CPU workers for data preprocessing
        if cpu_workers is None:
            self.cpu_workers = min(8, mp.cpu_count())  # Cap at 8 for Windows stability
        else:
            self.cpu_workers = cpu_workers
        
        print(f"+ Hybrid CPU+GPU Training System Initialized")
        print(f"   GPU Available: {self.gpu_available}")
        print(f"   Device: {self.device}")
        print(f"   CPU Workers: {self.cpu_workers}")
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            # GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.set_per_process_memory_fraction(0.90)  # Leave some for system
        
        # Setup transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup optimized transforms for hybrid processing"""
        
        # Base transform for consistent preprocessing
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # CPU-optimized augmentation strategies (lighter processing)
        self.cpu_augmentations = [
            transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomRotation(degrees=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
    
    def get_optimal_batch_size(self):
        """Get optimal batch size for hybrid processing"""
        if self.gpu_available:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory_gb >= 8:
                return 6  # Conservative for hybrid processing
            elif gpu_memory_gb >= 6:
                return 4
            elif gpu_memory_gb >= 4:
                return 3  # Your GTX 1650 - reduced for stability
            else:
                return 2
        else:
            return 2  # CPU only

# CPU-optimized frame extraction (runs on CPU threads)
def extract_frames_cpu_optimized(video_path, num_frames=16, use_hand_detection=True):
    """CPU-optimized frame extraction with optional hand detection"""
    
    if use_hand_detection:
        hand_detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        try:
            hand_frames, full_frames = extract_hand_focused_frames(
                video_path, hand_detector, num_frames
            )
            
            # Convert to PIL Images
            pil_frames = []
            for frame in hand_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(frame_rgb))
            
            hand_detector.close()
            return pil_frames
            
        except Exception as e:
            print(f"Hand detection failed for {video_path}: {e}")
            hand_detector.close()
            # Fallback to standard extraction
    
    # Standard frame extraction (CPU optimized)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_idxs = list(range(total_frames))
        frame_idxs.extend([total_frames - 1] * (num_frames - total_frames))
    else:
        frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frame = cv2.cvtColor(np.array(frames[-1]), cv2.COLOR_RGB2BGR)
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return frames[:num_frames]

class HybridSASLDataset(Dataset):
    """Hybrid dataset that optimizes CPU preprocessing and GPU training"""
    
    def __init__(self, videos, labels, base_transform, cpu_augmentations, 
                 augment=False, augment_factor=6, cpu_workers=4):
        self.videos = videos
        self.labels = labels
        self.base_transform = base_transform
        self.cpu_augmentations = cpu_augmentations
        self.augment = augment
        self.augment_factor = augment_factor
        self.cpu_workers = cpu_workers
        
        # Pre-compute augmentation strategies
        if self.augment:
            self.expanded_videos = []
            self.expanded_labels = []
            self.augment_strategies = []
            
            for video, label in zip(videos, labels):
                # Add original
                self.expanded_videos.append(video)
                self.expanded_labels.append(label)
                self.augment_strategies.append(-1)  # -1 means base transform
                
                # Add augmented versions
                for i in range(augment_factor - 1):
                    self.expanded_videos.append(video)
                    self.expanded_labels.append(label)
                    self.augment_strategies.append(i % len(cpu_augmentations))
        else:
            self.expanded_videos = videos
            self.expanded_labels = labels
            self.augment_strategies = [-1] * len(videos)
        
        # Setup CPU preprocessing pool
        self.preprocessing_pool = ThreadPoolExecutor(max_workers=cpu_workers)
        
    def __len__(self):
        return len(self.expanded_videos)

    def __getitem__(self, idx):
        video_path = self.expanded_videos[idx]
        label = self.expanded_labels[idx]
        strategy_idx = self.augment_strategies[idx]
        
        # CPU preprocessing: Extract frames with hand detection
        frames = extract_frames_cpu_optimized(video_path, num_frames=16, use_hand_detection=True)
        
        # Apply transforms (CPU)
        if strategy_idx == -1:
            transform = self.base_transform
        else:
            transform = self.cpu_augmentations[strategy_idx]
        
        # Transform frames on CPU
        tensor_frames = torch.stack([transform(frame) for frame in frames])
        
        return tensor_frames, label
    
    def cleanup(self):
        """Cleanup CPU resources"""
        if hasattr(self, 'preprocessing_pool'):
            self.preprocessing_pool.shutdown(wait=True)

# Hybrid-optimized CNN-LSTM model
class HybridOptimizedCNN_LSTM(nn.Module):
    """CNN-LSTM model optimized for hybrid CPU+GPU processing"""
    
    def __init__(self, cnn, hidden_size=256, num_classes=20, num_layers=2, dropout=0.3):
        super(HybridOptimizedCNN_LSTM, self).__init__()
        self.cnn = cnn
        
        # Optimized LSTM
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # CNN processing (GPU optimized)
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing (GPU)
        lstm_out, _ = self.lstm(features)
        
        # Attention (GPU)
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling and classification (GPU)
        pooled_features = torch.mean(attended_features, dim=1)
        out = self.classifier(pooled_features)
        
        return out

def load_dataset_from_folder(root_dir):
    """Load video paths and labels from dataset folders"""
    video_paths = []
    labels = []
    class_names = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(root_dir, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith(".mp4"):
                    video_paths.append(os.path.join(class_folder, filename))
                    labels.append(class_to_idx[class_name])

    return video_paths, labels, class_names

def print_system_usage(device):
    """Print current system resource usage"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    # CPU info
    cpu_percent = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 'N/A'
    print(f"   CPU Usage: {cpu_percent}")

def main(batch_size_arg=None, epochs_arg=None):
    """Main hybrid training function"""
    
    # Initialize hybrid system
    hybrid_system = HybridCPUGPUTrainingSystem(enable_gpu=True, cpu_workers=6)
    device = hybrid_system.device
    
    # Configuration
    ENABLE_VALIDATION = True
    USE_HAND_DETECTION = True
    USE_MIXED_PRECISION = hybrid_system.gpu_available
    
    # Get optimal batch size or use provided
    if batch_size_arg is not None:
        batch_size = batch_size_arg
    else:
        batch_size = hybrid_system.get_optimal_batch_size()
    
    print(f"> Using batch size: {batch_size}")
    
    print("\n| Loading dataset with hybrid CPU+GPU processing...")
    
    # Find dataset
    dataset_paths = ["../dataset", "dataset", "./dataset"]
    root_dir = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            root_dir = path
            break
    
    if root_dir is None:
        raise FileNotFoundError("Dataset folder not found.")
    
    print(f"Using dataset path: {root_dir}")
    video_paths, labels, class_names = load_dataset_from_folder(root_dir)

    print(f"Classes: {class_names}")
    print(f"Total videos: {len(video_paths)}")

    # Save class names
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dir = os.path.join(project_root, "03_DATA_CONFIG")
    os.makedirs(config_dir, exist_ok=True)
    class_names_path = os.path.join(config_dir, "class_names.json")
    
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)

    # Data splitting
    if ENABLE_VALIDATION:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        single_video_classes = unique_labels[label_counts == 1]
        
        if len(single_video_classes) > 0:
            print(f"!!! Found {len(single_video_classes)} classes with only 1 video each")
            print(f"Classes: {[class_names[i] for i in single_video_classes]}")
            print("These will be used for training only (no test split)")
            
            # Separate single-video and multi-video classes
            train_paths = []
            test_paths = []
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
                
                train_paths = train_multi + single_video_paths
                train_labels = train_multi_labels + single_video_labels
                test_paths = test_multi
                test_labels = test_multi_labels
                print("Using non-stratified split due to limited multi-video data")
            else:
                # Use all data for training
                train_paths = video_paths
                train_labels = labels
                test_paths = []
                test_labels = []
                ENABLE_VALIDATION = False
                print("Insufficient data for validation - training without validation")
        else:
            # Normal stratified split
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                video_paths, labels, test_size=0.2, random_state=42, stratify=labels
            )
    
    # Create hybrid datasets
    print("Creating hybrid CPU+GPU optimized datasets...")
    train_dataset = HybridSASLDataset(
        train_paths, train_labels, 
        hybrid_system.base_transform, hybrid_system.cpu_augmentations,
        augment=True, augment_factor=6, cpu_workers=hybrid_system.cpu_workers
    )
    
    test_dataset = HybridSASLDataset(
        test_paths, test_labels, 
        hybrid_system.base_transform, hybrid_system.cpu_augmentations,
        augment=False, cpu_workers=2
    ) if test_paths else None
    
    # Create data loaders with hybrid optimization
    # Use 0 workers for Windows compatibility, but enable pin_memory for GPU
    pin_memory = hybrid_system.gpu_available
    
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, 
        num_workers=0, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False, 
        num_workers=0, pin_memory=pin_memory
    ) if test_dataset else None
    
    print(f"Training videos: {len(train_paths)}")
    print(f"Training dataset size with augmentation: {len(train_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")

    # Initialize model
    cnn_base = models.resnet18(pretrained=True)
    cnn_base = nn.Sequential(*list(cnn_base.children())[:-1])
    
    # Fine-tune strategy for hybrid processing
    for param in cnn_base.parameters():
        param.requires_grad = False
    
    # Unfreeze last layers for better performance
    for param in cnn_base[-2:].parameters():
        param.requires_grad = True

    model = HybridOptimizedCNN_LSTM(
        cnn=cnn_base, 
        num_classes=len(class_names),
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(f"+ Using Hybrid-Optimized CNN-LSTM")
    print_system_usage(device)

    # Setup output directory
    output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = "hybrid_sasl_model.pth"
    model_path = os.path.join(output_dir, model_name)
    best_model_path = os.path.join(output_dir, f"best_{model_name}")

    # Training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision for GPU
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None
    
    print(f">> Starting Hybrid CPU+GPU Training")
    print(f"   Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"   CPU Workers: {hybrid_system.cpu_workers}")
    
    # Use provided epochs or default
    num_epochs = epochs_arg if epochs_arg is not None else 15
    print(f"> Training for {num_epochs} epochs")
    
    best_val_acc = 0.0
    start_time = time.time()

    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            epoch_start = time.time()
            
            for batch_idx, (frames, labels_batch) in enumerate(train_loader):
                # Move to GPU (data comes from CPU preprocessing)
                frames = frames.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = model(frames)
                        loss = criterion(outputs, labels_batch)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    outputs = model(frames)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels_batch).sum().item()
                train_total += labels_batch.size(0)
                
                if batch_idx % 10 == 0:
                    batch_time = time.time() - epoch_start
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_time:.1f}s")
                    
                    if batch_idx % 30 == 0:  # System usage every 30 batches
                        print_system_usage(device)

            avg_train_loss = total_loss / len(train_loader)
            train_acc = train_correct / train_total
            epoch_time = time.time() - epoch_start
            
            # Validation phase
            if ENABLE_VALIDATION and test_loader is not None:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for frames, labels_batch in test_loader:
                        frames = frames.to(device, non_blocking=True)
                        labels_batch = labels_batch.to(device, non_blocking=True)
                        
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                outputs = model(frames)
                                loss = criterion(outputs, labels_batch)
                        else:
                            outputs = model(frames)
                            loss = criterion(outputs, labels_batch)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels_batch).sum().item()
                        val_total += labels_batch.size(0)
                
                avg_val_loss = val_loss / len(test_loader)
                val_acc = val_correct / val_total
                
                scheduler.step(avg_val_loss)
                
                print(f"\n+ Hybrid Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.1f}s")
                print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print_system_usage(device)
                print("-" * 60)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"* New best hybrid model saved: {best_val_acc:.4f}")
            else:
                scheduler.step(avg_train_loss)
                print(f"\n+ Hybrid Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.1f}s")
                print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print_system_usage(device)
                print("-" * 60)
    
    finally:
        # Cleanup
        print("~ Cleaning up hybrid system...")
        train_dataset.cleanup()
        if test_dataset:
            test_dataset.cleanup()

    # Save final model
    torch.save(model.state_dict(), model_path)
    total_time = time.time() - start_time
    
    print(f"\n+ Hybrid CPU+GPU Training completed in {total_time/60:.1f} minutes!")
    print(f"Final model saved to {model_path}")
    
    if ENABLE_VALIDATION and test_loader is not None:
        print(f"* Best validation accuracy: {best_val_acc:.4f}")
    
    print(f"Hybrid training used {len(hybrid_system.cpu_augmentations)} CPU augmentation strategies")
    print(f"Final training dataset size: {len(train_dataset)} samples")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid CPU+GPU SASL Training')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    main(batch_size_arg=args.batch_size, epochs_arg=args.epochs)
