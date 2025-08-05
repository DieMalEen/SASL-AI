# GPU-Optimized SASL Training Script
# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress MediaPipe specific warnings
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
import json
import time
from hand_detection import HandDetector, extract_hand_focused_frames

# GPU Configuration and Optimization
def setup_gpu_optimization():
    """Configure optimal GPU settings for training"""
    # Enable cuDNN optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set memory allocation strategy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory optimization
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🚀 GPU Acceleration Enabled!")
        print(f"   GPU: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        
        return device
    else:
        print("⚠️  CUDA not available, falling back to CPU")
        return torch.device("cpu")

# Optimized batch sizes for your GTX 1650 (4GB VRAM)
def get_optimal_batch_size(device):
    """Get optimal batch size based on available GPU memory"""
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory_gb >= 8:
            return 8  # High-end GPU
        elif gpu_memory_gb >= 6:
            return 6  # Mid-range GPU
        elif gpu_memory_gb >= 4:
            return 4  # Your GTX 1650
        else:
            return 2  # Low VRAM
    else:
        return 2  # CPU fallback

# Enhanced transforms optimized for GPU processing
gpu_optimized_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# GPU-optimized augmentation strategies
gpu_augmentation_strategies = [
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ColorJitter(brightness=0.1, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.4, hue=0.1),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomEqualize(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

def extract_frames_with_hands_gpu_optimized(video_path, num_frames=16):
    """GPU-optimized frame extraction with hand detection"""
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
        
        # Convert to PIL Images efficiently
        pil_frames = []
        for frame in hand_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))
        
        hand_detector.close()
        return pil_frames
        
    except Exception as e:
        print(f"Hand detection failed for {video_path}: {e}")
        hand_detector.close()
        return extract_frames_original_gpu(video_path, num_frames)

def extract_frames_original_gpu(video_path, num_frames=16):
    """GPU-optimized original frame extraction method"""
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

def preprocess_video_with_hands_gpu(video_path, transform, num_frames=16):
    """GPU-optimized video preprocessing with hand detection"""
    frames = extract_frames_with_hands_gpu_optimized(video_path, num_frames)
    return torch.stack([transform(frame) for frame in frames])

def preprocess_video_with_hand_strategy_gpu(video_path, strategy_idx, num_frames=16):
    """GPU-optimized video preprocessing with specific augmentation strategy"""
    frames = extract_frames_with_hands_gpu_optimized(video_path, num_frames)
    transform = gpu_augmentation_strategies[strategy_idx]
    return torch.stack([transform(frame) for frame in frames])

class GPUOptimizedSASLDataset(Dataset):
    """GPU-optimized dataset class for faster training"""
    def __init__(self, videos, labels, base_transform, augment=False, augment_factor=8):
        self.videos = videos
        self.labels = labels
        self.base_transform = base_transform
        self.augment = augment
        self.augment_factor = augment_factor
        
        if self.augment:
            self.expanded_videos = []
            self.expanded_labels = []
            self.augment_strategies = []
            
            for video, label in zip(videos, labels):
                # Add original with hand detection
                self.expanded_videos.append(video)
                self.expanded_labels.append(label)
                self.augment_strategies.append(-1)  # -1 means base transform
                
                # Add augmented versions
                for i in range(augment_factor - 1):
                    self.expanded_videos.append(video)
                    self.expanded_labels.append(label)
                    self.augment_strategies.append(i % len(gpu_augmentation_strategies))
        else:
            self.expanded_videos = videos
            self.expanded_labels = labels
            self.augment_strategies = [-1] * len(videos)

    def __len__(self):
        return len(self.expanded_videos)

    def __getitem__(self, idx):
        video_path = self.expanded_videos[idx]
        label = self.expanded_labels[idx]
        strategy_idx = self.augment_strategies[idx]
        
        if strategy_idx == -1:
            # Use base transform with hand detection
            frames = preprocess_video_with_hands_gpu(video_path, self.base_transform)
        else:
            # Use specific GPU-optimized augmentation strategy
            frames = preprocess_video_with_hand_strategy_gpu(video_path, strategy_idx)
            
        return frames, label

# GPU-Optimized CNN-LSTM model
class GPUOptimizedHandFocusedCNN_LSTM(nn.Module):
    """GPU-optimized CNN-LSTM model with mixed precision support"""
    def __init__(self, cnn, hidden_size=256, num_classes=20, num_layers=2, dropout=0.3):
        super(GPUOptimizedHandFocusedCNN_LSTM, self).__init__()
        self.cnn = cnn
        
        # Enhanced LSTM optimized for GPU
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Optimized attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced classifier with GPU-optimized layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),  # inplace=True for memory efficiency
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Extract CNN features with memory optimization
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.cuda.amp.autocast():  # Mixed precision
            features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention mechanism
        attended_features, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Use global average pooling over sequence dimension
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Final classification
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

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def main():
    """GPU-optimized main training function"""
    # Setup GPU optimization
    device = setup_gpu_optimization()
    
    # Configuration
    ENABLE_VALIDATION = True
    USE_HAND_DETECTION = True
    USE_MIXED_PRECISION = True  # For faster training on modern GPUs
    
    # Get optimal batch size for your GPU
    batch_size = get_optimal_batch_size(device)
    print(f"🎯 Optimized batch size for your GPU: {batch_size}")
    
    print("\n📊 Loading dataset with GPU-optimized hand detection...")
    # Try different dataset paths
    dataset_paths = ["../dataset", "dataset", "./dataset"]
    root_dir = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            root_dir = path
            break
    
    if root_dir is None:
        raise FileNotFoundError("Dataset folder not found. Please ensure 'dataset' folder exists.")
    
    print(f"Using dataset path: {root_dir}")
    video_paths, labels, class_names = load_dataset_from_folder(root_dir)

    print(f"Classes: {class_names}")
    print(f"Total videos: {len(video_paths)}")

    # Save class names to proper config location using absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dir = os.path.join(project_root, "03_DATA_CONFIG")
    os.makedirs(config_dir, exist_ok=True)
    class_names_path = os.path.join(config_dir, "class_names.json")
    
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")

    # Smart data splitting for single-video classes
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
    
    # Create GPU-optimized datasets
    print("Creating GPU-optimized datasets...")
    train_dataset = GPUOptimizedSASLDataset(
        train_paths, train_labels, gpu_optimized_transform, 
        augment=True, augment_factor=8  # Reduced for GPU memory efficiency
    )
    test_dataset = GPUOptimizedSASLDataset(
        test_paths, test_labels, gpu_optimized_transform, 
        augment=False
    ) if test_paths else None
    
    # Create data loaders with GPU optimization
    # Use more workers for GPU training (but 0 for Windows compatibility)
    num_workers = 0 if os.name == 'nt' else 4
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=False  # For Windows compatibility
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=False
    ) if test_dataset else None
    
    print(f"Training videos: {len(train_paths)}")
    print(f"Training dataset size with augmentation: {len(train_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    else:
        print("!!!  No test data available - training without validation")

    # Initialize GPU-optimized model
    cnn_base = models.resnet18(pretrained=True)
    cnn_base = nn.Sequential(*list(cnn_base.children())[:-1])
    
    # Fine-tune some CNN layers for better performance
    for param in cnn_base.parameters():
        param.requires_grad = False
    
    # Unfreeze last few layers for fine-tuning
    for param in cnn_base[-2:].parameters():
        param.requires_grad = True

    model = GPUOptimizedHandFocusedCNN_LSTM(
        cnn=cnn_base, 
        num_classes=len(class_names),
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ).to(device)

    print(f"🧠 Using GPU-Optimized Hand-Focused CNN-LSTM")
    print_gpu_memory_usage()

    # Setup output directory using absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = "gpu_optimized_hand_focused_sasl_model.pth"
    model_path = os.path.join(output_dir, model_name)
    best_model_path = os.path.join(output_dir, f"best_{model_name}")

    # GPU-optimized training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)  # Slightly higher LR for GPU
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and device.type == 'cuda' else None
    
    print(f"🚀 Starting GPU-accelerated training with mixed precision: {USE_MIXED_PRECISION and device.type == 'cuda'}")
    
    num_epochs = 15
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase with GPU optimization
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()
        
        for batch_idx, (frames, labels_batch) in enumerate(train_loader):
            frames, labels_batch = frames.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
            
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
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {batch_time:.1f}s")
                if batch_idx % 50 == 0:  # Print GPU usage every 50 batches
                    print_gpu_memory_usage()

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
                    frames, labels_batch = frames.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
                    
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
            
            print(f"\n⚡ GPU Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.1f}s")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print_gpu_memory_usage()
            print("-" * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"🏆 New best GPU model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            # Training without validation
            scheduler.step(avg_train_loss)
            
            print(f"\n⚡ GPU Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.1f}s")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print("(No validation - single video per class)")
            print_gpu_memory_usage()
            print("-" * 60)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(output_dir, f"gpu_epoch_{epoch+1}_{model_name}")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"🔄 GPU model checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), model_path)
    total_time = time.time() - start_time
    
    print(f"\n🎉 GPU Training completed successfully in {total_time/60:.1f} minutes!")
    print(f"Final model saved to {model_path}")
    
    if ENABLE_VALIDATION and test_loader is not None:
        print(f"🏆 Best validation accuracy achieved: {best_val_acc:.4f}")
    else:
        print("Training completed without validation (single video per class)")
        print("Consider adding more videos per class for better model evaluation")
    
    print(f"GPU training used {len(gpu_augmentation_strategies)} different augmentation strategies")
    print(f"Final training dataset size: {len(train_dataset)} samples")
    print_gpu_memory_usage()

if __name__ == '__main__':
    main()
