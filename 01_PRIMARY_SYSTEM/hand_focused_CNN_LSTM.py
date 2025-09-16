# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress MediaPipe specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import numpy as np
import json
import argparse
from hand_detection import HandDetector, extract_hand_focused_frames
import time
from torch.cuda.amp import autocast, GradScaler

# Check CUDA availability and set up mixed precision accordingly
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from torch.cuda.amp import autocast, GradScaler
    print("✓ CUDA available - Mixed precision training enabled")
else:
    # Fallback autocast for CPU
    from torch.amp import autocast
    print("⚠ CUDA not available - Using CPU training")

# Force lightweight model for system stability
try:
    import efficientnet_pytorch
    EFFICIENTNET_AVAILABLE = False  # Force disable to prevent resource issues
    print("⚠ EfficientNet disabled for system stability - using lightweight ResNet18")
    print("  This reduces memory usage and prevents system crashes")
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("⚠ EfficientNet not available - using lightweight ResNet18")
    print("  Install with: pip install efficientnet_pytorch (for optimal performance)")

# GPU-Optimized training parameters for A10-12Q (REDUCED for stability)
batch_size = 8  # REDUCED from 24 to prevent system crashes
num_epochs = 10  # REDUCED from 25 to finish faster
best_val_acc = 0.0

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# Legacy dataset classes (for fallback compatibility)
class AdvancedSASLDataset(Dataset):
    def __init__(self, videos, labels, base_transform, augment=True, augment_factor=10):
        self.videos = videos
        self.labels = labels
        self.base_transform = base_transform
        self.augment = augment
        self.augment_factor = augment_factor
        
        # Standard augmentation strategies (for fallback)
        self.augmentation_strategies = [
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
        
        if self.augment:
            self.expanded_videos = []
            self.expanded_labels = []
            self.augment_strategies = []
            
            for video, label in zip(videos, labels):
                # Add original
                self.expanded_videos.append(video)
                self.expanded_labels.append(label)
                self.augment_strategies.append(-1)  # -1 means no augmentation
                
                # Add augmented versions
                for i in range(augment_factor - 1):
                    self.expanded_videos.append(video)
                    self.expanded_labels.append(label)
                    self.augment_strategies.append(i % len(self.augmentation_strategies))
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
            # Use base transform (no augmentation)
            frames = self._preprocess_video_basic(video_path, self.base_transform)
        else:
            # Use specific augmentation strategy
            frames = self._preprocess_video_with_strategy(video_path, strategy_idx)
            
        return frames, label
    
    def _preprocess_video_basic(self, video_path, transform, num_frames=16):
        """Basic video preprocessing without hand detection"""
        frames = extract_frames_original(video_path, num_frames)
        return torch.stack([transform(frame) for frame in frames])
    
    def _preprocess_video_with_strategy(self, video_path, strategy_idx, num_frames=16):
        """Preprocess video with specific augmentation strategy"""
        frames = extract_frames_original(video_path, num_frames)
        transform = self.augmentation_strategies[strategy_idx]
        return torch.stack([transform(frame) for frame in frames])

# Standard CNN-LSTM model (for fallback)
class CNN_LSTM(nn.Module):
    def __init__(self, cnn, hidden_size=256, num_classes=20, num_layers=1):
        super(CNN_LSTM, self).__init__()
        self.cnn = cnn
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)  # Flatten batch and sequence for CNN
        features = self.cnn(x)  # Extract features with CNN
        features = features.view(batch_size, seq_len, -1)  # Reshape for LSTM
        lstm_out, _ = self.lstm(features)  # Pass through LSTM
        out = self.fc(lstm_out[:, -1, :])  # Use output from last time step
        return out

def extract_frames_with_hands(video_path, num_frames=16, show_progress=False):
    """Extract frames from video with hand detection focus"""
    if show_progress:
        print(f"   Processing: {os.path.basename(video_path)}")
    
    hand_detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )
    
    try:
        import threading
        import time as time_module
        
        # Use threading for timeout on Windows
        result_container = {'result': None, 'error': None}
        
        def run_hand_detection():
            try:
                result_container['result'] = extract_hand_focused_frames(
                    video_path, hand_detector, num_frames
                )
            except Exception as e:
                result_container['error'] = e
        
        # Start hand detection in thread with timeout
        thread = threading.Thread(target=run_hand_detection)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30.0)  # 30 second timeout
        
        if thread.is_alive():
            # Timeout occurred
            if show_progress:
                print(f"   ⚠ Hand detection timeout for {os.path.basename(video_path)}, using fallback")
            hand_detector.close()
            return extract_frames_original(video_path, num_frames)
        
        if result_container['error']:
            raise result_container['error']
        
        if result_container['result'] is None:
            raise ValueError("No result from hand detection")
        
        hand_frames, full_frames = result_container['result']
        
        # Convert to PIL Images
        pil_frames = []
        for frame in hand_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))
        
        hand_detector.close()
        return pil_frames
        
    except Exception as e:
        if show_progress:
            print(f"   ⚠ Hand detection error for {os.path.basename(video_path)}: {str(e)[:50]}...")
        hand_detector.close()
        return extract_frames_original(video_path, num_frames)

def extract_frames_original(video_path, num_frames=16):
    """Original frame extraction method as fallback"""
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

def preprocess_video_with_hands(video_path, transform, num_frames=16, show_progress=False):
    """Enhanced video preprocessing with hand detection"""
    frames = extract_frames_with_hands(video_path, num_frames, show_progress=show_progress)
    return torch.stack([transform(frame) for frame in frames])

# Enhanced transforms for hand-focused training
hand_focused_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Enhanced augmentation strategies for hand-focused data with much more diversity
hand_augmentation_strategies = [
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

def preprocess_video_with_hand_strategy(video_path, strategy_idx, num_frames=16, show_progress=False):
    """Preprocess video with specific hand-focused augmentation strategy"""
    frames = extract_frames_with_hands(video_path, num_frames, show_progress=show_progress)
    transform = hand_augmentation_strategies[strategy_idx]
    return torch.stack([transform(frame) for frame in frames])

class SimpleSASLDataset(Dataset):
    """Simple dataset class for SASL without hand detection (faster, more stable)"""
    def __init__(self, videos, labels, transform, augment=False, augment_factor=6):
        self.videos = videos
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.augment_factor = augment_factor
        
        print(f"📋 Simple dataset: {len(videos)} videos")
        
        if self.augment:
            self.expanded_videos = []
            self.expanded_labels = []
            
            for video, label in zip(videos, labels):
                # Add original
                self.expanded_videos.append(video)
                self.expanded_labels.append(label)
                
                # Add augmented versions
                for i in range(augment_factor - 1):
                    self.expanded_videos.append(video)
                    self.expanded_labels.append(label)
        else:
            self.expanded_videos = videos
            self.expanded_labels = labels
        
        print(f"✓ Dataset created: {len(self.expanded_videos)} samples")

    def __len__(self):
        return len(self.expanded_videos)

    def __getitem__(self, idx):
        video_path = self.expanded_videos[idx]
        label = self.expanded_labels[idx]
        
        # Use simple frame extraction (no hand detection)
        frames = extract_frames_original(video_path, num_frames=16)
        frames_tensor = torch.stack([self.transform(frame) for frame in frames])
            
        return frames_tensor, label

class HandFocusedSASLDataset(Dataset):
    """Enhanced dataset class that uses hand detection for better sign language recognition"""
    def __init__(self, videos, labels, base_transform, augment=False, augment_factor=6, show_progress=False):
        self.videos = videos
        self.labels = labels
        self.base_transform = base_transform
        self.augment = augment
        self.augment_factor = augment_factor
        self.show_progress = show_progress
        
        if show_progress:
            print(f"📋 Initializing dataset with {len(videos)} videos...")
        
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
                    self.augment_strategies.append(i % len(hand_augmentation_strategies))
        else:
            self.expanded_videos = videos
            self.expanded_labels = labels
            self.augment_strategies = [-1] * len(videos)
        
        if show_progress:
            total_samples = len(self.expanded_videos)
            print(f"✓ Dataset initialized: {total_samples} samples ({len(videos)} videos × {augment_factor if augment else 1})")

    def __len__(self):
        return len(self.expanded_videos)

    def __getitem__(self, idx):
        video_path = self.expanded_videos[idx]
        label = self.expanded_labels[idx]
        strategy_idx = self.augment_strategies[idx]
        
        if strategy_idx == -1:
            # Use base transform with hand detection
            frames = preprocess_video_with_hands(video_path, self.base_transform, show_progress=self.show_progress)
        else:
            # Use specific hand-focused augmentation strategy
            frames = preprocess_video_with_hand_strategy(video_path, strategy_idx, show_progress=self.show_progress)
            
        return frames, label

# GPU-Optimized CNN-LSTM model for A10-12Q
class GPUOptimizedCNN_LSTM(nn.Module):
    """GPU-optimized CNN-LSTM model with EfficientNet/ResNet backbone for A10-12Q"""
    def __init__(self, num_classes=20, hidden_size=512, num_layers=2, dropout=0.2):
        super(GPUOptimizedCNN_LSTM, self).__init__()
        
        # Use EfficientNet-B2 if available, otherwise ResNet50
        if EFFICIENTNET_AVAILABLE:
            try:
                from efficientnet_pytorch import EfficientNet
                self.backbone = EfficientNet.from_pretrained('efficientnet-b2', num_classes=hidden_size)
                backbone_features = hidden_size
                self.use_projection = False
                print("✓ Using EfficientNet-B2 backbone")
            except Exception as e:
                print(f"⚠ EfficientNet failed to load: {e}")
                print("  Falling back to ResNet50...")
                self._setup_resnet_backbone(hidden_size)
        else:
            self._setup_resnet_backbone(hidden_size)
        
        # Optimized LSTM with fewer parameters but better performance
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Efficient attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Streamlined classifier with batch norm for faster convergence
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights for faster convergence
        self._initialize_weights()

    def _setup_resnet_backbone(self, hidden_size):
        """Setup lightweight ResNet18 backbone for system stability"""
        # Use ResNet18 instead of ResNet50 to reduce memory usage
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        backbone_features = 512  # ResNet18 has 512 features vs 2048 for ResNet50
        # Add projection layer to match hidden_size
        self.projection = nn.Linear(backbone_features, hidden_size)
        self.use_projection = True
        print("✓ Using lightweight ResNet18 backbone with projection layer")

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Extract features efficiently
        x = x.view(batch_size * seq_len, C, H, W)
        
        # Use autocast appropriately based on device
        if CUDA_AVAILABLE:
            with autocast():
                if self.use_projection:
                    # ResNet backbone
                    features = self.backbone(x)
                    features = features.view(features.size(0), -1)
                    features = self.projection(features)
                else:
                    # EfficientNet backbone
                    features = self.backbone(x)
        else:
            # CPU training - no mixed precision
            if self.use_projection:
                # ResNet backbone
                features = self.backbone(x)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
            else:
                # EfficientNet backbone
                features = self.backbone(x)
        
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention mechanism
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension for stability
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

# Enhanced CNN-LSTM model with attention mechanism for hand features
class HandFocusedCNN_LSTM(nn.Module):
    """Enhanced CNN-LSTM model with attention mechanism for better hand feature extraction"""
    def __init__(self, cnn, hidden_size=256, num_classes=20, num_layers=2, dropout=0.3):
        super(HandFocusedCNN_LSTM, self).__init__()
        self.cnn = cnn
        
        # Enhanced LSTM with more layers and dropout
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for focusing on important temporal features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Extract CNN features
        x = x.view(batch_size * seq_len, C, H, W)
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

# Load and prepare CNN backbone
cnn_base = models.resnet18(pretrained=True)
cnn_base = nn.Sequential(*list(cnn_base.children())[:-1])
for param in cnn_base.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def plot_training_history(train_losses, train_accs, val_losses=None, val_accs=None, output_dir="./"):
    """Plot training and validation loss and accuracy curves"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot loss
        ax1.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=6)
        if val_losses:
            ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2, markersize=6)
        ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Plot accuracy
        ax2.plot(epochs, [acc * 100 for acc in train_accs], 'bo-', label='Training Accuracy', linewidth=2, markersize=6)
        if val_accs:
            ax2.plot(epochs, [acc * 100 for acc in val_accs], 'ro-', label='Validation Accuracy', linewidth=2, markersize=6)
        ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history plot saved to: {plot_path}")
        
    except ImportError:
        print("⚠ Matplotlib not available - skipping training history plot")
    except Exception as e:
        print(f"⚠ Error creating training plot: {e}")

def create_confusion_matrix(model, test_loader, class_names, output_dir, device):
    """Create and save confusion matrix"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Try to import seaborn, fallback if not available
        try:
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            use_seaborn = False
            print("⚠ Seaborn not available - using matplotlib for confusion matrix")
        
        model.eval()
        all_preds = []
        all_labels = []
        
        print("🔍 Generating predictions for confusion matrix...")
        
        with torch.no_grad():
            for batch_idx, (frames, labels_batch) in enumerate(test_loader):
                frames, labels_batch = frames.to(device), labels_batch.to(device)
                
                # Use mixed precision if available
                if CUDA_AVAILABLE:
                    with autocast():
                        outputs = model(frames)
                else:
                    outputs = model(frames)
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"   Processing batch {batch_idx + 1}/{len(test_loader)}")
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate accuracy per class
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot confusion matrix
        if use_seaborn:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax1, cbar_kws={'label': 'Number of Samples'})
        else:
            im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
            ax1.figure.colorbar(im, ax=ax1, label='Number of Samples')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax1.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            ax1.set_xticks(range(len(class_names)))
            ax1.set_yticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right')
            ax1.set_yticklabels(class_names)
        
        ax1.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # Plot per-class accuracy
        bars = ax2.bar(range(len(class_names)), class_accuracies * 100, 
                      color='skyblue', edgecolor='navy', linewidth=1.5)
        ax2.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add accuracy labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {cm_path}")
        
        # Save classification report
        report = classification_report(all_labels, all_preds, target_names=class_names)
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write(f"\n\nPer-Class Accuracies:\n")
            f.write("-" * 30 + "\n")
            for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
                f.write(f"{class_name}: {acc * 100:.2f}%\n")
        
        print(f"📄 Classification report saved to: {report_path}")
        
        # Create detailed accuracy summary
        accuracy_summary = {
            'overall_accuracy': np.mean(class_accuracies) * 100,
            'per_class_accuracy': {class_name: float(acc * 100) for class_name, acc in zip(class_names, class_accuracies)},
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        summary_path = os.path.join(output_dir, 'accuracy_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(accuracy_summary, f, indent=2)
        
        print(f"📋 Accuracy summary saved to: {summary_path}")
        
        return cm, class_accuracies
        
    except ImportError as e:
        print(f"⚠ Missing dependencies for confusion matrix: {e}")
        return None, None
    except Exception as e:
        print(f"⚠ Error creating confusion matrix: {e}")
        return None, None

def main(batch_size_arg=None, epochs_arg=None):
    """Main training function - FIXED for Windows multiprocessing"""
    global batch_size, num_epochs, best_val_acc
    
    # Override defaults if arguments provided
    if batch_size_arg is not None:
        batch_size = batch_size_arg
    if epochs_arg is not None:
        num_epochs = epochs_arg
    
    print(f"> Using batch size: {batch_size}")
    print(f"> Training for {num_epochs} epochs")
    
    # Configuration
    ENABLE_VALIDATION = True
    USE_HAND_DETECTION = False  # Temporarily disabled to avoid preprocessing hang
    
    if USE_HAND_DETECTION:
        print("Loading dataset with hand detection support...")
    else:
        print("⚠ Hand detection DISABLED - Using standard preprocessing for faster startup")
        print("  (Hand detection can be re-enabled later once preprocessing is optimized)")
    
    # Try different dataset paths depending on where script is run from
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
    
    # Create datasets - Using simple dataset with MINIMAL augmentation for system stability
    print("Creating lightweight datasets for system stability...")
    train_dataset = SimpleSASLDataset(
        train_paths, train_labels, hand_focused_transform, 
        augment=True, augment_factor=2  # REDUCED from 10 to 2
    )
    test_dataset = SimpleSASLDataset(
        test_paths, test_labels, hand_focused_transform, 
        augment=False
    ) if test_paths else None
    
    # Alternative: Traditional dataset for comparison
    # train_dataset = AdvancedSASLDataset(
    #     train_paths, train_labels, hand_focused_transform, 
    #     augment=True, augment_factor=8
    # )
    # test_dataset = AdvancedSASLDataset(
    #     test_paths, test_labels, hand_focused_transform, 
    #     augment=False
    # ) if test_paths else None
    
    # Create data loaders - Simplified for stability (avoiding multiprocessing issues)
    num_workers = 0  # Disable multiprocessing to avoid hang issues
    pin_memory = False  # Disabled to prevent system crashes
    
    print("📦 Creating data loaders (single-threaded for stability)...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    ) if test_dataset else None
    
    print(f"Training videos: {len(train_paths)}")
    print(f"Training dataset size with augmentation: {len(train_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    else:
        print("!!!  No test data available - training without validation")

    # Initialize GPU-optimized model for A10-12Q
    print("Initializing optimized model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print("✓ Mixed precision training enabled")
    else:
        print("⚠ Running on CPU - Mixed precision disabled")
        # Adjust batch size for CPU
        if batch_size > 8:
            print(f"⚠ Reducing batch size from {batch_size} to 4 for CPU training")
            batch_size = 4
    
    # Use the optimized model
    model = GPUOptimizedCNN_LSTM(
        num_classes=len(class_names),
        hidden_size=512,  # Increased for better capacity
        num_layers=2,
        dropout=0.2  # Reduced dropout for better learning
    ).to(device)
    
    model_name = "gpu_optimized_sasl_model.pth"
    print(f"Using Optimized CNN-LSTM model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup output directory for models using absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
    os.makedirs(output_dir, exist_ok=True)
    
    # Update model paths to save in output directory
    model_path = os.path.join(output_dir, model_name)

    # Advanced training setup for A10-12Q
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,  # Higher initial learning rate
        weight_decay=1e-4,  # Increased weight decay
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training (only if CUDA available)
    if CUDA_AVAILABLE:
        scaler = GradScaler()
        print("✓ Mixed precision scaler initialized")
    else:
        scaler = None
        print("⚠ Mixed precision disabled (CPU training)")
    
    print("Starting training with optimizations...")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training metrics
    training_start_time = time.time()
    best_train_acc = 0.0
    
    # Initialize tracking for plots
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase with mixed precision
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start_time = time.time()
        
        for batch_idx, (frames, labels_batch) in enumerate(train_loader):
            frames, labels_batch = frames.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use mixed precision only if CUDA is available
            if CUDA_AVAILABLE and scaler is not None:
                # Mixed precision forward pass
                with autocast():
                    outputs = model(frames)
                    loss = criterion(outputs, labels_batch)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU
                outputs = model(frames)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels_batch).sum().item()
            train_total += labels_batch.size(0)
            
            # Progress reporting
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Validation phase
        if ENABLE_VALIDATION and test_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for frames, labels_batch in test_loader:
                    frames, labels_batch = frames.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
                    
                    # Use mixed precision for validation if available
                    if CUDA_AVAILABLE:
                        with autocast():
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
            
            # Track metrics for plotting
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(output_dir, f"best_{model_name}")
                torch.save(model.state_dict(), best_model_path)
                print(f"✓ New best validation accuracy: {best_val_acc:.4f}")
        else:
            # Training without validation
            # Track metrics for plotting (training only)
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            print("(No validation - single video per class)")
            print("-" * 60)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}_{model_name}")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at epoch {epoch+1}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model with training statistics
    total_training_time = time.time() - training_start_time
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': num_epochs,
        'best_val_acc': best_val_acc if ENABLE_VALIDATION else None,
        'training_time': total_training_time,
        'model_params': sum(p.numel() for p in model.parameters()),
        'class_names': class_names
    }, model_path)
    
    # Generate visualizations
    print(f"\n📊 Generating training visualizations...")
    
    # Create training history plot
    if train_losses and train_accs:
        if ENABLE_VALIDATION and val_losses and val_accs:
            plot_training_history(train_losses, train_accs, val_losses, val_accs, output_dir)
        else:
            plot_training_history(train_losses, train_accs, None, None, output_dir)
    
    # Create confusion matrix (only if validation data exists)
    if ENABLE_VALIDATION and test_loader is not None:
        print("📊 Creating confusion matrix...")
        cm, class_accuracies = create_confusion_matrix(model, test_loader, class_names, output_dir, device)
        if cm is not None:
            print("✅ Confusion matrix analysis completed!")
    else:
        print("⚠ Skipping confusion matrix - no validation data available")
    
    print(f"\n🎉 GPU-Optimized Training Completed Successfully!")
    print("=" * 60)
    print(f"⏱️  Total training time: {total_training_time/60:.1f} minutes")
    print(f"📊 Final model saved to: {model_path}")
    print(f"🏆 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📈 Training dataset size: {len(train_dataset):,} samples")
    print(f"🔄 Augmentation strategies used: {len(hand_augmentation_strategies)}")
    print(f"⚡ GPU utilization: A10-12Q optimized")
    print(f"🎯 Mixed precision training: Enabled")
    
    if ENABLE_VALIDATION and test_loader is not None:
        print(f"🎖️  Best validation accuracy: {best_val_acc:.4f}")
        best_model_path = os.path.join(output_dir, f"best_{model_name}")
        if os.path.exists(best_model_path):
            print(f"✨ Best model saved to: {best_model_path}")
    else:
        print("ℹ️  Training completed without validation (single video per class)")
        print("💡 Consider adding more videos per class for better model evaluation")
    
    print("=" * 60)
    print("🚀 Your model is ready for inference!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU-Optimized SASL Training for A10-12Q')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training (default: 24)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (default: 25)')
    
    args = parser.parse_args()
    
    print("🔥 Starting GPU-Optimized SASL Training for NVIDIA A10-12Q")
    print("=" * 60)
    
    main(batch_size_arg=args.batch_size, epochs_arg=args.epochs)
