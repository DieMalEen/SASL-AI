# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings

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
from hand_detection import HandDetector, extract_hand_focused_frames

# Legacy dataset classes (for fallback compatibility)
class AdvancedSASLDataset(Dataset):
    def __init__(self, videos, labels, base_transform, augment=False, augment_factor=10):
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

def extract_frames_with_hands(video_path, num_frames=16):
    """Extract frames from video with hand detection focus"""
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

def preprocess_video_with_hands(video_path, transform, num_frames=16):
    """Enhanced video preprocessing with hand detection"""
    frames = extract_frames_with_hands(video_path, num_frames)
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

def preprocess_video_with_hand_strategy(video_path, strategy_idx, num_frames=16):
    """Preprocess video with specific hand-focused augmentation strategy"""
    frames = extract_frames_with_hands(video_path, num_frames)
    transform = hand_augmentation_strategies[strategy_idx]
    return torch.stack([transform(frame) for frame in frames])

class HandFocusedSASLDataset(Dataset):
    """Enhanced dataset class that uses hand detection for better sign language recognition"""
    def __init__(self, videos, labels, base_transform, augment=False, augment_factor=6):
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
                    self.augment_strategies.append(i % len(hand_augmentation_strategies))
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
            frames = preprocess_video_with_hands(video_path, self.base_transform)
        else:
            # Use specific hand-focused augmentation strategy
            frames = preprocess_video_with_hand_strategy(video_path, strategy_idx)
            
        return frames, label

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

def main():
    """Main training function - FIXED for Windows multiprocessing"""
    # Configuration
    ENABLE_VALIDATION = True
    USE_HAND_DETECTION = True
    
    print("Loading dataset with hand detection support...")
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

    # Save class names to proper config location
    config_dir = "../03_DATA_CONFIG"
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
            print(f"⚠️  Found {len(single_video_classes)} classes with only 1 video each")
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
                print("🔄 Using non-stratified split due to limited multi-video data")
            else:
                # Use all data for training
                train_paths = video_paths
                train_labels = labels
                test_paths = []
                test_labels = []
                ENABLE_VALIDATION = False
                print("🔄 Insufficient data for validation - training without validation")
        else:
            # Normal stratified split
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                video_paths, labels, test_size=0.2, random_state=42, stratify=labels
            )
    
    # Create datasets
    if USE_HAND_DETECTION:
        print("Creating hand-focused datasets...")
        train_dataset = HandFocusedSASLDataset(
            train_paths, train_labels, hand_focused_transform, 
            augment=True, augment_factor=10
        )
        test_dataset = HandFocusedSASLDataset(
            test_paths, test_labels, hand_focused_transform, 
            augment=False
        ) if test_paths else None
    else:
        train_dataset = AdvancedSASLDataset(
            train_paths, train_labels, hand_focused_transform, 
            augment=True, augment_factor=8
        )
        test_dataset = AdvancedSASLDataset(
            test_paths, test_labels, hand_focused_transform, 
            augment=False
        ) if test_paths else None
    
    # Create data loaders - FIXED: num_workers=0 for Windows
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0) if test_dataset else None
    
    print(f"Training videos: {len(train_paths)}")
    print(f"Training dataset size with augmentation: {len(train_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    else:
        print("⚠️  No test data available - training without validation")

    # Initialize model
    if USE_HAND_DETECTION:
        model = HandFocusedCNN_LSTM(
            cnn=cnn_base, 
            num_classes=len(class_names),
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        ).to(device)
        model_name = "hand_focused_sasl_model.pth"
    else:
        model = CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)
        model_name = "sasl_model.pth"

    print(f"Using model: {'Hand-Focused CNN-LSTM' if USE_HAND_DETECTION else 'Standard CNN-LSTM'}")

    # Setup output directory for models
    output_dir = "../05_OUTPUT_GENERATED"
    os.makedirs(output_dir, exist_ok=True)
    
    # Update model paths to save in output directory
    model_path = os.path.join(output_dir, model_name)
    best_model_path = os.path.join(output_dir, f"best_{model_name}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    num_epochs = 15
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (frames, labels_batch) in enumerate(train_loader):
            frames, labels_batch = frames.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels_batch).sum().item()
            train_total += labels_batch.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        if ENABLE_VALIDATION and test_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for frames, labels_batch in test_loader:
                    frames, labels_batch = frames.to(device), labels_batch.to(device)
                    outputs = model(frames)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels_batch).sum().item()
                    val_total += labels_batch.size(0)
            
            avg_val_loss = val_loss / len(test_loader)
            val_acc = val_correct / val_total
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            # Training without validation
            scheduler.step(avg_train_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print("(No validation - single video per class)")
            print("-" * 50)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(output_dir, f"epoch_{epoch+1}_{model_name}")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"📁 Model checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Final model saved to {model_path}")
    
    if ENABLE_VALIDATION and test_loader is not None:
        print(f"🏆 Best validation accuracy achieved: {best_val_acc:.4f}")
    else:
        print("📈 Training completed without validation (single video per class)")
        print("💡 Consider adding more videos per class for better model evaluation")
    
    print(f"🎓 Training used {len(hand_augmentation_strategies)} different augmentation strategies")
    print(f"📊 Final training dataset size: {len(train_dataset)} samples")

if __name__ == '__main__':
    main()
