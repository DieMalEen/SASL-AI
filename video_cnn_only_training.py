#!/usr/bin/env python3
"""
SASL CNN-Only Video Training System
==================================

Streamlined version that trains only the CNN+LSTM model for better performance.
Based on analysis showing CNN+LSTM achieves 90-100% accuracy vs lower pose accuracy.

Key Features:
- CNN+LSTM architecture with EfficientNet backbone
- No pose dependency - more robust and faster
- Simplified training pipeline
- Better performance through focused approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import json
import pickle
import timm
import random
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import multiprocessing as mp_cpu
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Set device and configure PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Using CPU - Consider GPU for faster training")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def process_single_video(args):
    """
    Process a single video file for CNN-only training
    """
    video_path, sequence_length, input_size, cache_dir = args
    
    try:
        # Check cache first
        cache_filename = f"{Path(video_path).stem}_{sequence_length}_{input_size[0]}x{input_size[1]}_cnn.pkl"
        cache_path = cache_dir / cache_filename
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (cached_data.get('sequence_length') == sequence_length and
                        cached_data.get('input_size') == input_size):
                        return (video_path, cached_data['video_seq'], True)
            except:
                pass  # Cache corrupted, process normally
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (video_path, None, False)
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, input_size)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        cap.release()
        
        # Adjust sequence length
        if len(frames) == 0:
            return (video_path, None, False)
        
        # Pad or trim to target length
        if len(frames) > sequence_length:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, sequence_length).astype(int)
            frames = [frames[i] for i in indices]
        elif len(frames) < sequence_length:
            # Pad with last frame
            while len(frames) < sequence_length:
                frames.append(frames[-1])
        
        video_seq = np.array(frames) / 255.0
        
        # Cache the results
        try:
            cached_data = {
                'video_seq': video_seq,
                'sequence_length': sequence_length,
                'input_size': input_size
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except:
            pass  # Ignore cache save errors
        
        return (video_path, video_seq, False)
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return (video_path, None, False)

class SASLVideoDataset(Dataset):
    """PyTorch Dataset for SASL video sequences (CNN-only)"""
    
    def __init__(self, video_sequences, labels, transform=None, augment_factor=0):
        self.video_sequences = video_sequences
        self.labels = labels
        self.transform = transform
        self.augment_factor = augment_factor
        
        # Apply augmentation if requested
        if augment_factor > 0:
            self._create_augmented_data()
    
    def _create_augmented_data(self):
        """Create augmented versions of the dataset"""
        print(f"Creating augmented data ({self.augment_factor}x multiplier)...")
        
        original_count = len(self.video_sequences)
        augmented_videos = list(self.video_sequences)
        augmented_labels = list(self.labels)
        
        for i in tqdm(range(original_count), desc="Augmenting data"):
            video_seq = self.video_sequences[i]
            label = self.labels[i]
            
            for _ in range(self.augment_factor):
                # Augment video sequence
                aug_video = self._augment_video_sequence(video_seq)
                augmented_videos.append(aug_video)
                augmented_labels.append(label)
        
        self.video_sequences = augmented_videos
        self.labels = augmented_labels
        
        print(f"Augmentation complete: {original_count} -> {len(self.video_sequences)} videos")
    
    def _augment_video_sequence(self, video_sequence):
        """Apply augmentation to a video sequence"""
        augmented_sequence = video_sequence.copy()
        
        # Randomly choose augmentation type
        aug_types = ["brightness", "contrast", "rotation", "noise", "temporal"]
        augmentation_type = np.random.choice(aug_types)
        
        if augmentation_type == "brightness":
            brightness_factor = np.random.uniform(0.7, 1.3)
            augmented_sequence = np.clip(augmented_sequence * brightness_factor, 0, 1)
            
        elif augmentation_type == "contrast":
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(augmented_sequence, axis=(1, 2, 3), keepdims=True)
            augmented_sequence = np.clip((augmented_sequence - mean) * contrast_factor + mean, 0, 1)
            
        elif augmentation_type == "rotation":
            angle = np.random.uniform(-5, 5)
            h, w = augmented_sequence.shape[1:3]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            for i in range(len(augmented_sequence)):
                augmented_sequence[i] = cv2.warpAffine(
                    augmented_sequence[i], rotation_matrix, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                
        elif augmentation_type == "noise":
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, augmented_sequence.shape)
            augmented_sequence = np.clip(augmented_sequence + noise, 0, 1)
            
        elif augmentation_type == "temporal":
            # Randomly drop or repeat some frames
            if np.random.random() < 0.5:
                # Drop random frames
                drop_indices = np.random.choice(len(augmented_sequence), 
                                             size=max(1, len(augmented_sequence) // 10), 
                                             replace=False)
                keep_indices = [i for i in range(len(augmented_sequence)) if i not in drop_indices]
                augmented_sequence = augmented_sequence[keep_indices]
                
                # Pad back to original length
                while len(augmented_sequence) < len(video_sequence):
                    augmented_sequence = np.append(augmented_sequence, [augmented_sequence[-1]], axis=0)
        
        return augmented_sequence
    
    def __len__(self):
        return len(self.video_sequences)
    
    def __getitem__(self, idx):
        video = torch.FloatTensor(self.video_sequences[idx]).permute(0, 3, 1, 2)  # (seq, C, H, W)
        label = torch.LongTensor([self.labels[idx]])
        
        if self.transform:
            # Apply transforms to each frame
            transformed_frames = []
            for frame in video:
                transformed_frames.append(self.transform(frame))
            video = torch.stack(transformed_frames)
        
        return video, label

class CNNLSTMModel(nn.Module):
    """CNN+LSTM model for video classification using PyTorch"""
    
    def __init__(self, num_classes, sequence_length=30, input_size=(224, 224)):
        super(CNNLSTMModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Pre-trained CNN backbone (EfficientNet)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Freeze backbone for transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features
        
        # Temporal processing layers
        self.temporal_conv = nn.Conv1d(feature_dim, 512, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through CNN
        x = x.view(-1, c, h, w)  # (batch*seq, c, h, w)
        features = self.backbone(x)  # (batch*seq, feature_dim)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)  # (batch, seq, feature_dim)
        
        # Temporal convolution
        x = features.transpose(1, 2)  # (batch, feature_dim, seq)
        x = torch.relu(self.temporal_bn(self.temporal_conv(x)))
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # (batch, seq, 512)
        
        # LSTM layers
        x, _ = self.lstm1(x)  # (batch, seq, 512)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)  # (batch, seq, 256)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 256)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CNNOnlyVideoSASLTrainer:
    """
    CNN-only PyTorch-based video SASL training system
    """
    
    def __init__(self, video_dataset_path="video_dataset", sequence_length=30, input_size=(224, 224), 
                 epochs=100, batch_size=4, augmentation_factor=0, 
                 learning_rate=0.001, num_workers=4):
        """
        Initialize the PyTorch CNN-only video-based SASL trainer
        """
        self.video_dataset_path = Path(video_dataset_path)
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.augmentation_factor = augmentation_factor
        self.learning_rate = learning_rate
        self.num_workers = min(num_workers, mp_cpu.cpu_count())
        self.num_classes = 0
        self.class_names = []
        
        # Create cache directory
        self.cache_dir = Path("outputs/video_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        if augmentation_factor < 0 or augmentation_factor > 3:
            raise ValueError("Augmentation factor must be between 0 and 3")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0")
        
        print(f"🚀 CNN-Only VideoSASLTrainer initialized:")
        print(f"  📁 Video dataset: {video_dataset_path}")
        print(f"  🎬 Sequence length: {sequence_length} frames")
        print(f"  📏 Input size: {input_size}")
        print(f"  🔄 Training epochs: {epochs}")
        print(f"  📦 Batch size: {batch_size}")
        print(f"  🔀 Augmentation factor: {augmentation_factor}x")
        print(f"  📈 Learning rate: {learning_rate}")
        print(f"  💻 Device: {device}")
        print(f"  👥 Workers: {self.num_workers}")
        
        # Estimate training time
        estimated_time_mins = epochs * (1 + augmentation_factor) * 0.3
        print(f"  ⏱️  Estimated training time: {estimated_time_mins:.0f}-{estimated_time_mins*2:.0f} minutes")
        
        # Create dataset directory
        self.video_dataset_path.mkdir(exist_ok=True)
    
    def load_video_dataset(self):
        """Load all videos with parallel processing"""
        print("\\nLoading video dataset with parallel processing...")
        start_time = time.time()
        
        # Get class directories
        class_dirs = [d for d in self.video_dataset_path.iterdir() if d.is_dir() and d.name != '.gitkeep']
        class_dirs.sort()
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.video_dataset_path}")
        
        print(f"Found {len(class_dirs)} SASL classes")
        
        # Collect all video files
        all_video_tasks = []
        class_names = []
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Get video files for this class
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(class_dir.glob(ext))
            
            print(f"  {class_name}: {len(video_files)} videos")
            
            # Add to processing tasks
            for video_file in video_files:
                task = (video_file, self.sequence_length, self.input_size, self.cache_dir)
                all_video_tasks.append(task)
        
        self.num_classes = len(class_names)
        self.class_names = class_names
        
        print(f"\\nProcessing {len(all_video_tasks)} videos with {self.num_workers} workers...")
        
        video_sequences = []
        labels = []
        cached_count = 0
        processed_count = 0
        
        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in all_video_tasks}
            
            with tqdm(total=len(all_video_tasks), desc="Processing videos", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                for future in as_completed(futures):
                    video_path, video_seq, was_cached = future.result()
                    
                    if video_seq is not None:
                        video_sequences.append(video_seq)
                        
                        # Determine label from path
                        class_name = Path(video_path).parent.name
                        label = class_names.index(class_name)
                        labels.append(label)
                        
                        if was_cached:
                            cached_count += 1
                        else:
                            processed_count += 1
                    
                    pbar.update(1)
        
        elapsed_time = time.time() - start_time
        
        print(f"\\nDataset loading complete!")
        print(f"  Total time: {elapsed_time:.1f}s")
        print(f"  Cached videos: {cached_count}")
        print(f"  Processed videos: {processed_count}")
        print(f"  Total loaded: {len(video_sequences)} videos")
        print(f"  Classes: {len(class_names)}")
        
        return video_sequences, labels, class_names
    
    def create_output_directories(self):
        """Create organized output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/training_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "confusion_matrices").mkdir(exist_ok=True)
        
        return self.output_dir
    
    def train_model(self):
        """Train CNN+LSTM model"""
        print("\\n" + "="*80)
        print("SASL CNN-ONLY VIDEO TRAINING")
        print("="*80)
        
        # Load dataset
        video_sequences, labels, class_names = self.load_video_dataset()
        
        if len(video_sequences) == 0:
            raise ValueError("No videos loaded. Check your dataset.")
        
        # Create output directories
        output_dir = self.create_output_directories()
        print(f"\\nOutput directory created: {output_dir}")
        
        print(f"\\nDataset Information:")
        print(f"  Video sequences: {len(video_sequences)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class names: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        # Split data
        video_train, video_test, labels_train, labels_test = train_test_split(
            video_sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\\nData split:")
        print(f"  Training: {len(video_train)} videos")
        print(f"  Testing: {len(video_test)} videos")
        
        # Create datasets and data loaders
        print(f"\\nCreating PyTorch datasets...")
        
        train_dataset = SASLVideoDataset(
            video_train, labels_train, augment_factor=self.augmentation_factor
        )
        
        test_dataset = SASLVideoDataset(video_test, labels_test)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=0
        )
        
        # Create model
        print(f"\\nCreating CNN+LSTM model...")
        model = CNNLSTMModel(self.num_classes, self.sequence_length, self.input_size).to(device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)
        
        print(f"Model created and moved to {device}")
        
        # Training results storage
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0
        patience_counter = 0
        
        # Training loop
        print(f"\\n" + "="*60)
        print("TRAINING CNN+LSTM MODEL")
        print("="*60)
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, (videos, labels_batch) in enumerate(train_pbar):
                videos = videos.to(device)
                labels_batch = labels_batch.squeeze().to(device)
                
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
                
                # Update progress bar
                train_acc = 100 * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_acc:.1f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
                for videos, labels_batch in val_pbar:
                    videos = videos.to(device)
                    labels_batch = labels_batch.squeeze().to(device)
                    
                    outputs = model(videos)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
                    
                    # Update progress bar
                    val_acc = 100 * val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{val_acc:.1f}%'
                    })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100 * train_correct / train_total
            epoch_val_loss = val_loss / len(test_loader)
            epoch_val_acc = 100 * val_correct / val_total
            
            # Store metrics
            training_history['train_loss'].append(epoch_train_loss)
            training_history['train_acc'].append(epoch_train_acc)
            training_history['val_loss'].append(epoch_val_loss)
            training_history['val_acc'].append(epoch_val_acc)
            
            # Learning rate scheduling
            scheduler.step(epoch_val_loss)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{self.epochs}:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
            print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.state_dict(), 
                          self.output_dir / "models" / "best_sasl_cnn_lstm_model.pth")
                print(f"  *** New best model saved! Accuracy: {best_acc:.2f}% ***")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 15:
                print(f"\\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Save final results
        results = {
            'training_history': training_history,
            'final_accuracy': best_acc,
            'num_classes': self.num_classes,
            'class_names': class_names,
            'dataset_size': len(video_sequences),
            'epochs_trained': epoch + 1,
            'batch_size': self.batch_size,
            'augmentation_factor': self.augmentation_factor,
            'training_date': datetime.now().isoformat()
        }
        
        # Save class names
        with open(self.output_dir / "results" / "class_names.json", 'w') as f:
            json.dump(class_names, f, indent=2)
        
        # Save comprehensive results
        with open(self.output_dir / "results" / "cnn_training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n" + "="*80)
        print("CNN-ONLY TRAINING COMPLETE!")
        print("="*80)
        print(f"Final Results:")
        print(f"  CNN+LSTM Accuracy: {best_acc:.1f}%")
        print(f"\\nOutput Directory: {self.output_dir}")
        print(f"  Model: {self.output_dir / 'models' / 'best_sasl_cnn_lstm_model.pth'}")
        print(f"  Results: {self.output_dir / 'results' / 'class_names.json'}")
        
        return model

if __name__ == "__main__":
    print("SASL CNN-Only Video Training System")
    print("=" * 60)
    
    # Example usage
    trainer = CNNOnlyVideoSASLTrainer(
        video_dataset_path="video_dataset",
        sequence_length=30,
        input_size=(224, 224),
        epochs=50,
        batch_size=4,
        augmentation_factor=1
    )
    
    model = trainer.train_model()
    
    if model is not None:
        print("SUCCESS: CNN-only model trained successfully!")
    else:
        print("Training failed. Check your dataset.")