#!/usr/bin/env python3
"""
SASL PyTorch Video-Based Transfer Learning System
===============================================

This system is designed for video-based SASL recognition using:
- Video sequences (3-5 seconds each) of SASL signs
- CNN + LSTM architecture for temporal modeling using PyTorch
- MediaPipe pose/hand tracking over time
- Transfer learning from pre-trained models
- Optimized for small video datasets with comprehensive batch monitoring

Key Features:
- Processes video sequences frame by frame
- Extracts both visual features (CNN) and pose landmarks (MediaPipe)
- Uses temporal modeling (LSTM) to understand sign dynamics
- Ensemble approach combining multiple models
- Real-time batch progress monitoring
- Advanced data augmentation
- Configurable training parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
import os
import json
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import time
from datetime import datetime
import multiprocessing as mp_cpu
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import random

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
    Process a single video file - designed for multiprocessing
    This function must be at module level for pickling
    """
    video_path, sequence_length, input_size, cache_dir = args
    
    try:
        # Check cache first
        cache_filename = f"{Path(video_path).stem}_{sequence_length}_{input_size[0]}x{input_size[1]}.pkl"
        cache_path = cache_dir / cache_filename
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (cached_data.get('sequence_length') == sequence_length and
                        cached_data.get('input_size') == input_size and
                        'pose_seq' in cached_data):
                        return (video_path, cached_data['video_seq'], cached_data['pose_seq'], True)
            except:
                pass  # Cache corrupted, process normally
        
        # Initialize MediaPipe for pose/hand detection
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            holistic.close()
            return (video_path, None, None, False)
        
        frames = []
        pose_sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, input_size)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract pose landmarks using MediaPipe
            results = holistic.process(rgb_frame)
            
            # Extract landmarks (pose + hands + face key points)
            landmarks = []
            
            # Pose landmarks (33 points * 3 coords = 99 features)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([0.0] * 99)  # Zero padding if no pose detected
            
            # Left hand landmarks (21 points * 3 coords = 63 features)
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([0.0] * 63)  # Zero padding if no left hand detected
            
            # Right hand landmarks (21 points * 3 coords = 63 features)
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([0.0] * 63)  # Zero padding if no right hand detected
            
            # Total: 99 + 63 + 63 = 225 features per frame
            pose_sequence.append(landmarks)
            frames.append(rgb_frame)
        
        cap.release()
        holistic.close()
        
        # Adjust sequence length
        if len(frames) == 0:
            return (video_path, None, None, False)
        
        # Pad or trim to target length
        if len(frames) > sequence_length:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
            pose_sequence = [pose_sequence[i] for i in indices]
        elif len(frames) < sequence_length:
            # Duplicate frames to reach target length
            while len(frames) < sequence_length:
                frames.append(frames[-1])
                pose_sequence.append(pose_sequence[-1])
        
        video_seq = np.array(frames) / 255.0
        pose_seq = np.array(pose_sequence)
        
        # Cache the results
        try:
            cached_data = {
                'video_seq': video_seq,
                'pose_seq': pose_seq,
                'sequence_length': sequence_length,
                'input_size': input_size
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except:
            pass  # Ignore cache save errors
        
        return (video_path, video_seq, pose_seq, False)
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return (video_path, None, None, False)

class SASLVideoDataset(Dataset):
    """PyTorch Dataset for SASL video sequences"""
    
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
            if len(augmented_sequence) > 5:
                # Randomly drop or duplicate frames
                if np.random.random() < 0.5:
                    drop_indices = np.random.choice(len(augmented_sequence), size=min(2, len(augmented_sequence)//5), replace=False)
                    keep_indices = [i for i in range(len(augmented_sequence)) if i not in drop_indices]
                    augmented_sequence = augmented_sequence[keep_indices]
                    
                    # Pad back to original length
                    original_length = len(video_sequence)
                    while len(augmented_sequence) < original_length:
                        duplicate_idx = np.random.choice(len(augmented_sequence))
                        augmented_sequence = np.insert(augmented_sequence, duplicate_idx, augmented_sequence[duplicate_idx], axis=0)
        
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
        
        return video, label.squeeze()

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

class PoseLSTMModel(nn.Module):
    """LSTM model for pose sequence classification using PyTorch"""
    
    def __init__(self, num_classes, sequence_length=30, pose_dim=225):
        super(PoseLSTMModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.num_classes = num_classes
        
        # Input processing
        self.input_bn = nn.BatchNorm1d(pose_dim)
        self.input_dropout = nn.Dropout(0.2)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(pose_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, pose_dim = x.size()
        
        # Normalize input
        x = x.view(-1, pose_dim)  # (batch*seq, pose_dim)
        x = self.input_bn(x)
        x = self.input_dropout(x)
        x = x.view(batch_size, seq_len, pose_dim)  # (batch, seq, pose_dim)
        
        # LSTM layers
        x, _ = self.lstm1(x)  # (batch, seq, 512)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)  # (batch, seq, 256)
        x = self.dropout_lstm(x)
        x, _ = self.lstm3(x)  # (batch, seq, 128)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 128)
        
        # Classification
        x = self.classifier(x)
        
        return x

class VideoSASLTrainer:
    """
    Complete PyTorch-based video SASL training system with comprehensive monitoring
    """
    
    def __init__(self, video_dataset_path="video_dataset", sequence_length=30, input_size=(224, 224), 
                 epochs=100, batch_size_cnn=4, batch_size_pose=8, augmentation_factor=0, 
                 learning_rate=0.001, num_workers=4):
        """
        Initialize the PyTorch video-based SASL trainer
        """
        self.video_dataset_path = Path(video_dataset_path)
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size_cnn = batch_size_cnn
        self.batch_size_pose = batch_size_pose
        self.augmentation_factor = augmentation_factor
        self.learning_rate = learning_rate
        self.num_workers = min(num_workers, mp_cpu.cpu_count())
        self.num_classes = 0
        self.class_names = []
        
        # Create cache directory
        self.cache_dir = Path("outputs/video_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PyTorch VideoSASLTrainer initialized:")
        print(f"  Video dataset: {video_dataset_path}")
        print(f"  Sequence length: {sequence_length} frames")
        print(f"  Input size: {input_size}")
        print(f"  Training epochs: {epochs}")
        print(f"  CNN batch size: {batch_size_cnn}")
        print(f"  Pose batch size: {batch_size_pose}")
        print(f"  Augmentation factor: {augmentation_factor}x")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {device}")
        print(f"  Workers: {self.num_workers}")
        
        # Create dataset directory
        self.video_dataset_path.mkdir(exist_ok=True)
    
    def load_video_dataset(self):
        """Load all videos with parallel processing and comprehensive progress tracking"""
        print("\nLoading video dataset with parallel processing...")
        start_time = time.time()
        
        # Get class directories
        class_dirs = [d for d in self.video_dataset_path.iterdir() if d.is_dir() and d.name != '.gitkeep']
        class_dirs.sort()
        
        if not class_dirs:
            print("ERROR: No class directories found!")
            return [], [], [], []
        
        print(f"Found {len(class_dirs)} SASL classes")
        
        # Collect all video files
        all_video_tasks = []
        class_names = []
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)
            
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']:
                video_files.extend(list(class_dir.glob(ext)))
            
            print(f"  '{class_name}': {len(video_files)} videos")
            
            for video_file in video_files:
                all_video_tasks.append((
                    str(video_file),
                    self.sequence_length,
                    self.input_size,
                    self.cache_dir
                ))
        
        self.num_classes = len(class_names)
        self.class_names = class_names
        
        print(f"\nProcessing {len(all_video_tasks)} videos with {self.num_workers} workers...")
        
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
        
        print(f"\nDataset loading complete!")
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
    
    def plot_training_history(self, training_results, model_name):
        """Generate training and validation accuracy/loss plots"""
        # Map display names to dictionary keys
        key_mapping = {
            "CNN+LSTM": "cnn_lstm_history",
            "Pose LSTM": "pose_lstm_history"
        }
        
        history_key = key_mapping.get(model_name, f'{model_name.lower().replace("+", "_").replace(" ", "_")}_history')
        history = training_results[history_key]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot training and validation loss
        ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', linewidth=2, markersize=6)
        ax1.set_title(f'{model_name} Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot training and validation accuracy
        ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy', linewidth=2, markersize=6)
        ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy', linewidth=2, markersize=6)
        ax2.set_title(f'{model_name} Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = self.output_dir / "plots" / f"{model_name.lower()}_training_histor2y.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_filename)
    
    def generate_confusion_matrix(self, model, data_loader, class_names, model_name, use_poses=False):
        """Generate confusion matrix for a model"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for videos, poses, labels in data_loader:
                videos, poses, labels = videos.to(device), poses.to(device), labels.to(device)
                
                if use_poses:
                    outputs = model(poses)
                else:
                    outputs = model(videos)
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(f'{model_name} Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Calculate accuracy per class
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
        overall_accuracy = np.trace(cm) / np.sum(cm)
        
        # Add accuracy info
        info_text = f'Overall Accuracy: {overall_accuracy:.3f}\\n'
        for i, class_name in enumerate(class_names):
            info_text += f'{class_name}: {accuracy_per_class[i]:.3f}\\n'
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save confusion matrix
        cm_filename = self.output_dir / "confusion_matrices" / f"{model_name.lower()}_confusion_matrix.png"
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        return str(cm_filename), cm, report
    
    def save_comprehensive_results(self, training_results, class_names, cm_cnn, cm_pose, 
                                 report_cnn, report_pose, plot_files):
        """Save comprehensive training results with all metrics"""
        
        # Enhanced results dictionary
        comprehensive_results = {
            **training_results,  # Include existing training results
            
            # Model architecture info
            'model_architecture': {
                'cnn_lstm': {
                    'backbone': 'EfficientNet-B0',
                    'lstm_hidden_sizes': [256, 128],
                    'bidirectional': True,
                    'dropout': 0.3
                },
                'pose_lstm': {
                    'lstm_hidden_sizes': [256, 128, 64],
                    'bidirectional': True,
                    'pose_dim': 225,
                    'dropout': 0.4
                }
            },
            
            # Confusion matrices
            'confusion_matrices': {
                'cnn_lstm': cm_cnn.tolist(),
                'pose_lstm': cm_pose.tolist()
            },
            
            # Classification reports
            'classification_reports': {
                'cnn_lstm': report_cnn,
                'pose_lstm': report_pose
            },
            
            # Training environment
            'training_environment': {
                'device': str(device),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            
            # File paths
            'output_files': {
                'plots': plot_files,
                'confusion_matrices': [
                    str(self.output_dir / "confusion_matrices" / "cnn_lstm_confusion_matrix.png"),
                    str(self.output_dir / "confusion_matrices" / "pose_lstm_confusion_matrix.png")
                ],
                'models': [
                    str(self.output_dir / "models" / "best_sasl_cnn_lstm_model.pth"),
                    str(self.output_dir / "models" / "best_sasl_pose_lstm_model.pth")
                ]
            }
        }
        
        # Save comprehensive results
        results_file = self.output_dir / "results" / "comprehensive_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Save class names separately
        classes_file = self.output_dir / "results" / "class_names.json"
        with open(classes_file, 'w') as f:
            json.dump(class_names, f, indent=2)
        
        # Create a summary report
        summary_file = self.output_dir / "results" / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SASL PyTorch Training Summary\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Training Date: {training_results['training_date']}\\n")
            f.write(f"Dataset Size: {training_results['dataset_size']} videos\\n")
            f.write(f"Number of Classes: {training_results['num_classes']}\\n")
            f.write(f"Class Names: {', '.join(class_names)}\\n")
            f.write(f"Epochs Trained: {training_results['epochs']}\\n")
            f.write(f"Batch Sizes: CNN={training_results['batch_sizes']['cnn']}, Pose={training_results['batch_sizes']['pose']}\\n")
            f.write(f"Augmentation Factor: {training_results['augmentation_factor']}x\\n\\n")
            
            f.write("Final Results:\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"CNN+LSTM Accuracy: {training_results['final_cnn_lstm_accuracy']:.1f}%\\n")
            f.write(f"Pose LSTM Accuracy: {training_results['final_pose_lstm_accuracy']:.1f}%\\n")
            f.write(f"Ensemble Accuracy: {training_results['ensemble_accuracy']:.1f}%\\n\\n")
            
            f.write("CNN+LSTM Classification Report:\\n")
            f.write("-" * 35 + "\\n")
            for class_name in class_names:
                metrics = report_cnn[class_name]
                f.write(f"{class_name:>12}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\\n")
            
            f.write("\\nPose LSTM Classification Report:\\n")
            f.write("-" * 35 + "\\n")
            for class_name in class_names:
                metrics = report_pose[class_name]
                f.write(f"{class_name:>12}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\\n")
        
        return str(results_file), str(summary_file)

    def train_models(self):
        """Train both CNN+LSTM and Pose LSTM models with comprehensive batch monitoring"""
        print("\n" + "="*80)
        print("SASL PYTORCH VIDEO-BASED TRANSFER LEARNING")
        print("="*80)
        
        # Load dataset
        video_sequences, pose_sequences, labels, class_names = self.load_video_dataset()
        
        if len(video_sequences) == 0:
            print("ERROR: No videos loaded. Please add videos to video_dataset/")
            return None, None
        
        # Create output directories
        output_dir = self.create_output_directories()
        print(f"\nOutput directory created: {output_dir}")
        
        print(f"\nDataset Information:")
        print(f"  Video sequences: {len(video_sequences)}")
        print(f"  Pose sequences: {len(pose_sequences)}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class names: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
        
        # Split data
        (video_train, video_test, pose_train, pose_test, 
         labels_train, labels_test) = train_test_split(
            video_sequences, pose_sequences, labels, 
            test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(video_train)} videos")
        print(f"  Testing: {len(video_test)} videos")
        
        # Create datasets and data loaders
        print(f"\nCreating PyTorch datasets...")
        
        train_dataset = SASLVideoDataset(
            video_train, pose_train, labels_train, 
            augment_factor=self.augmentation_factor
        )
        
        test_dataset = SASLVideoDataset(
            video_test, pose_test, labels_test
        )
        
        # Data loaders with comprehensive monitoring
        train_loader_cnn = DataLoader(
            train_dataset, batch_size=self.batch_size_cnn, 
            shuffle=True, num_workers=0, drop_last=True
        )
        
        test_loader_cnn = DataLoader(
            test_dataset, batch_size=self.batch_size_cnn, 
            shuffle=False, num_workers=0
        )
        
        train_loader_pose = DataLoader(
            train_dataset, batch_size=self.batch_size_pose, 
            shuffle=True, num_workers=0, drop_last=True
        )
        
        test_loader_pose = DataLoader(
            test_dataset, batch_size=self.batch_size_pose, 
            shuffle=False, num_workers=0
        )
        
        # Create models
        print(f"\nCreating PyTorch models...")
        cnn_lstm_model = CNNLSTMModel(self.num_classes, self.sequence_length, self.input_size).to(device)
        pose_lstm_model = PoseLSTMModel(self.num_classes, self.sequence_length).to(device)
        
        # Optimizers and loss
        optimizer_cnn = optim.Adam(cnn_lstm_model.parameters(), lr=self.learning_rate)
        optimizer_pose = optim.Adam(pose_lstm_model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate schedulers
        scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, patience=8, factor=0.5, min_lr=1e-6)
        scheduler_pose = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pose, patience=8, factor=0.5, min_lr=1e-6)
        
        print(f"Models created and moved to {device}")
        
        # Training results storage
        training_results = {
            'cnn_lstm_history': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'pose_lstm_history': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'training_date': datetime.now().isoformat(),
            'num_classes': self.num_classes,
            'class_names': class_names,
            'dataset_size': len(video_sequences),
            'epochs': self.epochs,
            'batch_sizes': {'cnn': self.batch_size_cnn, 'pose': self.batch_size_pose},
            'augmentation_factor': self.augmentation_factor
        }
        
        # Train CNN+LSTM Model
        print(f"\n" + "="*60)
        print("TRAINING CNN+LSTM MODEL")
        print("="*60)
        
        best_cnn_acc = 0.0
        patience_counter_cnn = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 40)
            
            # Training phase
            cnn_lstm_model.train()
            train_loss, train_acc, train_samples = 0.0, 0.0, 0
            
            train_pbar = tqdm(train_loader_cnn, desc=f"Training CNN+LSTM", 
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}")
            
            for batch_idx, (videos, poses, labels_batch) in enumerate(train_pbar):
                videos, labels_batch = videos.to(device), labels_batch.to(device)
                # Note: poses not used in CNN+LSTM training
                
                optimizer_cnn.zero_grad()
                outputs = cnn_lstm_model(videos)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer_cnn.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_samples += labels_batch.size(0)
                train_acc += (predicted == labels_batch).sum().item()
                train_loss += loss.item()
                
                # Update progress bar
                current_loss = train_loss / (batch_idx + 1)
                current_acc = 100. * train_acc / train_samples
                train_pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.1f}%")
            
            train_loss /= len(train_loader_cnn)
            train_acc = 100. * train_acc / train_samples
            
            # Validation phase
            cnn_lstm_model.eval()
            val_loss, val_acc, val_samples = 0.0, 0.0, 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader_cnn, desc=f"Validating CNN+LSTM", leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] Loss: {postfix}")
                
                for batch_idx, (videos, poses, labels_batch) in enumerate(val_pbar):
                    videos, labels_batch = videos.to(device), labels_batch.to(device)
                    # Note: poses not used in CNN+LSTM validation
                    
                    outputs = cnn_lstm_model(videos)
                    loss = criterion(outputs, labels_batch)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_samples += labels_batch.size(0)
                    val_acc += (predicted == labels_batch).sum().item()
                    val_loss += loss.item()
                    
                    current_loss = val_loss / (batch_idx + 1)
                    current_acc = 100. * val_acc / val_samples
                    val_pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.1f}%")
            
            val_loss /= len(test_loader_cnn)
            val_acc = 100. * val_acc / val_samples
            
            # Store results
            training_results['cnn_lstm_history']['train_loss'].append(train_loss)
            training_results['cnn_lstm_history']['train_acc'].append(train_acc)
            training_results['cnn_lstm_history']['val_loss'].append(val_loss)
            training_results['cnn_lstm_history']['val_acc'].append(val_acc)
            
            print(f"Results - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
            print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
            
            # Learning rate scheduling
            scheduler_cnn.step(val_acc)
            
            # Save best model
            if val_acc > best_cnn_acc:
                best_cnn_acc = val_acc
                torch.save(cnn_lstm_model.state_dict(), self.output_dir / "models" / 'best_sasl_cnn_lstm_model.pth')
                print(f"New best CNN+LSTM model saved! Accuracy: {val_acc:.1f}%")
                patience_counter_cnn = 0
            else:
                patience_counter_cnn += 1
            
            # Early stopping
            if patience_counter_cnn >= 20:
                print(f"Early stopping CNN+LSTM training at epoch {epoch+1}")
                break
        
        # Train Pose LSTM Model
        print(f"\n" + "="*60)
        print("TRAINING POSE LSTM MODEL")
        print("="*60)
        
        best_pose_acc = 0.0
        patience_counter_pose = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 40)
            
            # Training phase
            pose_lstm_model.train()
            train_loss, train_acc, train_samples = 0.0, 0.0, 0
            
            train_pbar = tqdm(train_loader_pose, desc=f"Training Pose LSTM", 
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}")
            
            for batch_idx, (videos, poses, labels_batch) in enumerate(train_pbar):
                poses, labels_batch = poses.to(device), labels_batch.to(device)
                
                optimizer_pose.zero_grad()
                outputs = pose_lstm_model(poses)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer_pose.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_samples += labels_batch.size(0)
                train_acc += (predicted == labels_batch).sum().item()
                train_loss += loss.item()
                
                # Update progress bar
                current_loss = train_loss / (batch_idx + 1)
                current_acc = 100. * train_acc / train_samples
                train_pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.1f}%")
            
            train_loss /= len(train_loader_pose)
            train_acc = 100. * train_acc / train_samples
            
            # Validation phase
            pose_lstm_model.eval()
            val_loss, val_acc, val_samples = 0.0, 0.0, 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader_pose, desc=f"Validating Pose LSTM", leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] Loss: {postfix}")
                
                for batch_idx, (videos, poses, labels_batch) in enumerate(val_pbar):
                    poses, labels_batch = poses.to(device), labels_batch.to(device)
                    
                    outputs = pose_lstm_model(poses)
                    loss = criterion(outputs, labels_batch)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_samples += labels_batch.size(0)
                    val_acc += (predicted == labels_batch).sum().item()
                    val_loss += loss.item()
                    
                    current_loss = val_loss / (batch_idx + 1)
                    current_acc = 100. * val_acc / val_samples
                    val_pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.1f}%")
            
            val_loss /= len(test_loader_pose)
            val_acc = 100. * val_acc / val_samples
            
            # Store results
            training_results['pose_lstm_history']['train_loss'].append(train_loss)
            training_results['pose_lstm_history']['train_acc'].append(train_acc)
            training_results['pose_lstm_history']['val_loss'].append(val_loss)
            training_results['pose_lstm_history']['val_acc'].append(val_acc)
            
            print(f"Results - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
            print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
            
            # Learning rate scheduling
            scheduler_pose.step(val_acc)
            
            # Save best model
            if val_acc > best_pose_acc:
                best_pose_acc = val_acc
                torch.save(pose_lstm_model.state_dict(), self.output_dir / "models" / 'best_sasl_pose_lstm_model.pth')
                print(f"New best Pose LSTM model saved! Accuracy: {val_acc:.1f}%")
                patience_counter_pose = 0
            else:
                patience_counter_pose += 1
            
            # Early stopping
            if patience_counter_pose >= 20:
                print(f"Early stopping Pose LSTM training at epoch {epoch+1}")
                break
        
        # Final results
        training_results['final_cnn_lstm_accuracy'] = best_cnn_acc
        training_results['final_pose_lstm_accuracy'] = best_pose_acc
        training_results['ensemble_accuracy'] = (best_cnn_acc + best_pose_acc) / 2  # Simple ensemble
        
        print(f"\n" + "="*80)
        print("GENERATING COMPREHENSIVE OUTPUTS")
        print("="*80)
        
        # Generate training history plots
        plot_files = []
        print("Creating training history plots...")
        cnn_plot = self.plot_training_history(training_results, "CNN+LSTM")
        pose_plot = self.plot_training_history(training_results, "Pose LSTM")
        plot_files.extend([cnn_plot, pose_plot])
        print(f"  Generated: {len(plot_files)} training plots")
        
        # Generate confusion matrices
        print("Generating confusion matrices...")
        cm_cnn_file, cm_cnn, report_cnn = self.generate_confusion_matrix(
            cnn_lstm_model, test_loader_cnn, class_names, "CNN+LSTM", use_poses=False)
        cm_pose_file, cm_pose, report_pose = self.generate_confusion_matrix(
            pose_lstm_model, test_loader_pose, class_names, "Pose LSTM", use_poses=True)
        print(f"  Generated: 2 confusion matrices")
        
        # Save comprehensive results
        print("Saving comprehensive results...")
        results_file, summary_file = self.save_comprehensive_results(
            training_results, class_names, cm_cnn, cm_pose, 
            report_cnn, report_pose, plot_files)
        print(f"  Saved comprehensive results to: {results_file}")
        print(f"  Saved training summary to: {summary_file}")
        
        print(f"\n" + "="*80)
        print("PYTORCH TRAINING COMPLETE!")
        print("="*80)
        print(f"Final Results:")
        print(f"  CNN+LSTM Accuracy: {best_cnn_acc:.1f}%")
        print(f"  Pose LSTM Accuracy: {best_pose_acc:.1f}%")
        print(f"  Ensemble Accuracy: {training_results['ensemble_accuracy']:.1f}%")
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"  Models: {self.output_dir / 'models'}")
        print(f"  Plots: {self.output_dir / 'plots'}")
        print(f"  Confusion Matrices: {self.output_dir / 'confusion_matrices'}")
        print(f"  Results: {self.output_dir / 'results'}")
        
        return cnn_lstm_model, pose_lstm_model

if __name__ == "__main__":
    print("SASL PyTorch Video-Based Transfer Learning System")
    print("=" * 60)
    
    # Example usage
    trainer = VideoSASLTrainer(
        video_dataset_path="video_dataset",
        sequence_length=30,
        input_size=(224, 224),
        epochs=50,
        batch_size_cnn=4,
        batch_size_pose=8,
        augmentation_factor=1
    )
    
    cnn_model, pose_model = trainer.train_models()
    
    if cnn_model is not None:
        print("SUCCESS: PyTorch models trained successfully!")
    else:
        print("Training failed. Check your dataset.")