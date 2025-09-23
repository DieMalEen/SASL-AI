#!/usr/bin/env python3
"""
SASL Unified Multi-Modal Video-Based Transfer Learning System
===========================================================

This system uses a unified multi-modal architecture for video-based SASL recognition:
- Video sequences (3-5 seconds each) of SASL signs
- Unified CNN+LSTM and Pose LSTM architecture with learned fusion weights
- MediaPipe pose/hand tracking over time
- Transfer learning from pre-trained models
- Early fusion for optimal modality combination
- Optimized for small video datasets with comprehensive batch monitoring

Key Features:
- Single unified model instead of separate CNN+LSTM and Pose LSTM models
- Learnable fusion weights (no fixed 50-50 averaging)
- Joint feature learning between visual and pose modalities
- More efficient inference (single forward pass)
- Advanced data augmentation
- Configurable training parameters through menu system
- Better accuracy through end-to-end optimization
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
                        cached_data.get('input_size') == input_size):
                        return (video_path, cached_data['video_seq'], cached_data['pose_seq'], True)
            except:
                pass  # Cache corrupted, process normally
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
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
            frames.append(rgb_frame)
            
            # Process pose and hands
            pose_results = pose_detector.process(rgb_frame)
            hand_results = hand_detector.process(rgb_frame)
            
            landmarks = []
            
            # Add pose landmarks (33 points × 3 coordinates = 99 features)
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([0.0] * 99)
            
            # Add hand landmarks (2 hands × 21 points × 3 coordinates = 126 features)
            hands_added = 0
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if hands_added < 2:
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        hands_added += 1
            
            # Pad with zeros if less than 2 hands detected
            while hands_added < 2:
                landmarks.extend([0.0] * 63)  # 21 points × 3 coordinates
                hands_added += 1
            
            pose_sequence.append(landmarks[:225])  # Ensure consistent size
        
        cap.release()
        
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
        
        # Cleanup
        pose_detector.close()
        hand_detector.close()
        
        return (video_path, video_seq, pose_seq, False)
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return (video_path, None, None, False)

class SASLVideoDataset(Dataset):
    """PyTorch Dataset for SASL video sequences"""
    
    def __init__(self, video_sequences, pose_sequences, labels, transform=None, augment_factor=0):
        self.video_sequences = video_sequences
        self.pose_sequences = pose_sequences
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
        augmented_poses = list(self.pose_sequences)
        augmented_labels = list(self.labels)
        
        for i in tqdm(range(original_count), desc="Augmenting data"):
            video_seq = self.video_sequences[i]
            pose_seq = self.pose_sequences[i]
            label = self.labels[i]
            
            for _ in range(self.augment_factor):
                # Augment video sequence
                aug_video = self._augment_video_sequence(video_seq)
                augmented_videos.append(aug_video)
                
                # Augment pose sequence
                aug_pose = self._augment_pose_sequence(pose_seq)
                augmented_poses.append(aug_pose)
                
                augmented_labels.append(label)
        
        self.video_sequences = augmented_videos
        self.pose_sequences = augmented_poses
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
    
    def _augment_pose_sequence(self, pose_sequence):
        """Apply augmentation to pose landmarks"""
        augmented_pose = pose_sequence.copy()
        
        # Add small noise to pose landmarks
        noise_std = 0.02
        noise = np.random.normal(0, noise_std, augmented_pose.shape)
        
        # Only add noise to non-zero landmarks
        mask = augmented_pose != 0
        augmented_pose[mask] += noise[mask]
        
        # Ensure landmarks stay in valid range [0, 1]
        augmented_pose = np.clip(augmented_pose, 0, 1)
        
        return augmented_pose
    
    def __len__(self):
        return len(self.video_sequences)
    
    def __getitem__(self, idx):
        video = torch.FloatTensor(self.video_sequences[idx]).permute(0, 3, 1, 2)  # (seq, C, H, W)
        pose = torch.FloatTensor(self.pose_sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        if self.transform:
            # Apply transforms to each frame
            transformed_frames = []
            for frame in video:
                transformed_frames.append(self.transform(frame))
            video = torch.stack(transformed_frames)
        
        return video, pose, label.squeeze()

class UnifiedSASLModel(nn.Module):
    """
    Unified Multi-Modal SASL Model
    
    Combines CNN+LSTM (video) and Pose LSTM (landmarks) into a single model.
    Uses early fusion with learnable weights to optimally combine visual and pose features.
    """
    
    def __init__(self, num_classes, sequence_length=30, input_size=(224, 224), pose_dim=225):
        super(UnifiedSASLModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.pose_dim = pose_dim
        self.num_classes = num_classes
        
        print(f"Building Unified SASL Model:")
        print(f"  Input size: {input_size}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Pose dimension: {pose_dim}")
        print(f"  Number of classes: {num_classes}")
        
        # =================================================================
        # VISUAL PROCESSING BRANCH (CNN + Temporal Convolution)
        # =================================================================
        print("  Initializing visual processing branch...")
        
        # Pre-trained CNN backbone (EfficientNet)
        self.cnn_backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Freeze backbone for transfer learning
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
        
        # Get CNN feature dimension
        self.cnn_feature_dim = self.cnn_backbone.num_features  # 1280 for EfficientNet-B0
        print(f"    CNN feature dimension: {self.cnn_feature_dim}")
        
        # Temporal processing for CNN features
        self.visual_temporal_conv = nn.Conv1d(self.cnn_feature_dim, 512, kernel_size=3, padding=1)
        self.visual_temporal_bn = nn.BatchNorm1d(512)
        self.visual_dropout = nn.Dropout(0.3)
        
        # =================================================================
        # POSE PROCESSING BRANCH
        # =================================================================
        print("  Initializing pose processing branch...")
        
        # Pose feature processing
        self.pose_input_bn = nn.BatchNorm1d(pose_dim)
        self.pose_input_dropout = nn.Dropout(0.2)
        
        # Project pose features to match visual feature dimension
        self.pose_projection = nn.Sequential(
            nn.Linear(pose_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # =================================================================
        # FUSION LAYER
        # =================================================================
        print("  Initializing fusion layer...")
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Initialize equally
        
        # Combined feature dimension after fusion
        self.fused_dim = 512
        
        # =================================================================
        # UNIFIED TEMPORAL MODELING (LSTM)
        # =================================================================
        print("  Initializing unified LSTM layers...")
        
        # Multi-layer LSTM for temporal modeling of fused features
        self.unified_lstm1 = nn.LSTM(self.fused_dim, 256, bidirectional=True, batch_first=True, dropout=0.3)
        self.unified_lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.3)
        self.unified_lstm3 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.2)
        
        # =================================================================
        # CLASSIFICATION HEAD
        # =================================================================
        print("  Initializing classification head...")
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
        print(f"  Unified model initialized successfully!")
        
    def forward(self, videos, poses):
        """
        Forward pass through unified model
        
        Args:
            videos: (batch_size, seq_len, channels, height, width)
            poses: (batch_size, seq_len, pose_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len = videos.size(0), videos.size(1)
        
        # =================================================================
        # PROCESS VISUAL FEATURES
        # =================================================================
        
        # Process each frame through CNN
        videos_flat = videos.view(-1, *videos.shape[2:])  # (batch*seq, C, H, W)
        visual_features = self.cnn_backbone(videos_flat)  # (batch*seq, cnn_feature_dim)
        
        # Reshape back to sequence
        visual_features = visual_features.view(batch_size, seq_len, self.cnn_feature_dim)  # (batch, seq, cnn_feature_dim)
        
        # Temporal convolution for visual features
        visual_temp = visual_features.transpose(1, 2)  # (batch, cnn_feature_dim, seq)
        visual_temp = torch.relu(self.visual_temporal_bn(self.visual_temporal_conv(visual_temp)))
        visual_temp = self.visual_dropout(visual_temp)
        visual_features_processed = visual_temp.transpose(1, 2)  # (batch, seq, 512)
        
        # =================================================================
        # PROCESS POSE FEATURES
        # =================================================================
        
        # Normalize and process pose input
        poses_flat = poses.view(-1, self.pose_dim)  # (batch*seq, pose_dim)
        poses_normalized = self.pose_input_bn(poses_flat)
        poses_normalized = self.pose_input_dropout(poses_normalized)
        
        # Project pose features to match visual dimension
        pose_features_projected = self.pose_projection(poses_normalized)  # (batch*seq, 512)
        pose_features_processed = pose_features_projected.view(batch_size, seq_len, 512)  # (batch, seq, 512)
        
        # =================================================================
        # FUSION LAYER
        # =================================================================
        
        # Apply learnable fusion weights (softmax to ensure they sum to 1)
        fusion_weights_normalized = torch.softmax(self.fusion_weights, dim=0)
        
        # Weighted fusion of visual and pose features
        fused_features = (fusion_weights_normalized[0] * visual_features_processed + 
                         fusion_weights_normalized[1] * pose_features_processed)
        
        # =================================================================
        # UNIFIED TEMPORAL MODELING
        # =================================================================
        
        # Process fused features through unified LSTM layers
        x, _ = self.unified_lstm1(fused_features)  # (batch, seq, 512)
        x, _ = self.unified_lstm2(x)  # (batch, seq, 256)
        x, _ = self.unified_lstm3(x)  # (batch, seq, 128)
        
        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, 128)
        
        # =================================================================
        # CLASSIFICATION
        # =================================================================
        
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits
    
    def get_fusion_weights(self):
        """Get current learned fusion weights"""
        weights = torch.softmax(self.fusion_weights, dim=0)
        return {
            'visual_weight': weights[0].item(),
            'pose_weight': weights[1].item()
        }

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
        pose_sequences = []
        labels = []
        cached_count = 0
        processed_count = 0
        
        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in all_video_tasks}
            
            with tqdm(total=len(all_video_tasks), desc="Processing videos", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                for future in as_completed(futures):
                    video_path, video_seq, pose_seq, was_cached = future.result()
                    
                    if video_seq is not None and pose_seq is not None:
                        video_sequences.append(video_seq)
                        pose_sequences.append(pose_seq)
                        
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
        
        return video_sequences, pose_sequences, labels, class_names
    
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
            "Pose LSTM": "pose_lstm_history",
            "Unified Model": "unified_model_history"
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
        plot_filename = self.output_dir / "plots" / f"{model_name.lower().replace(' ', '_')}_training_history.png"
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
                
                if model_name == "Unified Model":
                    # Unified model uses both videos and poses
                    outputs = model(videos, poses)
                elif use_poses:
                    # Pose-only models
                    outputs = model(poses)
                else:
                    # Video-only models
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
        cm_filename = self.output_dir / "confusion_matrices" / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        return str(cm_filename), cm, report
    
    def save_comprehensive_results(self, training_results, class_names, cm_unified, cm_legacy, 
                                 report_unified, report_legacy, plot_files):
        """Save comprehensive training results with all metrics"""
        
        # Enhanced results dictionary
        comprehensive_results = {
            **training_results,  # Include existing training results
            
            # Model architecture info
            'model_architecture': {
                'unified_model': {
                    'visual_branch': 'EfficientNet-B0',
                    'pose_branch': '3-layer Bidirectional LSTM',
                    'fusion_method': 'Learnable weighted fusion',
                    'unified_lstm': '3-layer Bidirectional LSTM',
                    'pose_dim': 225,
                    'sequence_length': 30,
                    'input_size': [224, 224]
                }
            },
            
            # Confusion matrices
            'confusion_matrices': {
                'unified_model': cm_unified.tolist() if cm_unified is not None else None
            },
            
            # Classification reports
            'classification_reports': {
                'unified_model': report_unified if report_unified is not None else None
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
                    str(self.output_dir / "confusion_matrices" / "unified_model_confusion_matrix.png")
                ],
                'models': [
                    str(self.output_dir / "models" / "best_sasl_cnn_lstm_model.pth")
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
            f.write("SASL PyTorch Unified Model Training Summary\\n")
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
            f.write(f"Unified Model Accuracy: {training_results['final_unified_accuracy']:.1f}%\\n")
            final_weights = training_results['final_fusion_weights']
            f.write(f"Final Fusion Weights - Visual: {final_weights['visual_weight']:.3f}, Pose: {final_weights['pose_weight']:.3f}\\n\\n")
            
            f.write("Unified Model Classification Report:\\n")
            f.write("-" * 40 + "\\n")
            if report_unified:
                for class_name in class_names:
                    metrics = report_unified[class_name]
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
        
        # Create unified model
        print(f"\nCreating unified PyTorch model...")
        unified_model = UnifiedSASLModel(self.num_classes, self.sequence_length, self.input_size).to(device)
        
        # Optimizer and loss
        optimizer_unified = optim.Adam(
            filter(lambda p: p.requires_grad, unified_model.parameters()), 
            lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning rate scheduler
        scheduler_unified = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_unified, patience=8, factor=0.5, min_lr=1e-6, verbose=True
        )
        
        print(f"Unified model created and moved to {device}")
        
        # Training results storage
        training_results = {
            'unified_model_history': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []},
            'training_date': datetime.now().isoformat(),
            'num_classes': self.num_classes,
            'class_names': class_names,
            'dataset_size': len(video_sequences),
            'epochs': self.epochs,
            'batch_sizes': {'cnn': self.batch_size_cnn, 'pose': self.batch_size_pose},
            'augmentation_factor': self.augmentation_factor,
            'fusion_weights_history': []
        }
        
        # Train Unified Model
        print(f"\n" + "="*60)
        print("TRAINING UNIFIED SASL MODEL")
        print("="*60)
        
        best_unified_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 40)
            
            # Training phase
            unified_model.train()
            train_loss, train_acc, train_samples = 0.0, 0.0, 0
            
            train_pbar = tqdm(train_loader_cnn, desc=f"Training Unified Model", 
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}")
            
            for batch_idx, (videos, poses, labels_batch) in enumerate(train_pbar):
                videos, poses, labels_batch = videos.to(device), poses.to(device), labels_batch.to(device)
                
                optimizer_unified.zero_grad()
                outputs = unified_model(videos, poses)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer_unified.step()
                
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
            unified_model.eval()
            val_loss, val_acc, val_samples = 0.0, 0.0, 0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader_cnn, desc=f"Validating Unified Model", leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] Loss: {postfix}")
                
                for batch_idx, (videos, poses, labels_batch) in enumerate(val_pbar):
                    videos, poses, labels_batch = videos.to(device), poses.to(device), labels_batch.to(device)
                    
                    outputs = unified_model(videos, poses)
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
            
            # Get current fusion weights
            fusion_weights = unified_model.get_fusion_weights()
            training_results['fusion_weights_history'].append({
                'epoch': epoch + 1,
                'visual_weight': fusion_weights['visual_weight'],
                'pose_weight': fusion_weights['pose_weight']
            })
            
            # Store results
            training_results['unified_model_history']['train_loss'].append(train_loss)
            training_results['unified_model_history']['train_acc'].append(train_acc)
            training_results['unified_model_history']['val_loss'].append(val_loss)
            training_results['unified_model_history']['val_acc'].append(val_acc)
            
            print(f"Results - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
            print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
            print(f"           Fusion Weights - Visual: {fusion_weights['visual_weight']:.3f}, Pose: {fusion_weights['pose_weight']:.3f}")
            
            # Learning rate scheduling
            scheduler_unified.step(val_loss)
            
            # Save best model
            if val_acc > best_unified_acc:
                best_unified_acc = val_acc
                torch.save(unified_model.state_dict(), self.output_dir / "models" / 'best_sasl_cnn_lstm_model.pth')
                print(f"New best unified model saved! Accuracy: {val_acc:.1f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 15:
                print(f"Early stopping unified training at epoch {epoch+1}")
                break
        
        # Final results
        training_results['final_unified_accuracy'] = best_unified_acc
        training_results['final_fusion_weights'] = unified_model.get_fusion_weights()
        
        print(f"\n" + "="*80)
        print("GENERATING COMPREHENSIVE OUTPUTS")
        print("="*80)
        
        # Generate training history plots
        plot_files = []
        print("Creating training history plots...")
        unified_plot = self.plot_training_history(training_results, "Unified Model")
        plot_files.append(unified_plot)
        print(f"  Generated: {len(plot_files)} training plots")
        
        # Generate confusion matrices
        print("Generating confusion matrices...")
        cm_unified_file, cm_unified, report_unified = self.generate_confusion_matrix(
            unified_model, test_loader_cnn, class_names, "Unified Model", use_poses=True)
        print(f"  Generated: 1 confusion matrix")
        
        # Save comprehensive results
        print("Saving comprehensive results...")
        results_file, summary_file = self.save_comprehensive_results(
            training_results, class_names, cm_unified, None, 
            report_unified, None, plot_files)
        print(f"  Saved comprehensive results to: {results_file}")
        print(f"  Saved training summary to: {summary_file}")
        
        print(f"\n" + "="*80)
        print("PYTORCH UNIFIED TRAINING COMPLETE!")
        print("="*80)
        print(f"Final Results:")
        print(f"  Unified Model Accuracy: {best_unified_acc:.1f}%")
        final_weights = training_results['final_fusion_weights']
        print(f"  Final Fusion Weights - Visual: {final_weights['visual_weight']:.3f}, Pose: {final_weights['pose_weight']:.3f}")
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"  Models: {self.output_dir / 'models'}")
        print(f"  Plots: {self.output_dir / 'plots'}")
        print(f"  Confusion Matrices: {self.output_dir / 'confusion_matrices'}")
        print(f"  Results: {self.output_dir / 'results'}")
        
        return unified_model

if __name__ == "__main__":
    print("SASL PyTorch Video-Based Transfer Learning System")
    print("=" * 60)
    
    try:
        # Example usage with error monitoring
        trainer = VideoSASLTrainer(
            video_dataset_path="video_dataset",
            sequence_length=30,
            input_size=(224, 224),
            epochs=50,
            batch_size_cnn=4,
            batch_size_pose=8,
            augmentation_factor=1
        )
        
        # Show GPU memory info
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
            print(f"GPU Memory: {memory_gb:.1f}GB total")
            torch.cuda.empty_cache()
        
        print(f"\nStarting training (errors will be displayed, no screen clearing)...")
        print("If training crashes around epoch 50, check the error message below:")
        print("=" * 60)
        
        cnn_model, pose_model = trainer.train_models()
        
        if cnn_model is not None:
            print("\n" + "="*60)
            print("SUCCESS: PyTorch models trained successfully!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("TRAINING FAILED - Check error messages above")
            print("="*60)
    
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR DURING TRAINING - NO SCREEN CLEARING")
        print("="*60)
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        if "out of memory" in str(e).lower():
            print(f"\nGPU OUT OF MEMORY ERROR!")
            print(f"Solutions:")
            print(f"1. Reduce batch_size_cnn from 4 to 2")
            print(f"2. Reduce batch_size_pose from 8 to 4")
            print(f"3. Reduce sequence_length from 30 to 20")
        
        print(f"\nFull error details:")
        import traceback
        traceback.print_exc()
        
        print(f"\nPress Enter to continue...")
        input()