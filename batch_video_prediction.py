#!/usr/bin/env python3
"""
SASL Batch Video Prediction Script
==================================

Standalone script for predicting hand signs on multiple videos in a folder.
Uses trained CNN+LSTM models to classify videos.

Usage:
    python batch_video_prediction.py

The script will prompt for:
- Path to trained model (.pth file)
- Path to class names file (.json file) 
- Path to folder containing videos to predict
- Optional output file for results

Example:
    Model path: outputs/training_20251016_130042/models/best_sasl_cnn_lstm_model.pth
    Classes path: outputs/training_20251016_130042/results/class_names.json
    Video folder: test_videos/
    Output file: predictions_results.json (optional)
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import the exact same model classes from the training file to ensure compatibility
try:
    from video_cnn_only_training import CNNLSTMModel, CombinedCNNHandModel, HandLandmarkLSTM, MEDIAPIPE_AVAILABLE
    print("Using CNNLSTMModel, CombinedCNNHandModel, and HandLandmarkLSTM from training file for compatibility")
    
    # Import MediaPipe if available
    if MEDIAPIPE_AVAILABLE:
        import mediapipe as mp
        print("MediaPipe available for hand landmark extraction")
    else:
        print("MediaPipe not available - will use CNN-only fallback")
        
except ImportError as e:
    print(f"Warning: Could not import models from training file: {e}")
    print("Using fallback model definitions matching training file architecture")
    MEDIAPIPE_AVAILABLE = False
    
    # Fallback model definitions - EXACT COPIES from training file
    import timm
    
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

    class HandLandmarkLSTM(nn.Module):
        """LSTM model for hand landmark sequences"""
        
        def __init__(self, num_classes, sequence_length=30, hand_features=126):
            super(HandLandmarkLSTM, self).__init__()
            
            self.sequence_length = sequence_length
            self.hand_features = hand_features  # 2 hands * 21 landmarks * 3 coords
            self.num_classes = num_classes
            
            # Input processing
            self.input_projection = nn.Linear(hand_features, 256)
            self.input_dropout = nn.Dropout(0.2)
            
            # LSTM layers for temporal modeling
            self.lstm1 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=0.3)
            self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.3)
            
            # Classification layers
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            # x shape: (batch_size, sequence_length, hand_features)
            
            # Project hand features
            x = self.input_projection(x)  # (batch, seq, 256)
            x = torch.relu(x)
            x = self.input_dropout(x)
            
            # LSTM processing
            x, _ = self.lstm1(x)  # (batch, seq, 256)
            x, _ = self.lstm2(x)  # (batch, seq, 128)
            
            # Global average pooling over sequence
            x = torch.mean(x, dim=1)  # (batch, 128)
            
            # Classification
            x = self.classifier(x)
            
            return x

    class CombinedCNNHandModel(nn.Module):
        """Combined model that processes both video frames and hand landmarks"""
        
        def __init__(self, num_classes, lstm_hidden_size=256, lstm_num_layers=2, 
                     hand_lstm_hidden_size=256, hand_lstm_num_layers=2,
                     fusion_hidden_size=512, dropout_rate=0.3):
            super(CombinedCNNHandModel, self).__init__()
            
            self.num_classes = num_classes
            
            # CNN branch for video frames
            self.cnn_branch = CNNLSTMModel(
                num_classes=num_classes,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                dropout_rate=dropout_rate
            )
            
            # Hand landmark branch
            self.hand_branch = HandLandmarkLSTM(
                num_classes=num_classes,
                lstm_hidden_size=hand_lstm_hidden_size,
                lstm_num_layers=hand_lstm_num_layers
            )
            
            # Fusion layer to combine predictions
            self.fusion = nn.Sequential(
                nn.Linear(num_classes * 2, fusion_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(fusion_hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
            # Learnable weights for combining branches
            self.cnn_weight = nn.Parameter(torch.tensor(0.7))
            self.hand_weight = nn.Parameter(torch.tensor(0.3))
        
        def forward(self, video_frames, hand_landmarks):
            # Get predictions from both branches
            cnn_logits = self.cnn_branch(video_frames)
            hand_logits = self.hand_branch(hand_landmarks)
            
            # Combine logits with learnable weights
            combined_features = torch.cat([cnn_logits, hand_logits], dim=1)
            
            # Final prediction through fusion layer
            final_logits = self.fusion(combined_features)
            
            # Also return individual predictions for analysis
            return final_logits, cnn_logits, hand_logits

def extract_hand_landmarks_from_frame(rgb_frame, hands_processor):
    """Extract hand landmarks from a single RGB frame"""
    if not MEDIAPIPE_AVAILABLE or hands_processor is None:
        return [0.0] * 126  # Return zeros if MediaPipe not available
    
    try:
        results = hands_processor.process(rgb_frame)
        
        frame_landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    # Normalize coordinates to [0,1] relative to frame size
                    hand_coords.extend([landmark.x, landmark.y, landmark.z])
                frame_landmarks.extend(hand_coords)
        
        # Pad to consistent size (2 hands * 21 landmarks * 3 coords = 126 features)
        while len(frame_landmarks) < 126:
            frame_landmarks.append(0.0)
        
        # Truncate if somehow more than 126 features
        frame_landmarks = frame_landmarks[:126]
        
        return frame_landmarks
        
    except Exception as e:
        print(f"Error extracting hand landmarks: {e}")
        return [0.0] * 126

def process_video_for_prediction(video_path, sequence_length=30, input_size=(224, 224), extract_hands=True):
    """
    Process a single video file for prediction, extracting both video frames and hand landmarks
    
    Args:
        video_path: Path to video file
        sequence_length: Number of frames to extract
        input_size: Target size for frames
        extract_hands: Whether to extract hand landmarks
        
    Returns:
        tuple: (video_sequence, hand_landmarks_sequence) or (None, None) if failed
    """
    try:
        # Initialize MediaPipe hands if needed
        hands_processor = None
        if extract_hands and MEDIAPIPE_AVAILABLE:
            mp_hands = mp.solutions.hands
            hands_processor = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None
        
        frames = []
        hand_landmarks_seq = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame to input size
            frame = cv2.resize(frame, input_size)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
            # Extract hand landmarks if requested
            if extract_hands:
                hand_landmarks = extract_hand_landmarks_from_frame(rgb_frame, hands_processor)
                hand_landmarks_seq.append(hand_landmarks)
            else:
                hand_landmarks_seq.append([0.0] * 126)
        
        cap.release()
        
        # Clean up MediaPipe
        if hands_processor is not None:
            hands_processor.close()
        
        if len(frames) == 0:
            return None, None
        
        # Adjust sequence length for both video and hand landmarks
        if len(frames) > sequence_length:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
            frames = [frames[i] for i in indices]
            hand_landmarks_seq = [hand_landmarks_seq[i] for i in indices]
        elif len(frames) < sequence_length:
            # Repeat frames and landmarks to reach target length
            while len(frames) < sequence_length:
                frames.append(frames[-1])  # Repeat last frame
                hand_landmarks_seq.append(hand_landmarks_seq[-1])  # Repeat last landmarks
        
        # Convert to numpy arrays and normalize (MATCH TRAINING FORMAT EXACTLY)
        video_sequence = np.array(frames, dtype=np.float32) / 255.0  # Normalize first
        hand_landmarks_sequence = np.array(hand_landmarks_seq, dtype=np.float32)
        
        # Convert to tensor format matching training: (seq, C, H, W)
        # Training uses: torch.FloatTensor(video).permute(0, 3, 1, 2)
        # We need to match this exactly
        video_sequence = np.transpose(video_sequence, (0, 3, 1, 2))  # (seq, H, W, C) -> (seq, C, H, W)
        
        return video_sequence, hand_landmarks_sequence
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None, None

def predict_videos_in_folder(model_path, class_names_path, video_folder_path, 
                           output_predictions_path=None, confidence_threshold=0.5):
    """
    Predict hand signs for all videos in a folder using either CNN-only or Combined CNN+Hand model
    
    Args:
        model_path: Path to the trained model (.pth file)
        class_names_path: Path to class names JSON file
        video_folder_path: Path to folder containing videos to predict
        output_predictions_path: Optional path to save predictions JSON file
        confidence_threshold: Minimum confidence threshold for valid predictions
        
    Returns:
        dict: Dictionary with video filenames as keys and prediction results as values
    """
    print(f"Batch Video Prediction System")
    print("=" * 50)
    
    # Load class names
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"ERROR: Could not load class names from {class_names_path}: {e}")
        return None
    
    # Load trained model and determine type
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's a combined model by looking for hand-related keys in state_dict
        state = checkpoint.get('model_state_dict', checkpoint)
        state_keys = list(state.keys())
        is_combined_model = any(
            key.startswith('hand_branch') or key.startswith('fusion') or key.startswith('hand_')
            for key in state_keys
        )
        
        if is_combined_model:
            # Load Combined CNN+Hand model
            model = CombinedCNNHandModel(num_classes=len(class_names)).to(device)
            print(f"Loading Combined CNN+Hand model with {len(class_names)} classes")
        else:
            # Load CNN-only model
            model = CNNLSTMModel(num_classes=len(class_names)).to(device)
            print(f"Loading CNN-only model with {len(class_names)} classes")
        
        # Load state dict (handle both checkpoint format and direct state dict)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}: {e}")
        return None
    
    # Find all video files in the folder
    video_folder = Path(video_folder_path)
    if not video_folder.exists():
        print(f"ERROR: Video folder does not exist: {video_folder_path}")
        return None
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_folder.glob(f"*{ext}"))
        video_files.extend(video_folder.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"ERROR: No video files found in {video_folder_path}")
        print(f"Supported formats: {', '.join(video_extensions)}")
        return None
    
    print(f"Found {len(video_files)} video files to process...")
    print(f"Model type: {'Combined CNN+Hand' if is_combined_model else 'CNN-only'}")
    
    predictions_results = {}
    
    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            # Process video to get frame sequence (and hand landmarks if needed)
            video_sequence, hand_landmarks_sequence = process_video_for_prediction(
                video_file, extract_hands=is_combined_model
            )
            
            if video_sequence is None:
                predictions_results[video_file.name] = {
                    'status': 'failed',
                    'error': 'Could not process video',
                    'prediction': None,
                    'confidence': 0.0,
                    'all_probabilities': None
                }
                continue
            
            # Convert to tensors with CORRECT format matching training and real-time recognition
            # Input shape: (seq, C, H, W) -> need (batch, seq, C, H, W) after unsqueeze
            video_tensor = torch.tensor(video_sequence).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                if is_combined_model:
                    # Use both video and hand landmarks
                    hand_tensor = torch.tensor(hand_landmarks_sequence).float().unsqueeze(0).to(device)
                    outputs, cnn_outputs, hand_outputs = model(video_tensor, hand_tensor)
                else:
                    # Use video only
                    outputs = model(video_tensor)
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_class_idx = predicted_class_idx.item()
                predicted_class = class_names[predicted_class_idx]
                
                # Get all class probabilities
                all_probs = {class_names[i]: probabilities[0][i].item() 
                           for i in range(len(class_names))}
            
            # Store results
            predictions_results[video_file.name] = {
                'status': 'success',
                'prediction': predicted_class,
                'confidence': confidence,
                'above_threshold': confidence >= confidence_threshold,
                'all_probabilities': all_probs,
                'model_type': 'Combined CNN+Hand' if is_combined_model else 'CNN-only'
            }
            
            # Print result
            status_indicator = "+" if confidence >= confidence_threshold else "?"
            print(f"  {status_indicator} {video_file.name}: {predicted_class} ({confidence:.3f})")
            
        except Exception as e:
            predictions_results[video_file.name] = {
                'status': 'error',
                'error': str(e),
                'prediction': None,
                'confidence': 0.0,
                'all_probabilities': None
            }
            print(f"  ERROR processing {video_file.name}: {e}")
    
    # Generate summary
    successful_predictions = sum(1 for r in predictions_results.values() 
                               if r['status'] == 'success')
    high_confidence_predictions = sum(1 for r in predictions_results.values() 
                                    if r['status'] == 'success' and r['above_threshold'])
    
    print(f"\nPrediction Summary:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully processed: {successful_predictions}")
    print(f"  High confidence (>{confidence_threshold:.2f}): {high_confidence_predictions}")
    print(f"  Failed/Errors: {len(video_files) - successful_predictions}")
    print(f"  Model type: {'Combined CNN+Hand' if is_combined_model else 'CNN-only'}")
    
    # Save results to file if requested
    if output_predictions_path:
        try:
            output_path = Path(output_predictions_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            results_with_metadata = {
                'metadata': {
                    'model_path': str(model_path),
                    'class_names_path': str(class_names_path),
                    'video_folder_path': str(video_folder_path),
                    'confidence_threshold': confidence_threshold,
                    'model_type': 'Combined CNN+Hand' if is_combined_model else 'CNN-only',
                    'total_videos': len(video_files),
                    'successful_predictions': successful_predictions,
                    'high_confidence_predictions': high_confidence_predictions,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'predictions': predictions_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_with_metadata, f, indent=2)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"WARNING: Could not save results to file: {e}")
    
    return predictions_results

def main():
    print("SASL Batch Video Prediction")
    print("=" * 50)
    
    print("Automatically detecting trained models and setting up prediction...")
    
    # Automatically find the latest trained model
    outputs_dir = Path("outputs")
    model_path = None
    classes_path = None
    
    if outputs_dir.exists():
        # Find the latest training directory
        training_dirs = [d for d in outputs_dir.iterdir() 
                        if d.is_dir() and d.name.startswith("training_")]
        
        if training_dirs:
            latest_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            print(f"Found latest training session: {latest_dir.name}")
            
            # Check for model file
            models_dir = latest_dir / "models"
            potential_model = models_dir / "best_sasl_cnn_lstm_model.pth"
            
            if potential_model.exists():
                model_path = potential_model
                print(f"Found model: {model_path}")
            else:
                print(f"ERROR: Model file not found in {models_dir}")
                return
            
            # Check for class names file
            results_dir = latest_dir / "results"
            potential_classes = results_dir / "class_names.json"
            
            if not potential_classes.exists():
                # Try alternative location
                alt_results_dir = Path("outputs/training_20251016_130042/results")
                potential_classes = alt_results_dir / "class_names.json"
            
            if potential_classes.exists():
                classes_path = potential_classes
                print(f"Found classes: {classes_path}")
            else:
                print(f"ERROR: Class names file not found")
                print(f"Checked: {results_dir / 'class_names.json'}")
                print(f"Checked: {alt_results_dir / 'class_names.json'}")
                return
        else:
            print("ERROR: No training sessions found in outputs directory")
            print("Please train a model first using option 2 in main.py")
            return
    else:
        print("ERROR: Outputs directory not found")
        print("Please train a model first using option 2 in main.py")
        return
    
    # Set default video folder path
    default_video_folder = Path("test_videos")
    
    # Ask user for video folder (with default option)
    print(f"\nVideo folder setup:")
    if default_video_folder.exists():
        print(f"Default video folder found: {default_video_folder}")
        use_default = input("Use default test_videos folder? (y/n, default=y): ").strip().lower()
        if use_default in ['', 'y', 'yes']:
            video_folder = default_video_folder
        else:
            video_folder_input = input("Enter path to video folder: ").strip()
            video_folder = Path(video_folder_input)
    else:
        print(f"Default video folder ({default_video_folder}) not found.")
        video_folder_input = input("Enter path to folder containing videos to predict: ").strip()
        if not video_folder_input:
            print("ERROR: Video folder path is required!")
            return
        video_folder = Path(video_folder_input)
    
    if not video_folder.exists():
        print(f"ERROR: Video folder not found: {video_folder}")
        print("Please create the folder and add some video files to predict.")
        return
    
    if not video_folder.is_dir():
        print(f"ERROR: Path is not a directory: {video_folder}")
        return
    
    # Optional output file with default
    default_output = Path("prediction_results.json")
    print(f"\nOutput file setup:")
    save_results = input(f"Save results to file? (y/n, default=y): ").strip().lower()
    if save_results in ['', 'y', 'yes']:
        output_file_input = input(f"Output file path (default={default_output}): ").strip()
        output_file = Path(output_file_input) if output_file_input else default_output
    else:
        output_file = None
    
    # Optional confidence threshold with default
    threshold_input = input("Confidence threshold (0.0-1.0, default=0.5): ").strip()
    try:
        confidence_threshold = float(threshold_input) if threshold_input else 0.5
        if not (0.0 <= confidence_threshold <= 1.0):
            confidence_threshold = 0.5
            print(f"Invalid threshold, using default: 0.5")
    except ValueError:
        confidence_threshold = 0.5
        print(f"Invalid threshold, using default: 0.5")
    
    print(f"\nFinal Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Classes: {classes_path}")
    print(f"  Video folder: {video_folder}")
    print(f"  Output file: {output_file if output_file else 'None (display only)'}")
    print(f"  Confidence threshold: {confidence_threshold}")
    
    confirm = input(f"\nProceed with batch prediction? (y/n, default=y): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Prediction cancelled.")
        return
    
    try:
        print(f"\nStarting batch prediction...")
        
        # Run batch prediction
        results = predict_videos_in_folder(
            model_path=str(model_path),
            class_names_path=str(classes_path),
            video_folder_path=str(video_folder),
            output_predictions_path=str(output_file) if output_file else None,
            confidence_threshold=confidence_threshold
        )
        
        if results:
            print(f"\nBatch prediction completed successfully!")
            if output_file:
                print(f"Detailed results saved to: {output_file}")
            
            # Show summary of predictions
            print(f"\nPrediction Results Summary:")
            class_counts = {}
            for video_name, result in results.items():
                if result['status'] == 'success' and result['above_threshold']:
                    prediction = result['prediction']
                    class_counts[prediction] = class_counts.get(prediction, 0) + 1
            
            if class_counts:
                print("High-confidence predictions by class:")
                for class_name, count in sorted(class_counts.items()):
                    print(f"  {class_name}: {count} video(s)")
            else:
                print("No high-confidence predictions found.")
                print("Consider lowering the confidence threshold or checking video quality.")
        else:
            print("Batch prediction failed. Check your inputs and try again.")
    
    except KeyboardInterrupt:
        print(f"\nPrediction interrupted by user.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

def quick_predict():
    """
    Quick prediction with all automatic settings - no user input required
    """
    print("SASL Quick Batch Prediction")
    print("=" * 50)
    print("Using automatic settings for fastest prediction...")
    
    # Auto-detect paths
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("ERROR: No outputs directory found. Please train a model first.")
        return False
        
    # Find latest training
    training_dirs = [d for d in outputs_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("training_")]
    
    if not training_dirs:
        print("ERROR: No training sessions found. Please train a model first.")
        return False
        
    latest_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
    model_path = latest_dir / "models" / "best_sasl_cnn_lstm_model.pth"
    classes_path = latest_dir / "results" / "class_names.json"
    
    # Try alternative class names location
    if not classes_path.exists():
        alt_classes_path = Path("outputs/training_20251016_130042/results/class_names.json")
        if alt_classes_path.exists():
            classes_path = alt_classes_path
    
    # Check required files
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False
        
    if not classes_path.exists():
        print(f"ERROR: Class names not found at {classes_path}")
        return False
    
    # Use default video folder
    video_folder = Path("test_videos")
    if not video_folder.exists():
        video_folder.mkdir()
        print(f"Created video folder: {video_folder}")
        print("Please add some video files to this folder and run again.")
        return False
    
    # Default settings
    output_file = Path("quick_prediction_results.json")
    confidence_threshold = 0.5
    
    print(f"Auto-detected settings:")
    print(f"  Model: {model_path}")
    print(f"  Classes: {classes_path}")
    print(f"  Video folder: {video_folder}")
    print(f"  Output: {output_file}")
    print(f"  Confidence: {confidence_threshold}")
    
    # Run prediction
    results = predict_videos_in_folder(
        model_path=str(model_path),
        class_names_path=str(classes_path),
        video_folder_path=str(video_folder),
        output_predictions_path=str(output_file),
        confidence_threshold=confidence_threshold
    )
    
    return results is not None

if __name__ == "__main__":
    import sys
    
    # Check for quick mode argument
    if len(sys.argv) > 1 and sys.argv[1] in ['--quick', '-q']:
        success = quick_predict()
        if not success:
            print("\nQuick prediction failed. Try running without --quick for interactive mode.")
    else:
        main()