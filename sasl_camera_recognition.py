"""
Unified SASL Camera Recognition System
=====================================

Real-time South African Sign Language recognition using unified multi-modal model.
Combines video frames and pose landmarks in a single neural network.

Features:
- Single unified model for both visual and pose features
- Learnable fusion weights for optimal modality combination
- Real-time MediaPipe pose detection and processing
- Optimized inference with early fusion approach
- Confidence-based prediction filtering
- Live webcam recognition with smooth predictions

Author: SASL-AI Team
Date: 2024
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import timm
import json
from collections import deque
from pathlib import Path
import time
import argparse


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
        
        # =================================================================
        # VISUAL PROCESSING BRANCH (CNN + Temporal Convolution)
        # =================================================================
        
        # Pre-trained CNN backbone (EfficientNet)
        self.cnn_backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Freeze backbone for transfer learning
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
        
        # Get CNN feature dimension
        self.cnn_feature_dim = self.cnn_backbone.num_features  # 1280 for EfficientNet-B0
        
        # Temporal processing for CNN features
        self.visual_temporal_conv = nn.Conv1d(self.cnn_feature_dim, 512, kernel_size=3, padding=1)
        self.visual_temporal_bn = nn.BatchNorm1d(512)
        self.visual_dropout = nn.Dropout(0.3)
        
        # =================================================================
        # POSE PROCESSING BRANCH
        # =================================================================
        
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
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Initialize equally
        
        # Combined feature dimension after fusion
        self.fused_dim = 512
        
        # =================================================================
        # UNIFIED TEMPORAL MODELING (LSTM)
        # =================================================================
        
        # Multi-layer LSTM for temporal modeling of fused features
        self.unified_lstm1 = nn.LSTM(self.fused_dim, 256, bidirectional=True, batch_first=True, dropout=0.3)
        self.unified_lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.3)
        self.unified_lstm3 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.2)
        
        # =================================================================
        # CLASSIFICATION HEAD
        # =================================================================
        
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


class UnifiedSASLRecognizer:
    """Real-time SASL recognition using unified multi-modal model"""
    
    def __init__(self, model_path, class_names_path, sequence_length=30, confidence_threshold=0.7):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.input_size = (224, 224)
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        print(f"Unified SASL Recognizer initialized")
        print(f"  Classes: {self.num_classes}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        
        # Load unified model
        self.model = UnifiedSASLModel(
            num_classes=self.num_classes,
            sequence_length=self.sequence_length,
            input_size=self.input_size
        ).to(self.device)
        
        # Load trained weights
        if Path(model_path).exists():
            print(f"  Loading model weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # Print learned fusion weights
            fusion_weights = self.model.get_fusion_weights()
            print(f"  Learned fusion weights - Visual: {fusion_weights['visual_weight']:.3f}, Pose: {fusion_weights['pose_weight']:.3f}")
        else:
            print(f"  WARNING: Model file not found: {model_path}")
            print(f"  Using randomly initialized model")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher accuracy
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize sequence buffers
        self.frame_buffer = deque(maxlen=sequence_length)
        self.pose_buffer = deque(maxlen=sequence_length)
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)  # Last 5 predictions for smoothing
        
        print("Unified SASL Recognizer ready!")
    
    def extract_pose_landmarks(self, image):
        """Extract pose landmarks using MediaPipe"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract 3D landmarks (x, y, z for 33 points + visibility)
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([
                    landmark.x,  # Normalized x coordinate
                    landmark.y,  # Normalized y coordinate  
                    landmark.z,  # Relative depth
                    landmark.visibility  # Visibility score
                ])
            
            # Additional computed features
            # 1. Hand-related landmarks (more important for sign language)
            left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Compute relative positions and angles
            additional_features = [
                # Hand positions relative to shoulders
                left_wrist.x - left_shoulder.x,
                left_wrist.y - left_shoulder.y,
                right_wrist.x - right_shoulder.x,
                right_wrist.y - right_shoulder.y,
                
                # Arm angles (approximate)
                left_elbow.x - left_shoulder.x,
                left_elbow.y - left_shoulder.y,
                right_elbow.x - right_shoulder.x,
                right_elbow.y - right_shoulder.y,
                
                # Distance between hands
                abs(left_wrist.x - right_wrist.x),
                abs(left_wrist.y - right_wrist.y),
                
                # Body orientation features
                (left_shoulder.x + right_shoulder.x) / 2,  # Body center x
                (left_shoulder.y + right_shoulder.y) / 2,  # Body center y
                
                # Movement indicators (will be zero for single frame, computed during sequence processing)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 10 placeholders for movement features
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 9 more to reach 225 total
            ]
            
            landmarks.extend(additional_features)
            
            # Ensure exactly 225 features (33 * 4 + 93 = 225)
            while len(landmarks) < 225:
                landmarks.append(0.0)
            landmarks = landmarks[:225]  # Truncate if too many
            
            return np.array(landmarks, dtype=np.float32), True
        else:
            # Return zeros if no pose detected
            return np.zeros(225, dtype=np.float32), False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for CNN input"""
        # Resize and normalize
        frame_resized = cv2.resize(frame, self.input_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to PyTorch format (C, H, W)
        frame_tensor = torch.from_numpy(frame_normalized.transpose(2, 0, 1))
        
        return frame_tensor
    
    def update_buffers(self, frame, pose_landmarks):
        """Update sequence buffers with new frame and pose data"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffers
        self.frame_buffer.append(processed_frame)
        self.pose_buffer.append(pose_landmarks)
    
    def predict(self):
        """Make prediction using current sequence buffers"""
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0, None
        
        # Prepare batch data
        frames = torch.stack(list(self.frame_buffer)).unsqueeze(0)  # (1, seq_len, C, H, W)
        poses = torch.stack([torch.from_numpy(pose) for pose in self.pose_buffer]).unsqueeze(0)  # (1, seq_len, pose_dim)
        
        # Move to device
        frames = frames.to(self.device)
        poses = poses.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(frames, poses)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get current fusion weights for display
            fusion_weights = self.model.get_fusion_weights()
            
            return predicted_class, confidence_score, fusion_weights
    
    def smooth_prediction(self, prediction, confidence):
        """Apply temporal smoothing to predictions"""
        if prediction is None:
            return None, 0.0
        
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:  # Need at least 3 predictions
            return prediction, confidence
        
        # Find most common prediction in buffer
        predictions = [pred for pred, conf in self.prediction_buffer]
        confidences = [conf for pred, conf in self.prediction_buffer]
        
        # Simple voting - most frequent prediction
        from collections import Counter
        prediction_counts = Counter(predictions)
        most_common_pred = prediction_counts.most_common(1)[0][0]
        
        # Average confidence for the most common prediction
        avg_confidence = np.mean([conf for pred, conf in self.prediction_buffer if pred == most_common_pred])
        
        return most_common_pred, avg_confidence
    
    def run_camera(self, camera_index=0):
        """Run real-time recognition from camera"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\\nStarting camera recognition...")
        print(f"Press 'q' to quit, 's' to screenshot")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Mirror the frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Extract pose landmarks
            pose_landmarks, pose_detected = self.extract_pose_landmarks(frame)
            
            # Update buffers
            self.update_buffers(frame, pose_landmarks)
            
            # Make prediction if we have enough frames
            prediction, confidence, fusion_weights = self.predict()
            
            # Apply smoothing
            if prediction is not None and confidence > self.confidence_threshold:
                smooth_pred, smooth_conf = self.smooth_prediction(prediction, confidence)
            else:
                smooth_pred, smooth_conf = None, 0.0
            
            # Draw pose landmarks
            if pose_detected:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            else:
                fps = 0
            
            # Draw information on frame
            self.draw_info(frame, smooth_pred, smooth_conf, pose_detected, fusion_weights, fps)
            
            # Show frame
            cv2.imshow('Unified SASL Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_info(self, frame, prediction, confidence, pose_detected, fusion_weights, fps):
        """Draw prediction and status information on frame"""
        h, w = frame.shape[:2]
        
        # Create info panel
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Status indicators
        status_y = 25
        cv2.putText(panel, f"Pose: {'OK' if pose_detected else 'NO'}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if pose_detected else (0, 0, 255), 2)
        
        cv2.putText(panel, f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}", 
                   (120, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if fps > 0:
            cv2.putText(panel, f"FPS: {fps:.1f}", 
                       (280, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Prediction
        pred_y = 55
        if prediction and confidence > self.confidence_threshold:
            cv2.putText(panel, f"Sign: {prediction}", 
                       (10, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(panel, f"Confidence: {confidence:.2f}", 
                       (10, pred_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(panel, "Detecting...", 
                       (10, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Fusion weights
        if fusion_weights:
            cv2.putText(panel, f"Visual: {fusion_weights['visual_weight']:.2f} | Pose: {fusion_weights['pose_weight']:.2f}", 
                       (300, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Instructions
        cv2.putText(panel, "Press 'q' to quit, 's' for screenshot", 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine frame and panel
        combined = np.vstack([frame, panel])
        frame[:] = combined[:h]  # Update original frame


class SASLCameraRecognition:
    """
    Backward-compatible wrapper for the unified SASL recognizer
    Maintains the same interface as the original dual-model system
    """
    
    def __init__(self, cnn_model_path, pose_model_path, classes_path, 
                 sequence_length=30, input_size=(224, 224), confidence_threshold=0.3, 
                 show_overlays=True, minimal_ui=False):
        """
        Initialize using the old interface but internally use the unified model
        """
        print("Initializing backward-compatible SASL Camera Recognition")
        print("Note: Using unified model for both CNN and Pose processing")
        
        # Use the unified recognizer internally
        self.unified_recognizer = UnifiedSASLRecognizer(
            model_path=cnn_model_path,  # Use CNN model path (should be unified model)
            class_names_path=classes_path,
            sequence_length=sequence_length,
            confidence_threshold=confidence_threshold
        )
        
        self.show_overlays = show_overlays
        self.minimal_ui = minimal_ui
        
    def run_live_recognition(self):
        """Run live recognition using the old method name"""
        self.unified_recognizer.run_camera()


def main():
    parser = argparse.ArgumentParser(description='Unified SASL Camera Recognition')
    parser.add_argument('--model', type=str, 
                       default='05_OUTPUT_GENERATED/models/best_unified_sasl_model.pth',
                       help='Path to trained unified model')
    parser.add_argument('--classes', type=str,
                       default='03_DATA_CONFIG/class_names.json',
                       help='Path to class names JSON file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for predictions (default: 0.7)')
    parser.add_argument('--sequence-length', type=int, default=30,
                       help='Sequence length for temporal modeling (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    try:
        recognizer = UnifiedSASLRecognizer(
            model_path=args.model,
            class_names_path=args.classes,
            sequence_length=args.sequence_length,
            confidence_threshold=args.confidence
        )
        
        # Run camera recognition
        recognizer.run_camera(camera_index=args.camera)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure you have:")
        print(f"  - Trained model: {args.model}")
        print(f"  - Class names: {args.classes}")
    except Exception as e:
        print(f"Error initializing recognizer: {e}")


if __name__ == "__main__":
    main()