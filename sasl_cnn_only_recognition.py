#!/usr/bin/env python3
"""
SASL CNN-Only Camera Recognition System
=====================================

Streamlined real-time SASL recognition using only CNN+LSTM model.
No pose dependency - more robust, faster, and better performance.

Features:
- Real-time video processing with CNN+LSTM or Combined CNN+Hand Landmarks
- Optional MediaPipe hand detection for enhanced accuracy
- Automatic model detection and fallback to CNN-only
- Better performance with hand landmarks when available
- Faster processing
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from pathlib import Path
import time
from collections import deque
import timm

# Import the models from training file to ensure compatibility
try:
    from video_cnn_only_training import CNNLSTMModel, CombinedCNNHandModel, MEDIAPIPE_AVAILABLE
    print("Successfully imported models from training module")
    
    # MediaPipe for hand detection (if available)
    if MEDIAPIPE_AVAILABLE:
        import mediapipe as mp
        print("MediaPipe available for hand landmark extraction")
    else:
        print("MediaPipe not available - will use CNN-only fallback")
        
except ImportError as e:
    print(f"Warning: Could not import models from training module: {e}")
    print("   Using local model definition (may cause compatibility issues)")
    CNNLSTMModel = None
    CombinedCNNHandModel = None
    MEDIAPIPE_AVAILABLE = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition is now imported from video_cnn_only_training.py
# This ensures compatibility between training and inference

class SASLCNNOnlyCameraRecognition:
    """
    Real-time SASL recognition using CNN+Hand Landmark fusion model
    """
    
    def __init__(self, model_path, classes_path, 
                 sequence_length=30, input_size=(224, 224), confidence_threshold=0.3, 
                 use_hand_landmarks=True):
        """
        Initialize the SASL camera recognition system
        
        Args:
            model_path: Path to PyTorch model (.pth)
            classes_path: Path to class names JSON file
            sequence_length: Number of frames for temporal modeling
            input_size: Input image size for CNN
            confidence_threshold: Minimum confidence for predictions (raised to 0.5)
            use_hand_landmarks: Whether to use hand landmarks (if available)
        """
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.use_hand_landmarks = use_hand_landmarks and MEDIAPIPE_AVAILABLE
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        print(f"Initializing SASL Camera Recognition")
        print(f"Device: {device}")
        print(f"Classes: {self.num_classes}")
        print(f"Sequence length: {sequence_length}")
        print(f"Input size: {input_size}")
        print(f"Hand landmarks: {'Enabled' if self.use_hand_landmarks else 'Disabled'}")
        
        # Initialize MediaPipe if using hand landmarks
        if self.use_hand_landmarks:
            print("Initializing MediaPipe hand detection...")
            mp_hands = mp.solutions.hands
            # Match training thresholds (0.5) to reduce distribution shift
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.hands = None
        
        # Load Model with error handling
        print("Loading model...")
        
        # Determine which model to use
        use_combined_model = self.use_hand_landmarks and CombinedCNNHandModel is not None
        
        if use_combined_model:
            print("Using Combined CNN + Hand Landmark model")
            if CombinedCNNHandModel is None:
                raise ImportError("Could not import CombinedCNNHandModel. Please ensure video_cnn_only_training.py is available.")
            model_class = CombinedCNNHandModel
        else:
            print("Using CNN-only model (fallback)")
            if CNNLSTMModel is None:
                raise ImportError("Could not import CNNLSTMModel. Please ensure video_cnn_only_training.py is available.")
            model_class = CNNLSTMModel
        
        try:
            self.model = model_class(self.num_classes, sequence_length, input_size)
            self.is_combined_model = use_combined_model
            
            # Load model state with proper error handling
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading model weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            
            model_type = "Combined CNN+Hand" if use_combined_model else "CNN-only"
            print(f"{model_type} model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Frame buffer for temporal modeling
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Hand landmarks buffer (if using hand landmarks)
        if self.use_hand_landmarks:
            self.hand_landmarks_buffer = deque(maxlen=sequence_length)
        
        # Prediction smoothing - smaller buffer for more responsive predictions
        self.prediction_buffer = deque(maxlen=5)  # Reduced from 7 to 5 for more responsiveness
        
    # Hand landmarks overlay toggle
        self.show_hand_overlay = False  # Toggle for showing hand landmarks visualization
        
        # Debug information toggle
        self.show_debug_info = False  # Toggle for showing debug information
        
        # Frame counter for debugging
        self.frame_count = 0
        self.prediction_count = 0
        
        # Add debugging info toggle
        self.show_debug_info = False  # Toggle for showing debug information
        
        # Performance tracking
        self.frame_count = 0
        self.prediction_count = 0
        
        # Initialize MediaPipe drawing utilities for hand overlay
        if self.use_hand_landmarks:
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

        # Allow forcing CNN-only predictions even when a combined model is loaded
        self.force_cnn_only = False

        # Optional background segmentation to better match training data
        self.segment_background = False
        self.bg_segmenter = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.bg_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            except Exception:
                self.bg_segmenter = None
        
        recognition_type = "CNN+Hand Landmark" if self.use_hand_landmarks else "CNN-only"
        print(f"{recognition_type} camera recognition system ready!")
    
    def extract_hand_landmarks(self, rgb_frame, return_results=False):
        """Extract hand landmarks from RGB frame"""
        if not self.use_hand_landmarks or self.hands is None:
            if return_results:
                return [0.0] * 126, None
            return [0.0] * 126  # Return zeros if not using hand detection
        
        try:
            results = self.hands.process(rgb_frame)
            
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
            
            if return_results:
                return frame_landmarks, results
            return frame_landmarks
            
        except Exception as e:
            print(f"Error extracting hand landmarks: {e}")
            if return_results:
                return [0.0] * 126, None
            return [0.0] * 126
    
    def predict_sign(self):
        """Make prediction using the loaded model"""
        if len(self.frame_buffer) < self.sequence_length:
            return None, None, []
        
        # Check if we have enough hand landmarks (if using them)
        if (
            self.use_hand_landmarks 
            and not self.force_cnn_only 
            and len(self.hand_landmarks_buffer) < self.sequence_length
        ):
            return None, None, []
        
        # Prepare video sequence with proper normalization (same as training)
        video_sequence = np.array(list(self.frame_buffer)) / 255.0
        video_tensor = torch.FloatTensor(video_sequence).unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
        
        with torch.no_grad():
            if self.is_combined_model and self.use_hand_landmarks:
                # Combined model prediction with hand landmarks
                if self.force_cnn_only:
                    # Run combined model but use only the CNN branch output
                    # Fill a dummy hand tensor to satisfy the forward signature
                    hand_sequence = np.zeros((self.sequence_length, 126), dtype=np.float32)
                    hand_tensor = torch.FloatTensor(hand_sequence).unsqueeze(0).to(device)
                    final_outputs, cnn_outputs, hand_outputs = self.model(video_tensor, hand_tensor)
                    outputs = cnn_outputs
                else:
                    hand_sequence = np.array(list(self.hand_landmarks_buffer))
                    hand_tensor = torch.FloatTensor(hand_sequence).unsqueeze(0).to(device)
                    final_outputs, cnn_outputs, hand_outputs = self.model(video_tensor, hand_tensor)
                    outputs = final_outputs  # Use the combined prediction
                
                # Debug info for combined model
                if self.show_debug_info:
                    cnn_probs = torch.softmax(cnn_outputs, dim=1)
                    hand_probs = torch.softmax(hand_outputs, dim=1)
                    print(f"CNN top prob: {torch.max(cnn_probs).item():.3f}, Hand top prob: {torch.max(hand_probs).item():.3f}")
                    
            else:
                # CNN-only prediction
                outputs = self.model(video_tensor)
            
            probs = torch.softmax(outputs, dim=1)
            
            # Debug: Check if we're getting reasonable predictions
            max_prob = torch.max(probs).item()
            if self.show_debug_info:
                print(f"Frame {self.frame_count}: Max probability: {max_prob:.3f}, Threshold: {self.confidence_threshold}")
            
            # Get top-3 predictions
            top3_probs, top3_indices = torch.topk(probs, min(3, self.num_classes), dim=1)
            
            top3_predictions = []
            for i in range(min(3, self.num_classes)):
                class_idx = top3_indices[0, i].item()
                confidence = top3_probs[0, i].item()
                class_name = self.class_names[class_idx]
                top3_predictions.append((class_name, confidence))
            
            # Best prediction
            best_class = top3_predictions[0][0]
            best_confidence = top3_predictions[0][1]
            
            if self.show_debug_info:
                print(f"Raw prediction: {best_class} ({best_confidence:.3f})")
            
            self.prediction_count += 1
            
            return best_class, best_confidence, top3_predictions
    
    def smooth_prediction(self, prediction, confidence):
        """Simple prediction smoothing to reduce flicker without getting stuck"""
        self.prediction_buffer.append((prediction, confidence))
        
        # If buffer not full, return current prediction
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Simple majority voting with confidence threshold
        prediction_counts = {}
        confidence_sums = {}
        
        for pred, conf in self.prediction_buffer:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                confidence_sums[pred] = 0.0
            prediction_counts[pred] += 1
            confidence_sums[pred] += conf
        
        # Find most frequent prediction
        most_frequent = max(prediction_counts.items(), key=lambda x: x[1])
        most_frequent_pred = most_frequent[0]
        frequency = most_frequent[1]
        
        # Debug information
        if self.show_debug_info:
            print(f"Smoothing: {prediction} -> {most_frequent_pred} (freq: {frequency}/{len(self.prediction_buffer)})")
        
        # Only use smoothing if the prediction appears frequently enough
        if frequency >= 2:  # At least 2 out of last few predictions
            avg_confidence = confidence_sums[most_frequent_pred] / frequency
            return most_frequent_pred, avg_confidence
        
        # Otherwise return current prediction (allows for quick changes)
        return prediction, confidence
    
    def draw_hand_landmarks_overlay(self, frame, hand_results):
        """Draw hand landmarks overlay on the frame"""
        if not self.use_hand_landmarks or not self.show_hand_overlay or hand_results is None:
            return
        
        if not hasattr(self, 'mp_drawing') or not hasattr(self, 'mp_drawing_styles'):
            return
        
        try:
            # Draw hand landmarks and connections
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Add overlay status indicator
                overlay_text = f"Hand Overlay: ON ({len(hand_results.multi_hand_landmarks)} hands detected)"
                cv2.putText(frame, overlay_text, (10, frame.shape[0] - 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # No hands detected
                overlay_text = "Hand Overlay: ON (no hands detected)"
                cv2.putText(frame, overlay_text, (10, frame.shape[0] - 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        except Exception as e:
            print(f"Error drawing hand landmarks overlay: {e}")
    
    def draw_predictions(self, frame, best_prediction, best_confidence, top3_predictions):
        """Draw prediction results on frame"""
        height, width = frame.shape[:2]
        
        # Main prediction
        if best_prediction and best_confidence > self.confidence_threshold:
            # Background for main prediction
            cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 255, 0), 2)
            
            # Main prediction text
            main_text = f"Sign: {best_prediction}"
            conf_text = f"Confidence: {best_confidence:.3f}"
            
            cv2.putText(frame, main_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No confident prediction
            cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 255), 2)
            cv2.putText(frame, "No sign detected", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Top-3 predictions
        if top3_predictions:
            y_start = 100
            cv2.rectangle(frame, (10, y_start), (350, y_start + 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y_start), (350, y_start + 120), (255, 255, 0), 2)
            
            cv2.putText(frame, "Top Predictions:", (20, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, (pred, conf) in enumerate(top3_predictions):
                y_pos = y_start + 45 + i * 25
                text = f"{i+1}. {pred}: {conf:.3f}"
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Instructions
        recognition_type = "CNN+Hand Landmark" if self.use_hand_landmarks else "CNN-Only"
        instructions = [
            f"{recognition_type} Recognition:",
            "- Hold sign clearly for 1-2 seconds",
            "- Ensure good lighting", 
            "- Press 'q' to quit, 'r' to reset",
            "- Press 'd' to toggle debug info",
            "- Press 't' to adjust confidence threshold"
        ]
        
        # Add hand overlay instructions if MediaPipe is available
        if self.use_hand_landmarks:
            instructions.append("- Press 'h' to toggle hand overlay")
        
        y_start = height - 140 if self.use_hand_landmarks else height - 120
        instruction_height = len(instructions) * 20 + 20
        cv2.rectangle(frame, (10, y_start), (450, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, y_start), (450, height - 10), (0, 255, 255), 2)
        
        for i, instruction in enumerate(instructions):
            y_pos = y_start + 20 + i * 20
            cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_live_recognition(self):
        """Run live camera recognition"""
        recognition_type = "CNN+Hand Landmark" if self.use_hand_landmarks else "CNN-only"
        print(f"Starting live {recognition_type} SASL recognition...")
        print("Controls:")
        print("  - Hold signs clearly for 1-2 seconds")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset prediction buffer")
        if self.use_hand_landmarks:
            print("  - Press 'h' to toggle hand landmarks overlay")
            print("  - Press 'm' to toggle CNN-only vs combined predictions")
        print("  - Press 'b' to toggle background segmentation (black background)")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame for model input
                model_frame = cv2.resize(rgb_frame, self.input_size)

                # Optional background segmentation on model input (RGB)
                if self.segment_background and self.bg_segmenter is not None:
                    try:
                        seg_results = self.bg_segmenter.process(model_frame)
                        mask = (seg_results.segmentation_mask > 0.5).astype(np.uint8)
                        # Black background where mask==0
                        model_frame = model_frame * mask[..., None]
                    except Exception as e:
                        if self.show_debug_info:
                            print(f"Background segmentation error: {e}")
                
                # Update frame buffer
                self.frame_buffer.append(model_frame)
                
                # Extract and update hand landmarks if using them
                hand_results = None
                if self.use_hand_landmarks:
                    # IMPORTANT: Extract landmarks on the same resized input used during training
                    hand_landmarks, hand_results = self.extract_hand_landmarks(model_frame, return_results=True)
                    self.hand_landmarks_buffer.append(hand_landmarks)
                
                # Make prediction
                best_prediction, best_confidence, top3_predictions = self.predict_sign()
                
                # Smooth prediction
                if best_prediction:
                    best_prediction, best_confidence = self.smooth_prediction(best_prediction, best_confidence)
                
                # Draw predictions
                self.draw_predictions(frame, best_prediction, best_confidence, top3_predictions)
                
                # Draw hand landmarks overlay if enabled
                if self.use_hand_landmarks:
                    self.draw_hand_landmarks_overlay(frame, hand_results)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # FPS display
                cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                          (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # System info
                mode_text = (
                    "CNN-Only Mode" if (not self.use_hand_landmarks or self.force_cnn_only)
                    else "CNN+Hand Mode"
                )
                cv2.putText(frame, mode_text, (frame.shape[1] - 150, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Background segmentation status
                seg_text = "BG Seg: ON" if self.segment_background else "BG Seg: OFF"
                cv2.putText(frame, seg_text, (frame.shape[1] - 150, 85), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                
                # Display frame
                cv2.imshow('SASL CNN-Only Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset buffers
                    self.frame_buffer.clear()
                    if self.use_hand_landmarks:
                        self.hand_landmarks_buffer.clear()
                    self.prediction_buffer.clear()
                    print("Buffers reset")
                elif key == ord('h') and self.use_hand_landmarks:
                    # Toggle hand landmarks overlay
                    self.show_hand_overlay = not self.show_hand_overlay
                    overlay_status = "ON" if self.show_hand_overlay else "OFF"
                    print(f"Hand landmarks overlay: {overlay_status}")
                elif key == ord('m') and self.use_hand_landmarks:
                    # Toggle using only CNN branch vs combined fusion
                    self.force_cnn_only = not self.force_cnn_only
                    mode = "CNN-Only predictions" if self.force_cnn_only else "Combined predictions"
                    print(f"Prediction mode: {mode}")
                elif key == ord('b'):
                    # Toggle background segmentation
                    self.segment_background = not self.segment_background
                    status = "ON" if self.segment_background else "OFF"
                    print(f"Background segmentation: {status}")
                elif key == ord('d'):
                    # Toggle debug info
                    self.show_debug_info = not self.show_debug_info
                    debug_status = "ON" if self.show_debug_info else "OFF"
                    print(f"Debug info: {debug_status}")
                elif key == ord('t'):
                    # Adjust confidence threshold
                    if self.confidence_threshold == 0.3:
                        self.confidence_threshold = 0.5
                    elif self.confidence_threshold == 0.5:
                        self.confidence_threshold = 0.7
                    else:
                        self.confidence_threshold = 0.3
                    print(f"Confidence threshold: {self.confidence_threshold}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\\nRecognition stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Clean up MediaPipe resources
            if self.use_hand_landmarks and self.hands:
                self.hands.close()
            
            recognition_type = "CNN+Hand Landmark" if self.use_hand_landmarks else "CNN-only"
            print(f"{recognition_type} camera recognition finished")

def main():
    """Main function for testing"""
    print("SASL Camera Recognition System")
    print("=" * 50)
    print("Searching for trained models...")
    
    # Check if we can import the models
    if CNNLSTMModel is None and CombinedCNNHandModel is None:
        print("❌ ERROR: Could not import models from training module")
        print("\nTo fix this issue:")
        print("1. Make sure video_cnn_only_training.py exists")
        print("2. Make sure you're running from the correct directory")
        print("3. Check that the training module is not corrupted")
        return
    
    # Determine which model to prioritize
    use_combined_model = CombinedCNNHandModel is not None and MEDIAPIPE_AVAILABLE
    model_type = "Combined CNN+Hand Landmark" if use_combined_model else "CNN-only"
    print(f"Will attempt to use: {model_type} model")
    
    # Check for the latest training outputs
    outputs_dir = Path("outputs")
    model_files = []
    
    if outputs_dir.exists():
        # Find the latest training directory
        training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
        if training_dirs:
            # Sort by name (timestamp) and get the latest
            latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            print(f"Found latest training session: {latest_training_dir.name}")
            
            # Check for models in the latest training directory
            models_dir = latest_training_dir / "models"
            results_dir = latest_training_dir / "results"
            
            if models_dir.exists() and results_dir.exists():
                cnn_model_path = models_dir / "best_sasl_cnn_lstm_model.pth"
                
                # Check for class names
                classes_path = results_dir / "class_names.json"
                if not classes_path.exists():
                    classes_path = results_dir / "pytorch_sasl_classes.json"
                
                if cnn_model_path.exists() and classes_path.exists():
                    model_files = [str(cnn_model_path), str(classes_path)]
                    print(f"Found CNN+LSTM model: {cnn_model_path}")
                    print(f"Found class names: {classes_path}")
                else:
                    print(f"Missing model files in {models_dir}")
            else:
                print(f"Models or results directory not found in {latest_training_dir}")
        else:
            print("No training directories found in outputs/")
    
    # Fallback: Check for models in root directory or outputs/
    if not model_files:
        print("Checking for models in fallback locations...")
        fallback_files = [
            "best_sasl_cnn_lstm_model.pth",
            "class_names.json"
        ]
        
        # Check outputs directory first, then root
        for model_file in fallback_files:
            if (outputs_dir / model_file).exists():
                model_files.append(str(outputs_dir / model_file))
            elif Path(model_file).exists():
                model_files.append(model_file)
            else:
                model_files.append(None)
        
        # Check if all files found
        if None in model_files:
            missing_files = [f for f, path in zip(fallback_files, model_files) if path is None]
            print(f"ERROR: Missing model files: {missing_files}")
            print("\\nTo fix this issue:")
            print("1. Run CNN-only training: python video_cnn_only_training.py")
            print("2. Or collect training data: python sasl_video_collector.py")
            print("3. Make sure training completes successfully")
            return
        else:
            print("Found models in fallback locations")
    
    if not model_files or len(model_files) != 2:
        print("ERROR: Could not locate all required model files")
        print("\\nTo fix this issue:")
        print("1. Run CNN-only training: python video_cnn_only_training.py") 
        print("2. Or collect training data: python sasl_video_collector.py")
        print("3. Make sure training completes successfully")
        return
    
    try:
        print(f"\\nStarting {model_type} SASL Camera Recognition System...")
        print("Initializing camera and loading model...")
        
        # Initialize and run recognition with appropriate model type
        recognition = SASLCNNOnlyCameraRecognition(
            model_path=model_files[0],
            classes_path=model_files[1],
            use_hand_landmarks=use_combined_model
        )
        
        print("Model loaded successfully!")
        print("Starting live recognition...")
        print("\\nControls:")
        print("   Q = Quit")
        print("   R = Reset prediction buffer")
        if use_combined_model:
            print("   H = Toggle hand landmarks overlay")
        print("\\nCamera window will open shortly...")
        
        recognition.run_live_recognition()
        
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        print("Please check that all model files exist and are accessible")
    except Exception as e:
        print(f"ERROR during recognition: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()