# Suppress MediaPipe verbose logging (must be before any imports)
import os
import sys
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Optional: Disable GPU to reduce warnings

# Suppress MediaPipe specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import time
from collections import deque
try:
    from hand_detection import HandDetector
    HAND_DETECTION_AVAILABLE = True
except ImportError:
    print("Hand detection not available. Using fallback mode.")
    HAND_DETECTION_AVAILABLE = False

# Load class names from the saved JSON file
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '03_DATA_CONFIG'))
sys.path.insert(0, os.path.join(parent_dir, '02_FALLBACK_COMPATIBILITY'))

config_path = os.path.join(parent_dir, '03_DATA_CONFIG', 'class_names.json')
output_dir = os.path.join(parent_dir, '05_OUTPUT_GENERATED')

with open(config_path, "r") as f:
    class_names = json.load(f)

# Load trained model - try hand-focused CNN+LSTM first
try:
    # Import from hand_focused_CNN_LSTM in the same directory
    import importlib.util
    import os
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hand_focused_path = os.path.join(current_dir, "hand_focused_CNN_LSTM.py")
    
    spec = importlib.util.spec_from_file_location("hand_focused_CNN_LSTM", hand_focused_path)
    hand_focused_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hand_focused_module)
    
    HandFocusedCNN_LSTM = hand_focused_module.HandFocusedCNN_LSTM
    cnn_base = hand_focused_module.cnn_base
    MODEL_TYPE = "hand_focused"
    print("Using Hand-Focused CNN-LSTM model")
except (ImportError, AttributeError, FileNotFoundError) as e:
    # Fallback to standard model in the fallback directory
    print(f"Hand-focused model not available ({e}), using fallback model")
    try:
        # Try to import from fallback directory
        fallback_dir = os.path.join(os.path.dirname(current_dir), "02_FALLBACK_COMPATIBILITY")
        model_path = os.path.join(fallback_dir, "model.py")
        
        spec = importlib.util.spec_from_file_location("model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        CNN_LSTM = model_module.CNN_LSTM
        cnn_base = model_module.cnn_base
        MODEL_TYPE = "standard"
        print("Using standard CNN-LSTM model")
    except Exception as fallback_error:
        print(f"Error loading fallback model: {fallback_error}")
        print("Please ensure the model files are available")
        sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model based on availability
if MODEL_TYPE == "hand_focused":
    model = HandFocusedCNN_LSTM(
        cnn=cnn_base, 
        num_classes=len(class_names),
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Try to load hand-focused model, fallback to standard model
    model_paths = [
        os.path.join(output_dir, "hand_focused_sasl_model.pth"),
        os.path.join(output_dir, "best_hand_focused_sasl_model.pth"),
        "hand_focused_sasl_model.pth",  # Local fallback
        "best_hand_focused_sasl_model.pth"  # Local fallback
    ]
    
    model_loaded = False
    for model_path in model_paths:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded hand-focused model weights from {model_path}")
            model_loaded = True
            break
        except FileNotFoundError:
            continue
    
    if not model_loaded:
        print("Hand-focused model not found, using standard model")
        # We need to import CNN_LSTM if we're falling back
        try:
            # Import from fallback directory
            fallback_dir = os.path.join(os.path.dirname(current_dir), "02_FALLBACK_COMPATIBILITY")
            model_path_fallback = os.path.join(fallback_dir, "model.py")
            
            spec = importlib.util.spec_from_file_location("model", model_path_fallback)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            
            CNN_LSTM = model_module.CNN_LSTM
            model = CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)
            MODEL_TYPE = "standard"
        except Exception as e:
            print(f"Error loading standard model class: {e}")
            print("!!! Could not load any model! Please check your installation.")
            sys.exit(1)
        
        # Try to load standard model
        standard_model_paths = [
            os.path.join(output_dir, "sasl_model.pth"),
            "sasl_model.pth"  # Local fallback
        ]
        
        for model_path in standard_model_paths:
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded standard model weights from {model_path}")
                MODEL_TYPE = "standard"
                break
            except FileNotFoundError:
                continue
        else:
            print("!!! No trained model found! Please train a model first.")
            print("Run: python main.py -> Option 1 to train a model")
            sys.exit(1)
else:
    # Standard model - need to import CNN_LSTM
    try:
        # Import from fallback directory
        fallback_dir = os.path.join(os.path.dirname(current_dir), "02_FALLBACK_COMPATIBILITY")
        model_path_fallback = os.path.join(fallback_dir, "model.py")
        
        spec = importlib.util.spec_from_file_location("model", model_path_fallback)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        CNN_LSTM = model_module.CNN_LSTM
        model = CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)
        model.load_state_dict(torch.load("sasl_model.pth", map_location=device))
    except Exception as e:
        print(f"Error loading standard model: {e}")
        print("!!! Could not load standard model! Please check your installation.")
        sys.exit(1)

model.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_frames(frames):
    """Apply transform and stack frames"""
    return torch.stack([transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames])

def extract_hand_region_from_frame(frame, hand_detector):
    """Extract hand region from frame if hands are detected"""
    if not HAND_DETECTION_AVAILABLE or hand_detector is None:
        return frame
    
    try:
        hands_data = hand_detector.detect_hands(frame)
        if hands_data:
            # Get the most confident hand
            best_hand = max(hands_data, key=lambda x: x['confidence'])
            bbox = best_hand['bbox']
            
            # Extract hand region
            hand_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if hand_region.size > 0:
                # Resize to match original frame size for consistency
                hand_region = cv2.resize(hand_region, (frame.shape[1], frame.shape[0]))
                return hand_region
        
        return frame
    except Exception as e:
        print(f"Hand detection error: {e}")
        return frame

class EnhancedGestureDetector:
    def __init__(self, buffer_size=16, stability_threshold=5, use_hand_detection=True):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.prediction_history = deque(maxlen=stability_threshold)
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_prediction_time = time.time()
        self.use_hand_detection = use_hand_detection and HAND_DETECTION_AVAILABLE
        
        # Initialize hand detector if available
        if self.use_hand_detection:
            try:
                self.hand_detector = HandDetector(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.3
                )
                print("Hand detection initialized")
            except Exception as e:
                print(f"Failed to initialize hand detector: {e}")
                self.hand_detector = None
                self.use_hand_detection = False
        else:
            self.hand_detector = None
    
    def add_frame(self, frame):
        # Process frame with hand detection if available
        if self.use_hand_detection and self.hand_detector:
            processed_frame = extract_hand_region_from_frame(frame, self.hand_detector)
        else:
            processed_frame = frame
        
        self.frame_buffer.append(processed_frame)
    
    def is_buffer_ready(self):
        return len(self.frame_buffer) == self.frame_buffer.maxlen
    
    def predict_gesture(self, model, device):
        if not self.is_buffer_ready():
            return None, 0.0, None
        
        try:
            # Convert buffer to tensor and predict
            frames_tensor = preprocess_frames(list(self.frame_buffer))
            frames_tensor = frames_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(frames_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Add to prediction history for stability
                self.prediction_history.append((predicted_class, confidence_score))
                
                # Get stable prediction
                stable_prediction = self._get_stable_prediction()
                
                # Get attention weights if available (for hand-focused model)
                attention_info = None
                if hasattr(model, 'attention') and MODEL_TYPE == "hand_focused":
                    # This is simplified - in practice you'd need to modify the model
                    # to return attention weights
                    attention_info = "Hand-focused attention active"
                
                return stable_prediction, confidence_score, attention_info
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def _get_stable_prediction(self):
        if len(self.prediction_history) < 3:
            return self.prediction_history[-1][0] if self.prediction_history else None
        
        # Count recent predictions
        recent_predictions = [pred[0] for pred in list(self.prediction_history)[-3:]]
        
        # Return most common prediction
        prediction_counts = {}
        for pred in recent_predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        most_common = max(prediction_counts.items(), key=lambda x: x[1])
        return most_common[0]
    
    def get_hand_overlay_info(self, frame):
        """Get hand detection overlay information"""
        if not self.use_hand_detection or not self.hand_detector:
            return []
        
        try:
            hands_data = self.hand_detector.detect_hands(frame)
            return hands_data
        except Exception as e:
            print(f"Hand overlay error: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        if self.hand_detector:
            self.hand_detector.close()

def run_enhanced_camera():
    """Run enhanced camera with hand detection"""
    print("Starting Enhanced SASL Recognition Camera")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Hand Detection: {'Enabled' if HAND_DETECTION_AVAILABLE else 'Disabled'}")
    print("Press 'q' to quit, 'h' to toggle hand detection, 'r' to reset")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = EnhancedGestureDetector(
        buffer_size=16, 
        stability_threshold=5,
        use_hand_detection=HAND_DETECTION_AVAILABLE
    )
    
    fps_counter = 0
    fps_start_time = time.time()
    show_hand_overlay = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        fps_counter += 1
        
        # Add frame to detector
        detector.add_frame(frame.copy())
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw hand detection overlay if enabled
        if show_hand_overlay:
            hands_info = detector.get_hand_overlay_info(frame)
            if hands_info and detector.hand_detector:
                display_frame = detector.hand_detector.draw_hands(
                    display_frame, hands_info, 
                    draw_landmarks=True, 
                    draw_bbox=True
                )
        
        # Make prediction if buffer is ready
        if detector.is_buffer_ready():
            prediction, confidence, attention_info = detector.predict_gesture(model, device)
            
            if prediction:
                # Display prediction
                text = f"Gesture: {prediction} ({confidence:.2f})"
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display model type
                model_text = f"Model: {MODEL_TYPE.replace('_', ' ').title()}"
                cv2.putText(display_frame, model_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display attention info if available
                if attention_info:
                    cv2.putText(display_frame, attention_info, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Show buffer filling status
            buffer_status = f"Collecting frames: {len(detector.frame_buffer)}/{detector.frame_buffer.maxlen}"
            cv2.putText(display_frame, buffer_status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Calculate and display FPS
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        else:
            fps = fps_counter / (time.time() - fps_start_time) if fps_counter > 0 else 0
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display hand detection status
        hand_status = f"Hand Detection: {'ON' if show_hand_overlay and detector.use_hand_detection else 'OFF'}"
        cv2.putText(display_frame, hand_status, (10, display_frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display controls
        cv2.putText(display_frame, "Controls: 'q'=quit, 'h'=toggle hands, 'r'=reset", 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow('Enhanced SASL Recognition', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_hand_overlay = not show_hand_overlay
            print(f"Hand overlay: {'ON' if show_hand_overlay else 'OFF'}")
        elif key == ord('r'):
            detector.frame_buffer.clear()
            detector.prediction_history.clear()
            print("Detector reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    print("Camera session ended")

if __name__ == "__main__":
    run_enhanced_camera()
