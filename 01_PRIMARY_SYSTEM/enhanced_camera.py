# Suppress MediaPipe verbose logging (must be before any imports)
import os
import sys
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Optional: Disable GPU to reduce warnings

# Suppress MediaPipe specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

import json
import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from collections import deque
import importlib.util
from PIL import Image  # Add this import for PIL Image

try:
    from hand_detection import HandDetector
    HAND_DETECTION_AVAILABLE = True
except ImportError:
    print("Hand detection not available. Using fallback mode.")
    HAND_DETECTION_AVAILABLE = False

# Set up device and paths first (before loading class names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")

# Load class names from JSON file
def load_class_names():
    """Load class names from the JSON configuration file"""
    class_names_path = os.path.join(os.path.dirname(current_dir), "03_DATA_CONFIG", "class_names.json")
    
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} classes from class_names.json")
        return class_names
    except FileNotFoundError:
        print(f"ERROR: class_names.json not found at {class_names_path}")
        print("Please ensure the class_names.json file exists in the 03_DATA_CONFIG directory.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse class_names.json: {e}")
        print("Please check that class_names.json contains valid JSON format.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error loading class_names.json: {e}")
        sys.exit(1)

# Replace the hardcoded class_names list with dynamic loading
class_names = load_class_names()
print(f"Total classes configured: {len(class_names)}")

# Import CNN base for all models
try:
    # Try to get cnn_base from hand_focused_CNN_LSTM
    import importlib.util
    hand_focused_path = os.path.join(current_dir, "hand_focused_CNN_LSTM.py")
    spec = importlib.util.spec_from_file_location("hand_focused_CNN_LSTM", hand_focused_path)
    hand_focused_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hand_focused_module)
    cnn_base = hand_focused_module.cnn_base
    print("CNN base loaded from hand_focused_CNN_LSTM")
except Exception as e:
    # Fallback to standard model CNN base
    try:
        fallback_dir = os.path.join(os.path.dirname(current_dir), "02_FALLBACK_COMPATIBILITY")
        model_path = os.path.join(fallback_dir, "model.py")
        spec = importlib.util.spec_from_file_location("model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        cnn_base = model_module.cnn_base
        print("CNN base loaded from fallback model")
    except Exception as fallback_error:
        print(f"Error loading CNN base: {fallback_error}")
        sys.exit(1)

# Load trained model - try forced selection first, then auto-detection
forced_model_path = os.environ.get('SASL_FORCE_MODEL_PATH')
model = None
MODEL_TYPE = None

def validate_model(model, device, class_names):
    """Test the model with dummy data to check for NaN outputs"""
    try:
        # Create dummy input tensor
        dummy_input = torch.randn(1, 16, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            
            # Check for NaN or infinite values
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                return False
            
            # Check if softmax works
            probabilities = torch.softmax(outputs, dim=1)
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                return False
            
            return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

def load_model_from_checkpoint(model_path, checkpoint_data):
    """Load model based on checkpoint structure"""
    global model, MODEL_TYPE
    
    # Detect model architecture type based on checkpoint keys
    if 'lstm.weight_ih_l0_reverse' not in checkpoint_data and 'attention.in_proj_weight' not in checkpoint_data:
        # This is a FastCNNLSTM (GPU-optimized) model
        print(f"Detected FastCNNLSTM model")
        
        # Import and create FastCNNLSTM model
        import importlib.util
        gpu_training_path = os.path.join(current_dir, "gpu_optimized_training.py")
        spec = importlib.util.spec_from_file_location("gpu_optimized_training", gpu_training_path)
        gpu_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gpu_module)
        
        FastCNNLSTM = gpu_module.FastCNNLSTM
        model = FastCNNLSTM(num_classes=len(class_names)).to(device)
        model.load_state_dict(checkpoint_data)
        MODEL_TYPE = "gpu_optimized"
        
        # Validate the model for NaN outputs
        if not validate_model(model, device, class_names):
            print(f"⚠️  Model validation failed - model produces NaN outputs")
            model = None
            MODEL_TYPE = None
            return False
        
        print(f"Successfully loaded and validated FastCNNLSTM model")
        return True
        
    elif 'lstm.weight_ih_l0_reverse' in checkpoint_data and 'attention.in_proj_weight' in checkpoint_data:
        # This is a HandFocusedCNN_LSTM or HybridOptimizedCNN_LSTM model
        if "hybrid" in model_path.lower():
            print(f"Detected Hybrid CPU+GPU model")
            
            # Import hybrid model
            import importlib.util
            hybrid_training_path = os.path.join(current_dir, "hybrid_cpu_gpu_training.py")
            spec = importlib.util.spec_from_file_location("hybrid_cpu_gpu_training", hybrid_training_path)
            hybrid_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hybrid_module)
            
            HybridOptimizedCNN_LSTM = hybrid_module.HybridOptimizedCNN_LSTM
            model = HybridOptimizedCNN_LSTM(
                cnn=cnn_base, 
                num_classes=len(class_names),
                hidden_size=256,
                num_layers=2,
                dropout=0.3
            ).to(device)
            MODEL_TYPE = "hybrid_cpu_gpu"
        else:
            print(f"Detected Hand-Focused model")
            
            # Import hand-focused model
            import importlib.util
            hand_focused_path = os.path.join(current_dir, "hand_focused_CNN_LSTM.py")
            spec = importlib.util.spec_from_file_location("hand_focused_CNN_LSTM", hand_focused_path)
            hand_focused_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hand_focused_module)
            
            HandFocusedCNN_LSTM = hand_focused_module.HandFocusedCNN_LSTM
            model = HandFocusedCNN_LSTM(
                cnn=cnn_base, 
                num_classes=len(class_names),
                hidden_size=256,
                num_layers=2,
                dropout=0.3
            ).to(device)
            MODEL_TYPE = "hand_focused"
        
        model.load_state_dict(checkpoint_data)
        
        # Validate the model for NaN outputs
        if not validate_model(model, device, class_names):
            print(f"! Model validation failed - model produces NaN outputs")
            model = None
            MODEL_TYPE = None
            return False
        
        print(f"Successfully loaded and validated {MODEL_TYPE} model")
        return True
    
    return False

# Try forced model selection first
if forced_model_path and os.path.exists(forced_model_path):
    print(f"Using forced model selection: {forced_model_path}")
    try:
        checkpoint = torch.load(forced_model_path, map_location=device, weights_only=False)
        if load_model_from_checkpoint(forced_model_path, checkpoint):
            print(f"Forced model loaded successfully")
        else:
            print("Unknown model architecture or class mismatch, falling back to auto-detection")
            forced_model_path = None
    except Exception as e:
        print(f"Error loading forced model: {e}")
        print("Falling back to auto-detection")
        forced_model_path = None

# Auto-detection if no forced model or forced model failed
if model is None:
    print("Starting auto-detection...")
    
    # Try to load models with smart architecture detection - check all model types
    model_paths = [
        # GPU-optimized models (actual saved filenames)
        os.path.join(output_dir, "gpu_sasl_model.pth"),
        os.path.join(output_dir, "gpu_optimized_sasl_model.pth"),
        # Hybrid models (correct filename from hybrid training script)
        os.path.join(output_dir, "hybrid_sasl_model.pth"),
        os.path.join(output_dir, "hybrid_cpu_gpu_sasl_model.pth"),
        # Hand-focused models (actual saved filenames)
        os.path.join(output_dir, "hand_focused_sasl_model.pth"),
        # Local fallbacks
        "gpu_sasl_model.pth",
        "gpu_optimized_sasl_model.pth",
        "hybrid_sasl_model.pth",
        "hybrid_cpu_gpu_sasl_model.pth", 
        "hand_focused_sasl_model.pth"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if load_model_from_checkpoint(model_path, checkpoint):
                    print(f"Auto-detected and loaded model from {model_path}")
                    model_loaded = True
                    break
            except Exception as e:
                if "size mismatch" in str(e) or "Missing key" in str(e):
                    print(f"Architecture mismatch in {model_path} (wrong number of classes), trying next...")
                    continue
                else:
                    print(f"Error with {model_path}: {e}")
                    continue
    
    # Final fallback to standard model if nothing worked
    if not model_loaded:
        print("No compatible trained models found, using standard CNN-LSTM")
        try:
            # Import standard model
            fallback_dir = os.path.join(os.path.dirname(current_dir), "02_FALLBACK_COMPATIBILITY")
            model_path_fallback = os.path.join(fallback_dir, "model.py")
            
            spec = importlib.util.spec_from_file_location("model", model_path_fallback)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            
            CNN_LSTM = model_module.CNN_LSTM
            model = CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)
            MODEL_TYPE = "standard"
            print("Using standard CNN-LSTM model (no trained weights)")
        except Exception as e:
            print(f"Error loading standard model: {e}")
            print("!!! Could not load any model! Please check your installation.")
            sys.exit(1)

# Evaluation mode
model.eval()

# Ensure MODEL_TYPE is not None
if MODEL_TYPE is None:
    MODEL_TYPE = "unknown"
    print("Warning: MODEL_TYPE was None, set to 'unknown'")

print(f"Model ready: {MODEL_TYPE.upper()} architecture")

# Clean up environment variable
if 'SASL_FORCE_MODEL_PATH' in os.environ:
    del os.environ['SASL_FORCE_MODEL_PATH']

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
                
                # Check for NaN or infinite values in model output
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("Error: Model is producing invalid outputs. Model may be corrupted.")
                    print("Switching to fallback standard model...")
                    return "MODEL_ERROR", 0.0, "Model Error - Using Fallback"
                
                probabilities = torch.softmax(outputs, dim=1)
                
                # Check for NaN in probabilities after softmax
                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    print("Error: Softmax produced invalid probabilities")
                    return "MODEL_ERROR", 0.0, "Softmax Error"
                
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Ensure valid indices and confidence
                if predicted_idx.item() >= len(class_names) or torch.isnan(confidence).any():
                    print(f"Warning: Invalid prediction - idx: {predicted_idx.item()}, confidence: {confidence.item()}")
                    return None, 0.0, None
                
                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Filter out very low confidence predictions (very low threshold for better visibility)
                if confidence_score < 0.01:
                    return None, 0.0, None
                
                # Add to prediction history for stability
                self.prediction_history.append((predicted_class, confidence_score))
                
                # Get stable prediction
                stable_prediction = self._get_stable_prediction()
                
                # Get model-specific information
                attention_info = None
                if hasattr(model, 'attention') and MODEL_TYPE in ["hand_focused", "hybrid_cpu_gpu"]:
                    if MODEL_TYPE == "hand_focused":
                        attention_info = "Hand-focused attention active"
                    elif MODEL_TYPE == "hybrid_cpu_gpu":
                        attention_info = "Hybrid CPU+GPU with attention"
                elif MODEL_TYPE in ["gpu_optimized", "fast_cnn_lstm"]:
                    attention_info = "GPU-optimized (streamlined)"
                elif MODEL_TYPE == "standard":
                    attention_info = "Standard CNN-LSTM"
                
                return stable_prediction, confidence_score, attention_info
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def _get_stable_prediction(self):
        if len(self.prediction_history) < 1:  # Show predictions immediately
            return self.prediction_history[-1][0] if self.prediction_history else None
        
        # Count recent predictions (immediate response)
        recent_predictions = [pred[0] for pred in list(self.prediction_history)[-1:]]
        
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
        buffer_size=8,  # Reduced for faster initial predictions
        stability_threshold=2,  # Reduced for faster predictions
        use_hand_detection=HAND_DETECTION_AVAILABLE
    )
    
    show_hand_overlay = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use natural camera orientation (no mirroring)
        # frame = cv2.flip(frame, 1)  # Commented out for natural view
        
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
            
            if prediction and prediction != "MODEL_ERROR":
                # Display prediction with larger, more visible text
                text = f"Gesture: {prediction} ({confidence:.2f})"
                cv2.putText(display_frame, text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Display model type
                model_text = f"Model: {MODEL_TYPE.replace('_', ' ').title()}"
                cv2.putText(display_frame, model_text, (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display attention info if available
                if attention_info:
                    cv2.putText(display_frame, attention_info, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            elif prediction == "MODEL_ERROR":
                # Show model error message
                cv2.putText(display_frame, "Model Error - Check Console", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Show buffer filling status
            buffer_status = f"Collecting frames: {len(detector.frame_buffer)}/{detector.frame_buffer.maxlen}"
            cv2.putText(display_frame, buffer_status, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display hand detection status
        hand_status = f"Hand Detection: {'ON' if show_hand_overlay and detector.use_hand_detection else 'OFF'}"
        cv2.putText(display_frame, hand_status, (10, display_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display controls
        cv2.putText(display_frame, "Controls: 'q'=quit, 'h'=toggle hands, 'r'=reset", 
                   (10, display_frame.shape[0] - 20), 
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
