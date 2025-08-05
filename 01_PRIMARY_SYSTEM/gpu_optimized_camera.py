# GPU-Optimized Real-time SASL Recognition Camera
# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
import json
import time
from collections import deque
from PIL import Image
import threading
import queue
from hand_detection import HandDetector

class GPUOptimizedHandFocusedCNN_LSTM(nn.Module):
    """GPU-optimized CNN-LSTM model for real-time inference"""
    def __init__(self, cnn, hidden_size=256, num_classes=20, num_layers=2, dropout=0.3):
        super(GPUOptimizedHandFocusedCNN_LSTM, self).__init__()
        self.cnn = cnn
        
        # Enhanced LSTM optimized for GPU
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Optimized attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        # Extract CNN features with memory optimization
        x = x.view(batch_size * seq_len, C, H, W)
        with torch.cuda.amp.autocast():
            features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention mechanism
        attended_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use global average pooling
        pooled_features = torch.mean(attended_features, dim=1)
        
        # Final classification
        out = self.classifier(pooled_features)
        return out

class GPUOptimizedSASLCamera:
    """GPU-optimized real-time SASL recognition system"""
    
    def __init__(self, model_path, class_names_path, device=None, confidence_threshold=0.7):
        """Initialize GPU-optimized camera system"""
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"🚀 Initializing GPU-Optimized SASL Camera")
        print(f"   Device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        # Load and setup model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # GPU-optimized preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Hand detection with optimized settings
        self.hand_detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # Higher for better accuracy
            min_tracking_confidence=0.5
        )
        
        # Frame buffer for temporal processing
        self.frame_buffer = deque(maxlen=16)
        self.confidence_threshold = confidence_threshold
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.inference_times = deque(maxlen=30)
        
        # Threading for frame processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.processing_thread = None
        self.stop_processing = False
        
    def _load_model(self, model_path):
        """Load the trained model optimized for GPU inference"""
        # Create CNN backbone
        cnn_base = models.resnet18(pretrained=True)
        cnn_base = nn.Sequential(*list(cnn_base.children())[:-1])
        
        # Create model
        model = GPUOptimizedHandFocusedCNN_LSTM(
            cnn=cnn_base,
            num_classes=self.num_classes,
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        )
        
        # Load weights
        try:
            if self.device.type == 'cuda':
                state_dict = torch.load(model_path, map_location=self.device)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            
            # Optimize for inference
            if self.device.type == 'cuda':
                model = torch.jit.script(model)  # TorchScript optimization
            
            print(f"✅ Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to randomly initialized model")
            return model.to(self.device)
    
    def _extract_hand_region(self, frame):
        """Extract hand regions from frame with GPU optimization"""
        try:
            # Detect hands
            results = self.hand_detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks and results.multi_handedness:
                # Get the most confident hand
                best_hand_idx = 0
                best_confidence = 0
                
                for idx, handedness in enumerate(results.multi_handedness):
                    confidence = handedness.classification[0].score
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_hand_idx = idx
                
                if best_confidence > 0.7:  # High confidence threshold
                    landmarks = results.multi_hand_landmarks[best_hand_idx]
                    
                    # Calculate bounding box
                    h, w, _ = frame.shape
                    x_coords = [landmark.x * w for landmark in landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding
                    padding = 50
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Extract hand region
                    hand_region = frame[y_min:y_max, x_min:x_max]
                    
                    if hand_region.size > 0:
                        return hand_region, (x_min, y_min, x_max, y_max), best_confidence
            
            # Fallback to center crop if no hands detected
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            crop_size = min(h, w) // 2
            
            x_min = max(0, center_x - crop_size)
            y_min = max(0, center_y - crop_size)
            x_max = min(w, center_x + crop_size)
            y_max = min(h, center_y + crop_size)
            
            return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max), 0.0
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return frame, (0, 0, frame.shape[1], frame.shape[0]), 0.0
    
    def _preprocess_frame(self, frame):
        """GPU-optimized frame preprocessing"""
        # Extract hand region
        hand_region, bbox, confidence = self._extract_hand_region(frame)
        
        # Convert to PIL and apply transforms
        frame_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        tensor_frame = self.transform(pil_frame)
        
        return tensor_frame, bbox, confidence
    
    def _process_frame_sequence(self):
        """Background thread for processing frame sequences"""
        while not self.stop_processing:
            try:
                # Get frame from queue
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Process frame
                    tensor_frame, bbox, hand_confidence = self._preprocess_frame(frame)
                    self.frame_buffer.append(tensor_frame)
                    
                    # Only predict if we have enough frames
                    if len(self.frame_buffer) >= 8:
                        prediction, confidence = self._predict_sequence()
                        
                        # Put result in queue
                        if not self.result_queue.full():
                            self.result_queue.put({
                                'prediction': prediction,
                                'confidence': confidence,
                                'bbox': bbox,
                                'hand_confidence': hand_confidence
                            })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _predict_sequence(self):
        """GPU-optimized sequence prediction"""
        if len(self.frame_buffer) < 8:
            return "Collecting frames...", 0.0
        
        try:
            start_time = time.time()
            
            # Prepare sequence tensor
            sequence = torch.stack(list(self.frame_buffer)[-16:])  # Use last 16 frames
            sequence = sequence.unsqueeze(0).to(self.device, non_blocking=True)
            
            # Prediction with GPU optimization
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(sequence)
                else:
                    outputs = self.model(sequence)
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_label = self.class_names[predicted_class.item()]
                confidence_score = confidence.item()
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return predicted_label, confidence_score
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0
    
    def _draw_results(self, frame, prediction, confidence, bbox, hand_confidence):
        """Draw results on frame with GPU performance metrics"""
        h, w, _ = frame.shape
        
        # Draw hand bounding box
        x_min, y_min, x_max, y_max = bbox
        if hand_confidence > 0.5:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Hand: {hand_confidence:.2f}", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(frame, "Center crop", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Determine text color based on confidence
        if confidence >= self.confidence_threshold:
            color = (0, 255, 0)  # Green for high confidence
            status = "✓"
        elif confidence >= 0.4:
            color = (0, 165, 255)  # Orange for medium confidence
            status = "?"
        else:
            color = (0, 0, 255)  # Red for low confidence
            status = "✗"
        
        # Main prediction text
        main_text = f"{status} {prediction}: {confidence:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (20 + text_width, 50 + text_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (20 + text_width, 50 + text_height), color, 2)
        
        # Draw main text
        cv2.putText(frame, main_text, (15, 35 + text_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # GPU Performance metrics
        if self.device.type == 'cuda':
            # GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            
            gpu_text = f"GPU: {allocated:.0f}MB / {reserved:.0f}MB"
            cv2.putText(frame, gpu_text, (15, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Inference time
        if self.inference_times:
            avg_inference_time = np.mean(self.inference_times) * 1000  # ms
            fps_text = f"Inference: {avg_inference_time:.1f}ms"
            cv2.putText(frame, fps_text, (15, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Overall FPS
        current_time = time.time()
        self.fps_counter += 1
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.current_fps = fps
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        if hasattr(self, 'current_fps'):
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(frame, fps_text, (15, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Device indicator
        device_text = f"Device: {self.device.type.upper()}"
        device_color = (0, 255, 255) if self.device.type == 'cuda' else (128, 128, 128)
        cv2.putText(frame, device_text, (w - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, device_color, 1)
    
    def run(self, camera_index=0):
        """Run GPU-optimized real-time recognition"""
        print(f"🎥 Starting GPU-Optimized SASL Camera (Camera {camera_index})")
        print("Controls:")
        print("  'q' or 'ESC' - Quit")
        print("  'r' - Reset frame buffer") 
        print("  's' - Save current frame")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Start processing thread
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._process_frame_sequence)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Default values
        current_prediction = "Initializing..."
        current_confidence = 0.0
        current_bbox = (0, 0, 100, 100)
        current_hand_confidence = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Add frame to processing queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Get latest results (non-blocking)
                try:
                    while not self.result_queue.empty():
                        result = self.result_queue.get_nowait()
                        current_prediction = result['prediction']
                        current_confidence = result['confidence']
                        current_bbox = result['bbox']
                        current_hand_confidence = result['hand_confidence']
                except queue.Empty:
                    pass
                
                # Draw results on display frame
                display_frame = frame.copy()
                self._draw_results(display_frame, current_prediction, current_confidence, 
                                 current_bbox, current_hand_confidence)
                
                # Display frame
                cv2.imshow('GPU-Optimized SASL Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Reset buffer
                    self.frame_buffer.clear()
                    print("🔄 Frame buffer reset")
                elif key == ord('s'):  # Save frame
                    timestamp = int(time.time())
                    filename = f"sasl_capture_{timestamp}.jpg"
                    
                    # Get absolute path to output directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(script_dir)
                    output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    save_path = os.path.join(output_dir, filename)
                    cv2.imwrite(save_path, display_frame)
                    print(f"📸 Frame saved: {save_path}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Cleanup
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print("✅ GPU-Optimized camera system stopped")

def main():
    """Main function to run GPU-optimized camera"""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Try to find the trained model
    model_paths = [
        os.path.join(project_root, "05_OUTPUT_GENERATED", "gpu_optimized_hand_focused_sasl_model.pth"),
        os.path.join(project_root, "05_OUTPUT_GENERATED", "best_gpu_optimized_hand_focused_sasl_model.pth"),
        os.path.join(project_root, "05_OUTPUT_GENERATED", "hand_focused_sasl_model.pth"),
        os.path.join(project_root, "05_OUTPUT_GENERATED", "best_hand_focused_sasl_model.pth")
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found. Please train a model first.")
        print("Looking for models in:")
        for path in model_paths:
            print(f"  - {path}")
        return
    
    class_names_path = os.path.join(project_root, "03_DATA_CONFIG", "class_names.json")
    
    if not os.path.exists(class_names_path):
        print(f"❌ Class names file not found: {class_names_path}")
        return
    
    print(f"📁 Using model: {model_path}")
    print(f"📁 Using class names: {class_names_path}")
    
    # Initialize and run GPU-optimized camera
    try:
        camera = GPUOptimizedSASLCamera(
            model_path=model_path,
            class_names_path=class_names_path,
            confidence_threshold=0.6
        )
        camera.run(camera_index=0)
        
    except Exception as e:
        print(f"❌ Error running camera: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
