import threading
import queue
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import time
from collections import deque

# Load class names from the saved JSON file
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load trained model from the separate model file
from model import CNN_LSTM, cnn_base
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("sasl_model.pth", map_location=device))
model.eval()

# Transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_frames(frames):
    # Apply transform and stack frames
    return torch.stack([transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames])

class ThreadedCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        
    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True  # Dies when main thread dies
        self.thread.start()
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    
    def read(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
        
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class GestureDetector:
    def __init__(self, buffer_size=8, stability_threshold=3):  # Reduced for better performance
        self.frame_buffer = deque(maxlen=buffer_size)
        self.prediction_history = deque(maxlen=stability_threshold)
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_prediction_time = time.time()
        
    def add_frame(self, frame):
        self.frame_buffer.append(frame)
        
    def is_buffer_ready(self):
        return len(self.frame_buffer) == self.frame_buffer.maxlen
        
    def predict_gesture(self, model, device):
        if not self.is_buffer_ready():
            return None
            
        # Convert buffer to tensor and predict
        frames = list(self.frame_buffer)
        input_tensor = preprocess_frames(frames).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_names[predicted.item()]
            
        # Add to prediction history
        self.prediction_history.append(predicted_label)
        
        # Check if gesture is stable
        if len(self.prediction_history) == self.prediction_history.maxlen:
            unique_predictions = set(self.prediction_history)
            
            # If most predictions are the same, consider it a stable gesture
            if len(unique_predictions) == 1:
                stable_gesture = list(unique_predictions)[0]
                
                # New gesture detected
                if stable_gesture != self.current_gesture:
                    if self.current_gesture is not None:
                        print(f"Gesture ended: {self.current_gesture}")
                    
                    self.current_gesture = stable_gesture
                    self.gesture_start_time = time.time()
                    print(f"New gesture detected: {stable_gesture}")
                    
                return stable_gesture
                
        return self.current_gesture

# Initialize threaded camera and gesture detector
threaded_cam = ThreadedCamera()
threaded_cam.start()
detector = GestureDetector()

print("Starting SASL gesture recognition with threading...")
print("Show a gesture to the camera. Press 'q' to quit.")

# Performance tracking
frame_count = 0
display_frame_count = 0
start_time = time.time()
fps_update_time = start_time
frame_skip = 0

try:
    while True:
        frame = threaded_cam.read()
        if frame is not None:
            frame_count += 1
            
            # Only process every 3rd frame for gesture detection (performance boost)
            frame_skip += 1
            if frame_skip % 3 == 0:
                detector.add_frame(frame)
            
            # Predict every 200ms instead of 100ms for better performance
            current_time = time.time()
            if current_time - detector.last_prediction_time > 0.2:
                predicted_gesture = detector.predict_gesture(model, device)
                detector.last_prediction_time = current_time
            else:
                predicted_gesture = detector.current_gesture
            
            # Display current gesture
            if predicted_gesture:
                cv2.putText(frame, f'Gesture: {predicted_gesture}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                if detector.gesture_start_time:
                    duration = current_time - detector.gesture_start_time
                    cv2.putText(frame, f'Duration: {duration:.1f}s', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            # Calculate and display FPS every second
            if current_time - fps_update_time >= 1.0:
                fps = display_frame_count / (current_time - fps_update_time)
                fps_update_time = current_time
                display_frame_count = 0
                print(f"Display FPS: {fps:.1f}")
            
            cv2.putText(frame, 'Press Q to quit', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, 'Threaded Mode', (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            cv2.imshow('SASL Real-Time Detection (Threaded)', frame)
            display_frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Clean shutdown
    threaded_cam.stop()
    cv2.destroyAllWindows()
    print("SASL detection stopped.")