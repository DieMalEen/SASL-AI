import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
import os
import glob
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandGestureRecognizer:
    """Real-time hand gesture recognition with landmark overlay."""
    
    def __init__(self, model_dir):
        """
        Initialize the recognizer with a trained model.
        
        Args:
            model_dir: Directory containing the trained model files
        """
        print(f"Loading model from: {model_dir}")
        
        # Load model components
        self.model = joblib.load(f"{model_dir}/random_forest_model.joblib")
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.joblib")
        
        print(f"Model loaded successfully!")
        print(f"Recognizing {len(self.label_encoder.classes_)} gestures: {', '.join(self.label_encoder.classes_)}")
        
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # For smoothing predictions
        self.prediction_history = deque(maxlen=5)
        
        # Check model input size to determine if it supports two-hand format
        # Two-hand format: 126 features (63 left + 63 right)
        # Single-hand format: 63 features
        expected_features = self.scaler.n_features_in_
        self.two_hand_mode = (expected_features == 126)
        
        if self.two_hand_mode:
            print("✓ Model supports TWO-HAND gestures (left + right)")
        else:
            print("✓ Model uses SINGLE-HAND format (legacy)")
        
    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmark coordinates from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            numpy array of 63 features (21 landmarks x 3 coordinates)
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def predict_gesture(self, hands_data):
        """
        Predict gesture from hand landmarks.
        
        Args:
            hands_data: Dictionary with 'left' and/or 'right' keys containing landmark arrays
                       Format: {'left': array[63], 'right': array[63]}
            
        Returns:
            predicted_class: Predicted gesture class
            confidence: Prediction confidence
        """
        # Prepare feature vector based on model format
        if self.two_hand_mode:
            # Two-hand format: 126 features (left 63 + right 63)
            left_landmarks = hands_data.get('left', np.zeros(63))
            right_landmarks = hands_data.get('right', np.zeros(63))
            features = np.concatenate([left_landmarks, right_landmarks]).reshape(1, -1)
        else:
            # Single-hand format: use first available hand (63 features)
            if 'left' in hands_data:
                features = hands_data['left'].reshape(1, -1)
            elif 'right' in hands_data:
                features = hands_data['right'].reshape(1, -1)
            else:
                return None, 0.0
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        predicted_class = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        # Add to history for smoothing
        self.prediction_history.append(predicted_class)
        
        # Use most common prediction in recent history
        if len(self.prediction_history) >= 3:
            from collections import Counter
            smoothed_prediction = Counter(self.prediction_history).most_common(1)[0][0]
            return smoothed_prediction, confidence
        
        return predicted_class, confidence
    
    def draw_prediction_info(self, frame, predicted_class, confidence, hand_label, position):
        """
        Draw prediction information on the frame.
        
        Args:
            frame: OpenCV frame
            predicted_class: Predicted gesture class
            confidence: Prediction confidence
            hand_label: "Left" or "Right"
            position: (x, y) tuple for text position
        """
        x, y = position
        
        # Background rectangle for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 40), (x + 300, y + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Prediction text
        text = f"{hand_label} Hand: {predicted_class.upper()}"
        cv2.putText(frame, text, (x, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Confidence bar
        bar_width = int(confidence * 250)
        cv2.rectangle(frame, (x, y), (x + 250, y + 15), (100, 100, 100), -1)
        
        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        cv2.rectangle(frame, (x, y), (x + bar_width, y + 15), color, -1)
        cv2.putText(frame, f"{confidence*100:.1f}%", (x + 260, y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_instructions(self, frame):
        """Draw usage instructions on the frame."""
        instructions = [
            "SASL Hand Gesture Recognition",
            "Press 'Q' to quit",
            "Press 'S' to toggle landmarks",
            "Press 'F' to toggle FPS"
        ]
        
        y_offset = 30
        for i, instruction in enumerate(instructions):
            y = y_offset + i * 30
            # Background
            overlay = frame.copy()
            (text_width, text_height), _ = cv2.getTextSize(
                instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(overlay, (10, y - 25), (text_width + 20, y + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Text color
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, instruction, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    
    def run_camera(self, camera_id=0, show_landmarks=True, show_fps=True):
        """
        Run real-time hand gesture recognition on camera feed.
        
        Args:
            camera_id: Camera device ID (default: 0)
            show_landmarks: Whether to show hand landmarks overlay
            show_fps: Whether to show FPS counter
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("Starting real-time hand gesture recognition...")
        print("="*60)
        print("Press 'Q' to quit")
        print("Press 'S' to toggle landmark display")
        print("Press 'F' to toggle FPS display")
        print("="*60 + "\n")
        
        # FPS calculation
        import time
        prev_time = time.time()
        fps = 0
        
        # Toggle flags
        landmarks_visible = show_landmarks
        fps_visible = show_fps
        
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Draw instructions
            self.draw_instructions(frame)
            
            # Collect hand data
            hands_data = {}
            
            # Process detected hands
            if results.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # Get hand label (Left or Right)
                    hand_label = handedness.classification[0].label.lower()
                    
                    # Extract landmarks
                    hands_data[hand_label] = self.extract_landmarks(hand_landmarks)
                    
                    # Draw hand landmarks if enabled
                    if landmarks_visible:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                
                # Make prediction with all detected hands
                if hands_data:
                    predicted_class, confidence = self.predict_gesture(hands_data)
                    
                    if predicted_class is not None:
                        # Display which hands are being used
                        hand_labels = " + ".join([h.upper() for h in hands_data.keys()])
                        
                        # Get position for text (center-top of frame)
                        h, w, _ = frame.shape
                        x = w // 2 - 100
                        y = 100
                        
                        # Draw prediction info
                        self.draw_prediction_info(frame, predicted_class, confidence, 
                                                hand_labels, (x, y))
            else:
                # No hands detected
                overlay = frame.copy()
                cv2.rectangle(overlay, (frame.shape[1]//2 - 150, frame.shape[0]//2 - 30),
                            (frame.shape[1]//2 + 150, frame.shape[0]//2 + 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "No hands detected", 
                           (frame.shape[1]//2 - 140, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show FPS if enabled
            if fps_visible:
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('SASL Hand Gesture Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            elif key == ord('s') or key == ord('S'):
                landmarks_visible = not landmarks_visible
                status = "ON" if landmarks_visible else "OFF"
                print(f"Landmarks display: {status}")
            elif key == ord('f') or key == ord('F'):
                fps_visible = not fps_visible
                status = "ON" if fps_visible else "OFF"
                print(f"FPS display: {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nCamera released. Goodbye!")

def find_latest_model(base_dir="outputs"):
    """
    Find the most recent hand landmark model directory.
    
    Args:
        base_dir: Base directory to search for models
        
    Returns:
        Path to the latest model directory, or None if not found
    """
    pattern = os.path.join(base_dir, "hand_landmark_model_*")
    model_dirs = glob.glob(pattern)
    
    if not model_dirs:
        return None
    
    # Sort by directory name (which includes timestamp) and get the latest
    model_dirs.sort(reverse=True)
    return model_dirs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Real-time SASL hand gesture recognition with landmark overlay"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to the model directory (default: auto-detect latest)"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--no-landmarks", 
        action="store_true",
        help="Don't show hand landmarks overlay"
    )
    parser.add_argument(
        "--no-fps", 
        action="store_true",
        help="Don't show FPS counter"
    )
    
    args = parser.parse_args()
    
    # Auto-detect model if not specified
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("Error: No trained model found in outputs/")
            print("\nPlease train a model first by running:")
            print("  python train_hand_landmark_model.py")
            return
        print(f"Auto-detected model: {model_path}")
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found: {model_path}")
        print("\nPlease train a model first by running:")
        print("  python train_hand_landmark_model.py")
        return
    
    # Initialize recognizer
    recognizer = HandGestureRecognizer(model_path)
    
    # Run camera
    recognizer.run_camera(
        camera_id=args.camera,
        show_landmarks=not args.no_landmarks,
        show_fps=not args.no_fps
    )

if __name__ == "__main__":
    main()
