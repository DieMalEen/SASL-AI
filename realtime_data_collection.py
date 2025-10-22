import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class RealtimeDataCollector:
    """Real-time hand landmark data collection with keyboard input."""
    
    def __init__(self, csv_path="outputs/hand_landmarks.csv"):
        """
        Initialize the data collector.
        
        Args:
            csv_path: Path to the CSV file for saving data
        """
        self.csv_path = csv_path
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support both hands for two-hand gestures
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Valid classes with combined gestures
        # Remove standalone '0', '2', '7', 'l', 'o', 'v' and add combined classes
        base_classes = list('13456789abcdefghijkmnpqrstuwxyz')  # Removed 0, 2, 7, l, o, v
        base_classes.append('10')  # Add "10" as a valid class (triggered by '-')
        base_classes.append('l(7)')  # Combined L and 7 (triggered by 'l' or '7')
        base_classes.append('o(0)')  # Combined O and 0 (triggered by 'o' or '0')
        base_classes.append('v(2)')  # Combined V and 2 (triggered by 'v' or '2')
        self.valid_classes = base_classes
        self.reserved_keys = []  # No reserved keys - all letters available for gestures
        
        # Key mapping for combined classes
        self.key_to_class = {
            'l': 'l(7)',
            '7': 'l(7)',
            'o': 'o(0)',
            '0': 'o(0)',
            'v': 'v(2)',
            '2': 'v(2)',
            '-': '10'
        }
        
        # Statistics
        self.samples_collected = 0
        self.samples_per_class = {}
        
        # Load existing data if CSV exists
        self.load_existing_data()
        
        # Capture state
        self.capturing = False
        self.current_class = None
        self.last_capture_time = 0
        self.capture_cooldown = 0.5  # Seconds between captures
        self.continuous_capture = False  # Flag for holding SPACE
        self.frame_capture_count = 0  # Count captures per hold
        
    def load_existing_data(self):
        """Load existing CSV data to track samples per class."""
        if os.path.exists(self.csv_path):
            print(f"Found existing CSV: {self.csv_path}")
            try:
                with open(self.csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if row:  # Non-empty row
                            class_name = row[0]
                            self.samples_per_class[class_name] = self.samples_per_class.get(class_name, 0) + 1
                            self.samples_collected += 1
                print(f"Loaded {self.samples_collected} existing samples")
                print(f"Classes found: {dict(sorted(self.samples_per_class.items()))}")
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
        else:
            print(f"No existing CSV found. Will create new file: {self.csv_path}")
    
    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmark coordinates from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            list of 63 features (21 landmarks x 3 coordinates)
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def save_sample(self, class_name, hand_data):
        """
        Save a sample to the CSV file. Supports single hand or two-hand gestures.
        
        Args:
            class_name: Gesture class (0-9, a-z)
            hand_data: Dictionary with 'left' and/or 'right' keys containing landmark lists
                      Format: {'left': [63 features], 'right': [63 features]}
                      or {'left': [63 features]} for left-hand only
                      or {'right': [63 features]} for right-hand only
        """
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(self.csv_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        try:
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header if new file
                if not file_exists:
                    header = ['class', 'hands_used']  # hands_used: "left", "right", or "both"
                    # Left hand features
                    for i in range(21):
                        header.extend([f'left_x{i}', f'left_y{i}', f'left_z{i}'])
                    # Right hand features
                    for i in range(21):
                        header.extend([f'right_x{i}', f'right_y{i}', f'right_z{i}'])
                    writer.writerow(header)
                
                # Prepare row data
                left_landmarks = hand_data.get('left', [0.0] * 63)  # Zeros if no left hand
                right_landmarks = hand_data.get('right', [0.0] * 63)  # Zeros if no right hand
                
                # Determine hands_used
                has_left = 'left' in hand_data and any(x != 0.0 for x in hand_data['left'])
                has_right = 'right' in hand_data and any(x != 0.0 for x in hand_data['right'])
                
                if has_left and has_right:
                    hands_used = "both"
                elif has_left:
                    hands_used = "left"
                elif has_right:
                    hands_used = "right"
                else:
                    return False  # No valid hand data
                
                # Write data: class, hands_used, left features (63), right features (63)
                row = [class_name, hands_used] + left_landmarks + right_landmarks
                writer.writerow(row)
            
            # Update statistics
            self.samples_collected += 1
            self.samples_per_class[class_name] = self.samples_per_class.get(class_name, 0) + 1
            
            return True
        except Exception as e:
            print(f"Error saving sample: {e}")
            return False
    
    def draw_ui(self, frame, hand_detected, hands_data=None):
        """
        Draw user interface elements on the frame.
        
        Args:
            frame: OpenCV frame
            hand_detected: Boolean indicating if hand(s) are detected
            hands_data: Dictionary with 'left' and/or 'right' keys containing landmark data
        """
        h, w, _ = frame.shape
        
        # Title bar
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 255, 0), 2)
        
        title = "SASL Real-Time Data Collection (L/R Support)"
        cv2.putText(frame, title, (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        stats = f"Samples: {self.samples_collected} | Classes: {len(self.samples_per_class)}"
        cv2.putText(frame, stats, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Instructions panel
        instructions = [
            "Press 1,3-6,8,9, A-Z (except L,O,V) to select class",
            "Combined: L/7->l(7) | O/0->o(0) | V/2->v(2) | -->10",
            "HOLD ENTER to capture continuously",
            "Press ESC to quit | TAB = Show report",
            ""
        ]
        
        y_start = 120
        for i, instruction in enumerate(instructions):
            y = y_start + i * 30
            cv2.putText(frame, instruction, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hand detection status
        status_y = h - 150
        if hand_detected and hands_data:
            hand_parts = []
            if 'left' in hands_data:
                hand_parts.append("LEFT")
            if 'right' in hands_data:
                hand_parts.append("RIGHT")
            status_text = f"Hands Detected: {' + '.join(hand_parts)}"
            status_color = (0, 255, 0)
        else:
            status_text = "No Hands Detected"
            status_color = (0, 0, 255)
        
        cv2.rectangle(frame, (10, status_y - 35), (500, status_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Current capture mode
        if self.current_class:
            mode_y = h - 80
            
            # Show different message if continuous capture is active
            if self.continuous_capture:
                mode_text = f"🔴 RECORDING: {self.current_class.upper()} (#{self.frame_capture_count})"
                bg_color = (0, 0, 100)  # Dark red background
                text_color = (0, 100, 255)  # Red text
            else:
                mode_text = f"READY: {self.current_class.upper()}"
                bg_color = (0, 100, 0)  # Dark green background
                text_color = (0, 255, 0)  # Green text
            
            cv2.rectangle(frame, (10, mode_y - 35), (500, mode_y + 5), bg_color, -1)
            cv2.putText(frame, mode_text, (20, mode_y),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, text_color, 2)
            
            # Capture instruction
            if self.continuous_capture:
                cv2.putText(frame, "Release ENTER to stop", (20, mode_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
            else:
                cv2.putText(frame, "Hold ENTER to capture", (20, mode_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Recent samples info
        info_y = h - 20
        if self.current_class and self.current_class in self.samples_per_class:
            count = self.samples_per_class[self.current_class]
            info_text = f"Class '{self.current_class}': {count} samples"
        else:
            info_text = "Select a class (0-9, A-Z) to begin"
        
        cv2.putText(frame, info_text, (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def show_report(self):
        """Print a report of collected samples."""
        print("\n" + "="*60)
        print("DATA COLLECTION REPORT")
        print("="*60)
        print(f"Total samples collected: {self.samples_collected}")
        print(f"Total classes: {len(self.samples_per_class)}")
        print(f"CSV file: {self.csv_path}")
        print("\nSamples per class:")
        
        if self.samples_per_class:
            # Sort by class name
            sorted_classes = sorted(self.samples_per_class.items())
            for class_name, count in sorted_classes:
                bar = "█" * min(count, 50)
                print(f"  {class_name}: {count:4d} {bar}")
        else:
            print("  No samples collected yet")
        
        print("="*60 + "\n")
    
    def run(self, camera_id=0):
        """
        Run the real-time data collection.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("REAL-TIME DATA COLLECTION STARTED")
        print("="*60)
        print(f"Camera: {camera_id}")
        print(f"Output CSV: {self.csv_path}")
        print("\nInstructions:")
        print("  1. Show your hand gesture to the camera")
        print("  2. Press key for gesture class:")
        print("     - Numbers: 1, 3, 4, 5, 6, 8, 9")
        print("     - Letters: A-Z (except L, O, and V)")
        print("     - Combined gestures:")
        print("       * L or 7 -> 'l(7)' (L and seven look similar)")
        print("       * O or 0 -> 'o(0)' (O and zero look similar)")
        print("       * V or 2 -> 'v(2)' (V and two look similar)")
        print("       * - (minus) -> '10'")
        print("  3. HOLD ENTER to capture continuously (every frame)")
        print("     Release ENTER to stop capturing")
        print("  4. Press TAB to see collection report")
        print("  5. Press ESC to quit")
        print("="*60 + "\n")
        
        # Create window and make it visible
        window_name = 'SASL Data Collection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read frame from camera")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            hands_data = {}  # Dictionary to store both hands: {'left': landmarks, 'right': landmarks}
            
            # Draw landmarks if hands detected
            if results.multi_hand_landmarks:
                hand_detected = True
                
                # Process each detected hand
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Get hand label (Left or Right)
                    hand_label = handedness.classification[0].label.lower()
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract landmarks
                    hands_data[hand_label] = self.extract_landmarks(hand_landmarks)
            
            # Draw UI
            self.draw_ui(frame, hand_detected, hands_data)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input (1ms for responsive continuous capture)
            key = cv2.waitKey(1) & 0xFF
            
            # Check if ENTER is currently pressed (for continuous capture)
            if key == 13:  # ENTER key pressed
                if not self.continuous_capture:
                    # Start continuous capture mode
                    self.continuous_capture = True
                    self.frame_capture_count = 0
                    print(f"🔴 CONTINUOUS CAPTURE STARTED for class '{self.current_class}'")
            else:
                # ENTER released - stop continuous capture
                if self.continuous_capture:
                    self.continuous_capture = False
                    print(f"⏹️  CAPTURE STOPPED - Collected {self.frame_capture_count} samples")
                    self.frame_capture_count = 0
            
            # Perform capture if in continuous mode
            if self.continuous_capture:
                if self.current_class is None:
                    print("⚠️  Please select a class first (0-9, A-Z)")
                    self.continuous_capture = False
                elif not hand_detected:
                    # Skip this frame but stay in continuous mode
                    pass
                else:
                    # Capture every frame while SPACE is held
                    if self.save_sample(self.current_class, hands_data):
                        self.frame_capture_count += 1
                        count = self.samples_per_class.get(self.current_class, 0)
                        
                        # Build hand description
                        if 'left' in hands_data and 'right' in hands_data:
                            hand_desc = "Both"
                        elif 'left' in hands_data:
                            hand_desc = "Left"
                        else:
                            hand_desc = "Right"
                        
                        # Show quick feedback without blocking
                        print(f"✓ #{self.frame_capture_count} - {self.current_class} ({hand_desc}) | Total: {count}", end='\r')
            
            # Handle other keyboard inputs (not SPACE which is handled above)
            
            # Quit (ESC only)
            if key == 27:
                print("\nQuitting...")
                break
            
            # Show report (TAB key)
            elif key == 9:  # TAB key
                self.show_report()
                continue  # Skip to next frame
            
            # Handle special key mappings (combined classes and '10')
            elif key == ord('-') or key == ord('_'):
                self.current_class = '10'
                print(f"Set class to: {self.current_class}")
            
            # Set class (0-9, a-z, A-Z) - check for combined class mappings
            elif (ord('0') <= key <= ord('9')) or (ord('a') <= key <= ord('z')) or (ord('A') <= key <= ord('Z')):
                char = chr(key).lower()
                # Check if this key maps to a combined class
                if char in self.key_to_class:
                    self.current_class = self.key_to_class[char]
                    print(f"Set class to: {self.current_class} (pressed '{char}')")
                elif char in self.valid_classes:
                    self.current_class = char
                    print(f"Set class to: {self.current_class}")
                else:
                    print(f"⚠️  Invalid key '{char}' - use combined classes: o/0 → o(0), v/2 → v(2)")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Final report
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"Total samples collected this session: {self.samples_collected}")
        print(f"Data saved to: {self.csv_path}")
        
        if self.samples_per_class:
            print("\nFinal sample counts:")
            for class_name, count in sorted(self.samples_per_class.items()):
                print(f"  {class_name}: {count} samples")
        
        print("\nNext steps:")
        print("  1. Run: python train_hand_landmark_model.py")
        print("  2. Run: python realtime_camera_recognition.py")
        print("="*60 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-time hand landmark data collection with keyboard input"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/hand_landmarks.csv",
        help="Output CSV file path (default: outputs/hand_landmarks.csv)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Create collector
    collector = RealtimeDataCollector(csv_path=args.output)
    
    # Run collection
    collector.run(camera_id=args.camera)

if __name__ == "__main__":
    main()
