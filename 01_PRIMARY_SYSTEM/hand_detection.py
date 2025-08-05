# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
import math

# Suppress MediaPipe specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

class HandDetector:
    """
    Advanced hand detection and tracking using MediaPipe
    """
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark connections for drawing
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        
    def detect_hands(self, frame):
        """
        Detect hands in a frame and return hand landmarks and bounding boxes
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            hands_data: List of dictionaries containing hand information
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            )):
                # Extract hand information
                hand_info = self._extract_hand_info(hand_landmarks, handedness, frame.shape)
                hands_data.append(hand_info)
                
        return hands_data
    
    def _extract_hand_info(self, hand_landmarks, handedness, frame_shape):
        """
        Extract detailed information about detected hand
        """
        h, w, _ = frame_shape
        
        # Get landmark coordinates
        landmarks = []
        x_coords = []
        y_coords = []
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
            x_coords.append(x)
            y_coords.append(y)
        
        # Calculate bounding box with padding
        bbox = self._calculate_bbox(x_coords, y_coords, w, h)
        
        # Get hand label (Left/Right) - Switch from camera perspective to person's actual perspective
        hand_label = handedness.classification[0].label
        # Switch the perspective: camera's "Left" is person's "Right" and vice versa
        hand_label = "Right" if hand_label == "Left" else "Left"
        confidence = handedness.classification[0].score
        
        return {
            'landmarks': landmarks,
            'bbox': bbox,
            'hand_label': hand_label,
            'confidence': confidence,
            'raw_landmarks': hand_landmarks
        }
    
    def _calculate_bbox(self, x_coords, y_coords, frame_w, frame_h, padding_ratio=0.2):
        """
        Calculate bounding box around hand with padding
        """
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding
        width = max_x - min_x
        height = max_y - min_y
        
        # Use larger dimension for square bounding box
        size = max(width, height)
        padding = int(size * padding_ratio)
        
        # Calculate center
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # Calculate square bbox
        half_size = (size + padding) // 2
        
        bbox_x1 = max(0, center_x - half_size)
        bbox_y1 = max(0, center_y - half_size)
        bbox_x2 = min(frame_w, center_x + half_size)
        bbox_y2 = min(frame_h, center_y + half_size)
        
        return [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
    
    def draw_hands(self, frame, hands_data, draw_landmarks=True, draw_bbox=True):
        """
        Draw hand landmarks and bounding boxes on frame
        """
        annotated_frame = frame.copy()
        
        for hand_data in hands_data:
            if draw_bbox:
                # Draw bounding box
                bbox = hand_data['bbox']
                cv2.rectangle(annotated_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # Draw hand label
                label = f"{hand_data['hand_label']} ({hand_data['confidence']:.2f})"
                cv2.putText(annotated_frame, label, 
                          (bbox[0], bbox[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if draw_landmarks:
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_data['raw_landmarks'],
                    self.hand_connections,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_frame
    
    def extract_hand_regions(self, frame, hands_data, target_size=(224, 224)):
        """
        Extract and resize hand regions from frame
        """
        hand_regions = []
        
        for hand_data in hands_data:
            bbox = hand_data['bbox']
            
            # Extract hand region
            hand_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if hand_region.size > 0:
                # Resize to target size
                hand_region = cv2.resize(hand_region, target_size)
                hand_regions.append({
                    'region': hand_region,
                    'hand_label': hand_data['hand_label'],
                    'confidence': hand_data['confidence'],
                    'bbox': bbox
                })
        
        return hand_regions
    
    def get_hand_features(self, hands_data):
        """
        Extract numerical features from hand landmarks for additional processing
        """
        features = []
        
        for hand_data in hands_data:
            landmarks = hand_data['landmarks']
            
            # Calculate various hand features
            hand_features = self._calculate_hand_features(landmarks)
            hand_features['hand_label'] = hand_data['hand_label']
            hand_features['confidence'] = hand_data['confidence']
            
            features.append(hand_features)
        
        return features
    
    def _calculate_hand_features(self, landmarks):
        """
        Calculate geometric features from hand landmarks
        """
        if len(landmarks) < 21:  # MediaPipe provides 21 landmarks
            return {}
        
        # Convert to numpy array for easier calculation
        landmarks_array = np.array(landmarks)
        
        # Calculate center of hand
        center = np.mean(landmarks_array, axis=0)
        
        # Calculate distances from center to each landmark
        distances = [np.linalg.norm(landmark - center) for landmark in landmarks_array]
        
        # Calculate angles between consecutive landmarks
        angles = []
        for i in range(len(landmarks_array) - 1):
            v1 = landmarks_array[i] - center
            v2 = landmarks_array[i + 1] - center
            angle = math.atan2(np.cross(v1, v2), np.dot(v1, v2))
            angles.append(angle)
        
        return {
            'center': center.tolist(),
            'distances': distances,
            'angles': angles,
            'bbox_area': self._calculate_bbox_area(landmarks_array),
            'aspect_ratio': self._calculate_aspect_ratio(landmarks_array)
        }
    
    def _calculate_bbox_area(self, landmarks_array):
        """Calculate bounding box area of landmarks"""
        min_coords = np.min(landmarks_array, axis=0)
        max_coords = np.max(landmarks_array, axis=0)
        return (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
    
    def _calculate_aspect_ratio(self, landmarks_array):
        """Calculate aspect ratio of hand bounding box"""
        min_coords = np.min(landmarks_array, axis=0)
        max_coords = np.max(landmarks_array, axis=0)
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        return width / height if height > 0 else 1.0
    
    def close(self):
        """Clean up resources"""
        self.hands.close()

# Utility functions for hand-focused video processing
def extract_hand_focused_frames(video_path, hand_detector, num_frames=16, target_size=(224, 224)):
    """
    Extract frames from video with focus on hand regions
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_idxs = list(range(total_frames))
        frame_idxs.extend([total_frames - 1] * (num_frames - total_frames))
    else:
        frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    hand_focused_frames = []
    full_frames = []
    
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            if hand_focused_frames:
                frame = hand_focused_frames[-1].copy()
            else:
                frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        full_frames.append(frame)
        
        # Detect hands in frame
        hands_data = hand_detector.detect_hands(frame)
        
        if hands_data:
            # Extract hand regions
            hand_regions = hand_detector.extract_hand_regions(frame, hands_data, target_size)
            
            if hand_regions:
                # Use the first (most confident) hand region
                hand_focused_frames.append(hand_regions[0]['region'])
            else:
                # Fallback to resized full frame
                hand_focused_frames.append(cv2.resize(frame, target_size))
        else:
            # No hands detected, use resized full frame
            hand_focused_frames.append(cv2.resize(frame, target_size))
    
    cap.release()
    
    # Ensure we have exactly num_frames
    while len(hand_focused_frames) < num_frames:
        hand_focused_frames.append(hand_focused_frames[-1])
    
    return hand_focused_frames[:num_frames], full_frames[:num_frames]
