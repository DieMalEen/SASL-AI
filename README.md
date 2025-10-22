# SASL-AI Recognition System

<div align="center">

![SASL-AI Banner](https://img.shields.io/badge/SASL--AI-Hand_Landmarks-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**South African Sign Language Recognition using MediaPipe Hand Landmarks and Random Forest**

Real-time SASL gesture recognition with MediaPipe hand landmark extraction and machine learning classification for accurate and efficient sign detection.

</div>

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd SASL-AI
pip install -r requirements.txt

# Collect hand landmark data
python realtime_data_collection.py

# Train the Random Forest model
python train_hand_landmark_model.py

# Run live recognition
python realtime_camera_recognition.py
```

## System Overview

### Core Architecture
- **MediaPipe Hand Tracking**: Detects up to 2 hands simultaneously with 21 landmarks per hand
- **Hand Landmark Features**: 126-dimensional feature vectors (63 per hand: 21 landmarks x 3 coordinates)
- **Random Forest Classifier**: 200 decision trees for robust gesture classification
- **Real-time Processing**: 30+ FPS recognition with minimal computational requirements
- **Combined Gesture Classes**: Merges visually similar gestures (L/7, O/0, V/2) for improved accuracy

### Key Components
- **Real-time Data Collection**: Interactive keyboard-driven data collection with continuous capture mode
- **Dual-Hand Support**: Recognizes gestures using left hand, right hand, or both hands simultaneously
- **Scikit-learn Training Pipeline**: Efficient machine learning with comprehensive evaluation metrics
- **Live Camera Recognition**: Real-time gesture detection with confidence scoring and prediction smoothing

## Installation

### System Requirements
- **Python**: 3.11 or higher
- **OS**: Windows 10/11, macOS, or Linux
- **GPU**: Not required (CPU-based processing)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space for models and dataset
- **Camera**: Webcam or external camera for data collection and recognition

### Dependencies Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
# Computer Vision & Hand Tracking
opencv-python>=4.8.0
mediapipe>=0.10.7

# Machine Learning
scikit-learn>=1.3.0
numpy>=1.24.0

# Data Processing & Visualization
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Model Persistence
joblib>=1.3.0
```

## Usage Guide

### 1. Data Collection

Collect hand landmark data for SASL gestures:

```bash
python realtime_data_collection.py
```

**Collection Process:**
- Press keys to select gesture class (0-9, A-Z)
- Show gesture to camera
- HOLD ENTER to capture continuously (frame-by-frame)
- Release ENTER to stop capturing
- Press TAB to view collection statistics
- Press ESC to quit

**Supported Classes (34 total):**
- Numbers: 1, 3, 4, 5, 6, 8, 9, 10
- Letters: a, b, c, d, e, f, g, h, i, j, k, m, n, p, q, r, s, t, u, w, x, y, z
- Combined: l(7), o(0), v(2)

**Key Mappings:**
- `L` or `7` -> l(7) class (visually similar)
- `O` or `0` -> o(0) class (visually similar)
- `V` or `2` -> v(2) class (visually similar)
- `-` (minus) -> 10 class

**Dataset Output:**
```
outputs/
└── hand_landmarks.csv    # CSV with 128 columns
                          # Format: class, hands_used, 126 features
```

### 2. Model Training

Train the Random Forest classifier on collected hand landmark data:

```bash
python train_hand_landmark_model.py
```

**Training Features:**
- **Mixed Format Support**: Handles both old (65-column) and new (128-column) CSV formats
- **Class Combination**: Automatically merges visually similar classes (0/o, 2/v)
- **Random Forest**: 200 trees with max_depth=20 for robust classification
- **Train/Test Split**: 80/20 split with stratification
- **Feature Scaling**: StandardScaler normalization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Class Filtering**: Removes classes with fewer than 2 samples

**Training Output Structure:**
```
outputs/hand_landmark_model_YYYYMMDD_HHMMSS/
├── random_forest_model.joblib       # Trained Random Forest model
├── scaler.joblib                    # Feature scaler
├── label_encoder.joblib             # Class label encoder
├── confusion_matrix.png             # Confusion matrix visualization
├── feature_importance.png           # Top feature importance plot
├── class_distribution.png           # Training data distribution
└── training_report.txt              # Detailed classification report
```

### 3. Live Recognition

Launch real-time SASL gesture recognition:

```bash
python realtime_camera_recognition.py
```

**Recognition Features:**
- **Dual-Hand Detection**: Recognizes gestures using left, right, or both hands
- **Auto-Model Detection**: Automatically loads the latest trained model
- **Format Compatibility**: Works with both 63-feature (single hand) and 126-feature (dual hand) models
- **Prediction Smoothing**: Uses majority voting over last 10 frames for stable predictions
- **Confidence Display**: Shows prediction confidence percentage
- **Hand Visualization**: Draws MediaPipe hand landmarks on video feed

**On-Screen Display:**
- Current prediction with confidence
- Hand usage indicator (LEFT, RIGHT, or LEFT + RIGHT)
- Hand landmarks overlay
- Frame counter

**Controls:**
- **'Q'**: Quit recognition
- **ESC**: Quit recognition

## Technical Architecture

### Hand Landmark Extraction

**MediaPipe Hands Configuration:**
- **Max Hands**: 2 (supports dual-hand gestures)
- **Detection Confidence**: 0.7
- **Tracking Confidence**: 0.7
- **Static Image Mode**: False (optimized for video)

**Landmark Structure:**
- 21 landmarks per hand
- 3 coordinates per landmark (x, y, z)
- Total: 63 features per hand
- Combined: 126 features for two hands

**Landmark Points:**
```
WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP,
MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP,
RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP,
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
```

### Classification Model

**Random Forest Classifier:**
- **Estimators**: 200 decision trees
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Random State**: 42 (reproducible results)
- **Feature Preprocessing**: StandardScaler normalization

### Data Format

**CSV Structure (128 columns):**
1. `class` - Gesture class label (e.g., 'a', '5', 'o(0)', 'v(2)')
2. `hands_used` - Hand usage ('left', 'right', or 'both')
3. `left_x0` to `left_z20` - Left hand features (63 columns)
4. `right_x0` to `right_z20` - Right hand features (63 columns)

**Backward Compatibility:**
- Old 65-column format: class, hand_label, 63 features
- Automatically converted to 128-column format during training
- Missing hand features filled with zeros

### Performance Metrics

#### Model Performance
- **Accuracy**: 90-97% on well-trained classes with 50+ samples each
- **Processing Speed**: 30+ FPS real-time recognition (CPU-based)
- **Memory Usage**: <500MB during training, <200MB during inference
- **Training Time**: 1-5 minutes for 35 classes (CPU)
- **Model Size**: <50MB (lightweight Random Forest)

#### Dataset Recommendations
- **Samples per Class**: Minimum 30, optimal 100+ samples
- **Data Collection**: Use continuous capture mode (10-30 samples/second)
- **Camera Quality**: 720p minimum, good lighting conditions
- **Background**: Clean, contrasting background preferred
- **Hand Position**: Keep hands visible within camera frame
- **Variation**: Collect samples with different hand angles and positions

## Troubleshooting

### Common Issues & Solutions

#### Camera Not Detected
```python
# Test camera access
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
```

**Solutions:**
- Check camera permissions in OS settings
- Try different camera IDs (0, 1, 2)
- Ensure no other applications are using the camera
- Reconnect external camera if applicable

#### Hand Landmarks Not Detected
**Solutions:**
- Ensure adequate lighting in room
- Keep hands within camera frame
- Use plain/contrasting background
- Clean camera lens
- Check MediaPipe installation: `pip install --upgrade mediapipe`

#### Poor Recognition Accuracy
**Solutions:**
- Collect more training samples (100+ per class recommended)
- Ensure consistent gesture execution during collection
- Collect samples with varied hand angles
- Verify hand landmarks are detected during collection
- Retrain model after collecting more data

#### CSV Format Errors
**Solutions:**
- Delete corrupted hand_landmarks.csv and restart collection
- Training script automatically handles mixed formats
- Check for partial writes (incomplete rows)

#### Import/Dependency Errors
```bash
# Reinstall core dependencies
pip uninstall opencv-python mediapipe scikit-learn
pip install -r requirements.txt
```

### Performance Optimization

#### Data Collection Tips
- Use continuous capture (HOLD ENTER) for rapid collection
- Collect 50-100 samples per class minimum
- Vary hand position and angle slightly for robustness
- Ensure both hands are visible for dual-hand gestures
- Press TAB regularly to check sample distribution

#### Recognition Optimization
- Use good lighting conditions
- Position camera at chest/face level
- Keep hands centered in frame
- Execute gestures clearly and consistently
- Allow model to stabilize (prediction smoothing takes ~10 frames)

## Combined Gesture Classes

To improve recognition accuracy, visually similar gestures are combined into single classes:

### l(7) - Letter L and Number 7
- **Why Combined**: L-shaped hand position looks identical
- **Trigger Keys**: Press `L` or `7` to select this class
- **Label**: Displays as "l(7)" in recognition

### o(0) - Letter O and Number 0
- **Why Combined**: Circular/oval hand shape looks identical
- **Trigger Keys**: Press `O` or `0` to select this class
- **Label**: Displays as "o(0)" in recognition

### v(2) - Letter V and Number 2
- **Why Combined**: Two fingers extended (peace sign = number 2)
- **Trigger Keys**: Press `V` or `2` to select this class
- **Label**: Displays as "v(2)" in recognition

**Benefits:**
- Eliminates model confusion between visually identical gestures
- Improves overall accuracy by reducing ambiguous classifications
- Historical data automatically converted during training

See `COMBINED_CLASSES.md` for detailed information.

## File Structure

```
SASL-AI/
├── realtime_data_collection.py      # Interactive hand landmark data collection
├── train_hand_landmark_model.py     # Random Forest model training
├── realtime_camera_recognition.py   # Real-time gesture recognition
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
├── COMBINED_CLASSES.md              # Combined classes guide
├── TWO_HAND_GUIDE.md                # Two-hand feature documentation
├── CONTROLS_UPDATE.md               # Control key reference
├── CONTINUOUS_CAPTURE.md            # Continuous capture feature guide
├── outputs/                         # Training outputs and models
│   ├── hand_landmarks.csv           # Collected training data
│   ├── hand_landmark_model_*/       # Trained model outputs
│   └── video_cache/                 # Cache directory
└── __pycache__/                     # Python cache files
```

## Advanced Features

### Continuous Capture Mode
- HOLD ENTER to capture samples on every frame
- 10-30x faster than single-capture mode
- Live counter shows samples collected
- Visual feedback with red "RECORDING" indicator

### Dual-Hand Support
- Detects and tracks up to 2 hands simultaneously
- Tracks which hand(s) used for each gesture
- Supports left-only, right-only, and two-hand gestures
- Backward compatible with single-hand data

### Mixed Format Handling
- Training automatically handles both old and new CSV formats
- Converts 65-column (single hand) to 128-column (dual hand)
- No need to recollect old data

---

<div align="center">

**SASL Hand Landmark Recognition System**

![MediaPipe](https://img.shields.io/badge/MediaPipe-0F9D58?style=for-the-badge&logo=google&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Empowering communication through AI-powered sign language recognition

</div>