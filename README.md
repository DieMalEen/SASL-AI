# SASL-AI Recognition System

**South African Sign Language Recognition using Advanced Machine Learning**

A comprehensive SASL recognition system with two powerful approaches:
1. Hand Landmark Recognition: Fast, lightweight Random Forest classification using MediaPipe landmarks
2. Video-Based CNN Recognition: High-accuracy deep learning with EfficientNet + LSTM architecture

Real-time gesture recognition with multiple training pipelines, interactive data collection, and production-ready inference capabilities.

## Quick Start

### Option 1: Using the Main Menu (Recommended)

```bash
# Clone and setup
git clone <your-repo-url>
cd SASL-AI
pip install -r requirements.txt

# Launch interactive menu
python main.py
```

### Option 2: Manual Execution

#### Hand Landmark Approach
```bash
# Collect hand landmark data
python realtime_data_collection.py

# Train the Random Forest model
python train_hand_landmark_model.py

# Run live recognition
python realtime_camera_recognition.py
```

#### Video-Based CNN Approach
```bash
# Collect video dataset
python sasl_video_collector.py

# Train CNN+LSTM model
python video_cnn_only_training.py

# Run live recognition
python sasl_cnn_only_recognition.py
```


## System Overview

The SASL-AI system provides two complementary recognition approaches:

### Approach 1: Hand Landmark Recognition
- **Feature Extraction**: MediaPipe hand landmarks (21 points per hand, 3D coordinates)
- **Classifier**: Random Forest with 200 decision trees
- **Advantages**: Lightweight, fast training, CPU-efficient, minimal storage
- **Best For**: Quick prototyping, resource-constrained environments, simple gestures
- **Accuracy**: 90-97% on well-trained classes with 50+ samples

### Approach 2: Video-Based CNN Recognition
- **Architecture**: EfficientNet-B0 backbone + Bidirectional LSTM
- **Features**: Temporal modeling with 30-frame sequences
- **Advantages**: Higher accuracy, better temporal understanding, robust to variations
- **Best For**: Production systems, complex gestures, high-accuracy requirements
- **Accuracy**: 90-100% with proper training data

## Core Components

### Main Application
- **main.py**: Interactive menu system for the entire SASL-AI workflow
  - Video data collection interface
  - CNN-only model training
  - Live recognition launcher
  - Batch video prediction
  - Model management and system information

### Hand Landmark Pipeline
- **realtime_data_collection.py**: Interactive hand landmark data collector
  - Dual-hand support (up to 2 hands simultaneously)
  - Continuous capture mode (hold ENTER for rapid collection)
  - 34 gesture classes with combined visually similar gestures
  - Real-time statistics and quality feedback

- **train_hand_landmark_model.py**: Random Forest training system
  - Handles mixed CSV formats (65-column and 128-column)
  - Automatic class combination for similar gestures
  - Comprehensive evaluation metrics and visualizations
  - Exports trained models with scalers and encoders

- **realtime_camera_recognition.py**: Live hand landmark recognition
  - Automatic model detection
  - Prediction smoothing with temporal voting
  - Dual-hand gesture support
  - Real-time confidence display

### Video-Based CNN Pipeline
- **sasl_video_collector.py**: Video dataset collection tool
  - 3-5 second video sequences per sign
  - Real-time MediaPipe quality checking
  - Background removal options
  - Automatic file organization and preview

- **video_cnn_only_training.py**: CNN+LSTM model training
  - EfficientNet-B0 transfer learning
  - Bidirectional LSTM for temporal modeling
  - Optional hand landmark fusion for enhanced accuracy
  - Data augmentation and caching for efficiency
  - GPU acceleration support

- **sasl_cnn_only_recognition.py**: Live CNN-based recognition
  - Real-time video processing
  - Optional MediaPipe hand detection integration
  - Automatic model fallback (CNN+Landmarks or CNN-only)
  - Confidence thresholding and prediction smoothing

### Utilities
- **batch_video_prediction.py**: Batch inference tool
  - Process multiple videos automatically
  - Supports CNN+LSTM and Combined CNN+Hand models
  - Generates detailed JSON results
  - Progress tracking with tqdm

## Installation

### System Requirements
- **Python**: 3.11 or higher
- **OS**: Windows 10/11, macOS, or Linux
- **GPU**: Optional (CUDA-compatible for faster CNN training)
- **RAM**: 8GB minimum, 16GB recommended for CNN training
- **Storage**: 2GB free space for models and datasets
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

# Deep Learning (for CNN models)
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
timm>=0.9.0

# Machine Learning (for Random Forest)
scikit-learn>=1.3.0
numpy>=1.24.0

# Data Processing & Visualization
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Model Persistence & Progress
joblib>=1.3.0
tqdm>=4.65.0
```


## Usage Guide

### Main Menu System

The easiest way to use SASL-AI is through the interactive menu:

```bash
python main.py
```

**Menu Options:**
1. **Collect Video Data**: Launch interactive video collector for CNN training
2. **Train CNN-Only Models**: Train deep learning models on video dataset
3. **CNN-Only Live Recognition**: Real-time recognition using trained CNN models
4. **Batch Video Prediction**: Process multiple videos automatically
5. **Manage Models & Outputs**: View and clean training sessions
6. **System Information**: Check dependencies and dataset statistics
7. **Exit**: Close the application

### Hand Landmark Workflow

#### 1. Data Collection

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
- `L` or `7` -> l(7) class
- `O` or `0` -> o(0) class
- `V` or `2` -> v(2) class
- `-` (minus) -> 10 class

**Output:**
```
outputs/
└── hand_landmarks.csv    # CSV with 128 columns
                          # Format: class, hands_used, 126 features
```

#### 2. Model Training

Train the Random Forest classifier:

```bash
python train_hand_landmark_model.py
```

**Training Features:**
- Mixed format support (65-column and 128-column CSV)
- Automatic class combination for similar gestures
- 200-tree Random Forest with depth limit
- 80/20 train/test split with stratification
- StandardScaler normalization
- Comprehensive metrics and visualizations

**Output Structure:**
```
outputs/hand_landmark_model_YYYYMMDD_HHMMSS/
├── random_forest_model.joblib       # Trained model
├── scaler.joblib                    # Feature scaler
├── label_encoder.joblib             # Label encoder
├── confusion_matrix.png             # Confusion matrix
├── feature_importance.png           # Feature importance
├── class_distribution.png           # Data distribution
└── training_report.txt              # Classification report
```

#### 3. Live Recognition

Launch real-time hand landmark recognition:

```bash
python realtime_camera_recognition.py
```

**Features:**
- Auto-detection of latest trained model
- Dual-hand gesture support
- Prediction smoothing (10-frame voting)
- Real-time confidence display
- Hand landmark overlay

**Controls:**
- **'Q'** or **ESC**: Quit

### Video-Based CNN Workflow

#### 1. Video Collection

Collect video sequences for training:

```bash
python sasl_video_collector.py
```

**Or use the main menu (Option 1)**

**Collection Features:**
- 3-5 second video recordings per sign
- Real-time MediaPipe quality checking
- Background removal options (solid color, blur, transparent)
- Preview and review recorded videos
- Automatic file organization

**Output:**
```
video_dataset/
├── 1/
│   ├── 1_001_20251020_120000.mp4
│   ├── 1_002_20251020_120010.mp4
│   └── ...
├── a/
│   └── ...
└── ...
```

#### 2. CNN Training

Train CNN+LSTM models on video dataset:

```bash
python video_cnn_only_training.py
```

**Or use the main menu (Option 2)**

**Training Configuration:**
- Quick training: Default settings (50 epochs, batch 8, 1x augmentation)
- Custom configuration: Configure epochs, batch size, augmentation factor, learning rate

**Model Architecture:**
- EfficientNet-B0 backbone (pre-trained on ImageNet)
- Temporal Conv1D layers
- Bidirectional LSTM (2 layers)
- Optional hand landmark fusion
- Dropout regularization

**Training Features:**
- Transfer learning with frozen backbone
- Video frame caching for faster training
- Data augmentation (rotation, brightness, contrast, flipping)
- GPU acceleration support
- Early stopping and best model checkpointing
- Comprehensive evaluation metrics

**Output Structure:**
```
outputs/training_YYYYMMDD_HHMMSS/
├── models/
│   ├── best_sasl_cnn_lstm_model.pth
│   └── (optional) best_sasl_combined_model.pth
├── results/
│   ├── class_names.json
│   ├── training_config.json
│   ├── confusion_matrix.png
│   └── training_results.json
└── cache/
    └── (cached video frames)
```

#### 3. Live CNN Recognition

Launch real-time CNN-based recognition:

```bash
python sasl_cnn_only_recognition.py
```

**Or use the main menu (Option 3)**

**Features:**
- Automatic latest model detection
- Optional MediaPipe hand landmark integration
- Confidence thresholding (default 0.3)
- Temporal smoothing with frame buffer
- GPU acceleration if available

**Model Modes:**
- CNN+Hand Landmarks: Uses both visual features and hand landmarks
- CNN-only: Uses only visual features (fallback mode)

**Controls:**
- **'Q'** or **ESC**: Quit

#### 4. Batch Prediction

Process multiple videos automatically:

```bash
python batch_video_prediction.py
```

**Or use the main menu (Option 4)**

**Features:**
- Automatic model detection
- Processes all videos in test_videos/ folder
- Generates detailed JSON results
- Progress tracking
- Supports multiple video formats (mp4, avi, mov, mkv, wmv, flv)

**Output:**
- JSON file with predictions, confidences, and processing times


## Technical Architecture

### Hand Landmark Recognition Architecture

**MediaPipe Hands Configuration:**
- **Max Hands**: 2 (supports dual-hand gestures)
- **Detection Confidence**: 0.7 (data collection) / 0.5 (recognition)
- **Tracking Confidence**: 0.7 (data collection) / 0.5 (recognition)
- **Static Image Mode**: False (optimized for video streams)

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

**Random Forest Classifier:**
- **Estimators**: 200 decision trees
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Random State**: 42 (reproducible results)
- **Feature Preprocessing**: StandardScaler normalization

**CSV Data Format (128 columns):**
1. `class` - Gesture class label
2. `hands_used` - Hand usage ('left', 'right', or 'both')
3. `left_x0` to `left_z20` - Left hand features (63 columns)
4. `right_x0` to `right_z20` - Right hand features (63 columns)

### CNN-Based Recognition Architecture

**Model: CNNLSTMModel**
- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
  - 1280 feature dimensions
  - Frozen for transfer learning
- **Temporal Processing**:
  - Conv1D layer (1280 -> 512 features, kernel=3)
  - BatchNorm1D + Dropout (0.3)
- **LSTM Layers**:
  - LSTM1: Bidirectional (512 -> 256 hidden, outputs 512)
  - LSTM2: Bidirectional (512 -> 128 hidden, outputs 256)
  - Dropout (0.3) between layers
- **Classifier Head**:
  - Linear (256 -> 256) + ReLU + Dropout (0.5)
  - Linear (256 -> 128) + ReLU + Dropout (0.3)
  - Linear (128 -> num_classes)

**Model: CombinedCNNHandModel (Optional)**
- Combines CNNLSTMModel with hand landmark features
- **Hand Landmark Branch**:
  - HandLandmarkLSTM: Bidirectional LSTM (126 -> 64 hidden)
  - Processes 30-frame sequences of hand landmarks
- **Fusion**:
  - Concatenates CNN features (256) + Hand features (128)
  - Fusion classifier: Linear (384 -> 256 -> num_classes)

**Training Configuration:**
- **Input Size**: 224x224 RGB images
- **Sequence Length**: 30 frames per video
- **Normalization**: ImageNet mean and std
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: CrossEntropyLoss
- **Batch Size**: 8 (default, configurable)
- **Epochs**: 50 (default, configurable)

**Data Augmentation:**
- Random rotation (-15 to +15 degrees)
- Random brightness adjustment (0.8 to 1.2)
- Random contrast adjustment (0.8 to 1.2)
- Random horizontal flip (50% probability)

**Video Processing:**
- Uniform frame sampling (30 frames from full video)
- Frame resize to 224x224
- Tensor normalization with ImageNet statistics
- Optional hand landmark extraction with MediaPipe

### Performance Metrics

#### Hand Landmark Model
- **Accuracy**: 90-97% on well-trained classes with 50+ samples
- **Processing Speed**: 30+ FPS real-time recognition (CPU)
- **Memory Usage**: <500MB training, <200MB inference
- **Training Time**: 1-5 minutes for 35 classes (CPU)
- **Model Size**: <50MB

#### CNN+LSTM Model
- **Accuracy**: 90-100% with sufficient training data (5+ videos per class)
- **Processing Speed**: 15-30 FPS (depends on hardware)
- **Memory Usage**: 2-4GB training (GPU), <1GB inference
- **Training Time**: 10-60 minutes (depends on dataset size, epochs, GPU)
- **Model Size**: 20-50MB (depending on classes)

#### Dataset Recommendations

**Hand Landmark Collection:**
- Samples per class: Minimum 30, optimal 100+
- Collection speed: 10-30 samples/second (continuous mode)
- Camera: 720p minimum, good lighting
- Background: Clean, contrasting background
- Hand position: Visible, centered in frame

**Video Collection:**
- Videos per class: Minimum 5, optimal 10-20+
- Video duration: 3-5 seconds
- FPS: 30 (recommended)
- Resolution: 640x480 or higher
- Quality checks: MediaPipe hand detection during recording
- Variation: Different angles, lighting, backgrounds


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

#### Poor Hand Landmark Recognition Accuracy
**Solutions:**
- Collect more training samples (100+ per class recommended)
- Ensure consistent gesture execution during collection
- Collect samples with varied hand angles
- Verify hand landmarks are detected during collection
- Retrain model after collecting more data

#### Poor CNN Model Accuracy
**Solutions:**
- Collect more videos per class (10-20+ recommended)
- Increase training epochs (try 75-100)
- Enable data augmentation (factor 1-3)
- Ensure video quality is good (clear hands, proper lighting)
- Check MediaPipe hand detection during video collection
- Try Combined CNN+Hand model for better accuracy

#### CSV Format Errors
**Solutions:**
- Delete corrupted hand_landmarks.csv and restart collection
- Training script automatically handles mixed formats
- Check for partial writes (incomplete rows)

#### Import/Dependency Errors
```bash
# Reinstall core dependencies
pip uninstall opencv-python mediapipe scikit-learn torch torchvision
pip install -r requirements.txt
```

#### GPU/CUDA Issues (CNN Training)
**Solutions:**
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Update GPU drivers
- Reinstall PyTorch with CUDA: Visit pytorch.org for install command
- System will automatically fall back to CPU if GPU unavailable

#### Out of Memory During CNN Training
**Solutions:**
- Reduce batch size (try 4 or 2)
- Reduce sequence length (try 20 instead of 30)
- Close other GPU applications
- Use CPU mode if GPU memory insufficient
- Enable frame caching to reduce memory usage

### Performance Optimization

#### Hand Landmark Collection Tips
- Use continuous capture (HOLD ENTER) for rapid collection
- Collect 50-100 samples per class minimum
- Vary hand position and angle slightly for robustness
- Ensure both hands are visible for dual-hand gestures
- Press TAB regularly to check sample distribution

#### Video Collection Tips
- Record 10-20 videos per class
- Maintain consistent 3-5 second duration
- Vary lighting conditions slightly
- Include different backgrounds
- Ensure hands are clearly visible
- Use MediaPipe quality indicator before recording

#### CNN Training Optimization
- Start with default settings (50 epochs, batch 8)
- Use GPU for 5-10x faster training
- Enable 1x augmentation for better generalization
- Cache videos for faster repeated training
- Monitor training curves for overfitting

#### Recognition Optimization
- Use good lighting conditions
- Position camera at chest/face level
- Keep hands centered in frame
- Execute gestures clearly and consistently
- Allow model to stabilize (prediction smoothing)
- For CNN: Use GPU for faster inference


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

## File Structure

```
SASL-AI/
├── main.py                          # Interactive menu system (main entry point)
├── realtime_data_collection.py      # Hand landmark data collector
├── train_hand_landmark_model.py     # Random Forest training
├── realtime_camera_recognition.py   # Hand landmark live recognition
├── sasl_video_collector.py          # Video dataset collector
├── video_cnn_only_training.py       # CNN+LSTM training
├── sasl_cnn_only_recognition.py     # CNN-based live recognition
├── batch_video_prediction.py        # Batch video inference tool
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
├── outputs/                         # Training outputs and models
│   ├── hand_landmarks.csv           # Collected hand landmark data
│   ├── hand_landmark_model_*/       # Random Forest model outputs
│   ├── training_*/                  # CNN training session outputs
│   │   ├── models/                  # Trained PyTorch models (.pth)
│   │   ├── results/                 # Training results and metrics
│   │   └── cache/                   # Cached video frames
│   └── video_cache/                 # Video processing cache
├── video_dataset/                   # Video training dataset
│   ├── 1/                           # Class directories
│   ├── a/
│   └── ...
├── test_videos/                     # Videos for batch prediction
└── __pycache__/                     # Python cache files
```

## Advanced Features

### Continuous Capture Mode (Hand Landmarks)
- HOLD ENTER to capture samples on every frame
- 10-30x faster than single-capture mode
- Live counter shows samples collected
- Visual feedback with "RECORDING" indicator

### Dual-Hand Support
- Detects and tracks up to 2 hands simultaneously
- Tracks which hand(s) used for each gesture
- Supports left-only, right-only, and two-hand gestures
- Backward compatible with single-hand data

### Mixed Format Handling
- Training automatically handles both old and new CSV formats
- Converts 65-column (single hand) to 128-column (dual hand)
- No need to recollect old data

### Transfer Learning (CNN)
- Pre-trained EfficientNet-B0 on ImageNet
- Frozen backbone for efficient training
- Fine-tuned classification head
- Significantly reduces training time and data requirements

### Video Caching
- Preprocessed frames cached to disk
- 5-10x faster subsequent training runs
- Automatic cache invalidation on parameter changes
- Reduces memory usage during training

### Data Augmentation
- Configurable augmentation factor (0-3x)
- Rotation, brightness, contrast, and flipping
- Applied during training for better generalization
- No impact on inference speed

### Model Management
- Automatic latest model detection
- Multiple training session support
- Clean old models through main menu
- View training results and statistics

## System Requirements Summary

### Minimum Requirements
- Python 3.11+
- 8GB RAM
- Webcam
- 2GB storage
- CPU: Dual-core 2.0GHz+

### Recommended Configuration
- Python 3.11+
- 16GB RAM
- CUDA-compatible GPU (for CNN training)
- Webcam (720p or higher)
- 5GB storage
- CPU: Quad-core 3.0GHz+

## Project Highlights

### Two Complementary Approaches
- **Hand Landmarks**: Fast, lightweight, CPU-efficient
- **Video CNN**: High accuracy, robust, production-ready

### Complete Workflow
- Data collection tools for both approaches
- Training pipelines with comprehensive metrics
- Real-time recognition systems
- Batch inference capabilities
- Model management utilities

### User-Friendly
- Interactive main menu system
- Auto-detection of trained models
- Clear progress indicators
- Comprehensive error handling
- Detailed documentation

### Production-Ready
- GPU acceleration support
- Frame caching for efficiency
- Prediction smoothing for stability
- Configurable confidence thresholds
- Batch processing capabilities

---

**SASL-AI Recognition System**

Advanced South African Sign Language recognition using MediaPipe, Random Forest, and Deep Learning with EfficientNet + LSTM architecture.

Empowering communication through AI-powered sign language recognition.