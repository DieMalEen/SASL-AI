# SASL-AI Recognition System

<div align="center">

![SASL-AI Banner](https://img.shields.io/badge/SASL--AI-PyTorch-blue?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Advanced South African Sign Language Recognition using PyTorch Deep Learning**

Real-time SASL recognition with CNN+LSTM and MediaPipe hand landmark fusion models for enhanced accuracy and robustness.

</div>

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd SASL-AI
pip install -r requirements.txt

# Run video collection
python sasl_video_collector.py

# Train the model
python video_cnn_only_training.py

# Run live recognition
python sasl_cnn_only_recognition.py
```

## System Overview

### Core Architecture
- **CNN+LSTM Model**: EfficientNet backbone with bidirectional LSTM for temporal modeling
- **MediaPipe Hand Landmarks**: 21 hand landmarks per hand (up to 2 hands) for precise gesture analysis
- **Combined Fusion Model**: Integrates video frames and hand landmark data for superior accuracy
- **Real-time Processing**: 30+ FPS recognition with GPU acceleration
- **Hand Overlay Toggle**: Visual feedback showing detected hand landmarks

### Key Components
- **Video Data Collection**: Structured dataset creation with automated organization
- **PyTorch Training Pipeline**: Modern deep learning with comprehensive monitoring
- **Live Camera Recognition**: Real-time sign language detection with confidence scoring
- **Hand Landmark Integration**: MediaPipe-powered hand tracking for enhanced accuracy

## Installation

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows 10/11, macOS, or Linux
- **GPU**: CUDA-compatible GPU (optional, but recommended)
- **RAM**: 8GB minimum, 16GB recommended for training
- **Storage**: 5GB free space for models and dataset

### Dependencies Installation

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install PyTorch (GPU version with CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Required Python Packages

```
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
timm>=0.9.0

# Computer Vision & Processing  
opencv-python>=4.8.0
mediapipe>=0.10.7
pillow>=10.0.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Progress & Utilities
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Usage Guide

### 1. Data Collection

Create training videos for your SASL signs:

```bash
python sasl_video_collector.py
```

**Collection Process:**
- Select sign class name
- Record 2-5 second videos per sign
- Automatic file naming and organization
- Real-time video preview
- Structured dataset creation

**Dataset Structure:**
```
video_dataset/
├── sign1/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
├── sign2/
│   ├── video001.mp4
│   └── ...
└── ...
```

### 2. Model Training

Train the combined CNN+Hand landmark model:

```bash
python video_cnn_only_training.py
```

**Training Features:**
- **Dual Architecture**: CNN branch for video frames, Hand branch for MediaPipe landmarks
- **Data Augmentation**: Brightness, contrast, rotation, noise, and temporal augmentation
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Comprehensive Monitoring**: Real-time loss and accuracy tracking
- **GPU Acceleration**: Automatic CUDA detection and optimization

**Training Output Structure:**
```
outputs/training_YYYYMMDD_HHMMSS/
├── models/
│   └── best_sasl_cnn_lstm_model.pth
├── plots/
│   └── cnn_lstm_training_history.png
├── confusion_matrices/
│   └── cnn_hand_fusion_confusion_matrix.png
└── results/
    ├── cnn_training_results.json
    ├── class_names.json
    └── training_summary.txt
```

### 3. Live Recognition

Launch real-time SASL recognition:

```bash
python sasl_cnn_only_recognition.py
```

**Recognition Features:**
- **Combined Model Support**: Uses both CNN and hand landmark data when available
- **Automatic Fallback**: Falls back to CNN-only for older models
- **Hand Overlay Toggle**: Press 'H' to show/hide hand landmarks visualization
- **Confidence Filtering**: Adjustable confidence thresholds
- **Top-3 Predictions**: Multiple prediction display with confidence scores
- **Prediction Smoothing**: Temporal smoothing for stable results

**Controls:**
- **'Q'**: Quit recognition
- **'R'**: Reset prediction buffer
- **'H'**: Toggle hand landmarks overlay (when available)

## Technical Architecture

### Model Architecture

#### CNN+LSTM Branch
- **Backbone**: EfficientNet-B0 (pre-trained)
- **Temporal Processing**: 1D convolution + batch normalization
- **LSTM Layers**: 2x bidirectional LSTM (256, 128 hidden units)
- **Classification**: Multi-layer perceptron with dropout

#### Hand Landmark Branch
- **Input**: 126 features (2 hands × 21 landmarks × 3 coordinates)
- **Processing**: Linear projection to 256 dimensions
- **LSTM Layers**: 2x bidirectional LSTM (128, 64 hidden units)
- **Classification**: Multi-layer perceptron with dropout

#### Fusion Architecture
- **Feature Combination**: Concatenation of CNN and hand predictions
- **Fusion Network**: 3-layer MLP (256 → 128 → num_classes)
- **Learnable Weights**: Adaptive weighting (default: 0.7 CNN, 0.3 hand)
- **Multi-loss Training**: Combined loss + auxiliary losses for each branch

### Data Processing Pipeline

#### Video Processing
1. **Frame Extraction**: Extract frames from video files
2. **Resize**: Standardize to 224×224 pixels
3. **Sequence Creation**: Create 30-frame sequences
4. **Normalization**: Scale pixel values to [0,1]

#### Hand Landmark Processing
1. **MediaPipe Detection**: Extract 21 landmarks per hand
2. **Coordinate Normalization**: Normalize to frame dimensions
3. **Feature Vector**: Create 126-dimensional feature vector
4. **Sequence Alignment**: Align with video frame sequences

### Performance Metrics

#### Model Performance
- **Accuracy**: 85-95% on well-trained classes
- **Processing Speed**: 30+ FPS real-time recognition
- **Memory Usage**: 2-4GB GPU memory during training
- **Training Time**: 30-60 minutes for 50 classes (with GPU)

#### Dataset Recommendations
- **Videos per Class**: Minimum 50, optimal 100+ videos
- **Video Duration**: 2-5 seconds per video
- **Video Quality**: 720p minimum, good lighting
- **Background**: Clean, contrasting background
- **Signer Position**: Full upper body visible

## Troubleshooting

### Common Issues & Solutions

#### GPU/CUDA Issues
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**Solutions:**
- Install appropriate PyTorch CUDA version
- Update GPU drivers
- Verify CUDA toolkit installation

#### Memory Issues
- Reduce batch size in training configuration
- Use CPU training for smaller datasets
- Close other GPU-intensive applications

#### Poor Recognition Performance
- Ensure good lighting conditions
- Use clean, contrasting backgrounds
- Collect more training videos per class
- Verify hand landmarks are being detected

#### Import/Dependency Errors
```bash
# Reinstall core dependencies
pip uninstall torch torchvision torchaudio timm mediapipe opencv-python
pip install -r requirements.txt
```

### Performance Optimization

#### Training Optimization
- Use GPU acceleration when available
- Implement data augmentation for small datasets
- Monitor validation loss to prevent overfitting
- Adjust learning rate based on training progress

#### Recognition Optimization
- Enable hand landmark overlay to verify detection quality
- Adjust confidence thresholds based on use case
- Use prediction smoothing for stable results
- Optimize camera positioning and lighting

## File Structure

```
SASL-AI/
├── sasl_video_collector.py          # Video data collection
├── video_cnn_only_training.py       # CNN+Hand landmark training
├── sasl_cnn_only_recognition.py     # Real-time recognition
├── demo_hand_overlay.py             # Hand overlay demonstration
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
├── HAND_OVERLAY_FEATURES.md         # Hand overlay feature guide
├── video_dataset/                   # Training videos
├── outputs/                         # Training outputs
│   ├── training_*/                  # Training session outputs
│   └── video_cache/                 # Processed video cache
└── __pycache__/                     # Python cache files
```

---

<div align="center">

**Advanced SASL Recognition System**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

*Empowering communication through AI-powered sign language recognition*

</div>