# SASL-AI Recognition System (PyTorch Edition)

<div align="center">

![SASL-AI Banner](https://img.shields.io/badge/SASL--AI-PyTorch-blue?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Advanced South African Sign Language Recognition using PyTorch Deep Learning**

Real-time SASL recognition with CNN+LSTM and Pose LSTM models, comprehensive batch monitoring, and professional deployment capabilities.

</div>

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd SASL-AI
pip install -r requirements.txt

# Run the system
python main.py
```

## Features

### Core Capabilities
- **Real-time SASL Recognition**: Live camera-based sign language detection
- **Dual Model Architecture**: CNN+LSTM and Pose LSTM ensemble for robust predictions
- **PyTorch Backend**: Modern deep learning framework with GPU acceleration
- **Comprehensive Training**: Advanced batch monitoring with real-time progress tracking
- **Professional UI**: Clean interfaces with confidence scores and visual feedback

### Technical Features
- **Batch Visibility**: Real-time training progress with tqdm progress bars
- **GPU Optimization**: Automatic CUDA detection and memory management
- **Data Augmentation**: Advanced augmentation pipeline for robust model training
- **Transfer Learning**: EfficientNet backbone via timm library
- **Pose Integration**: MediaPipe pose and hand landmark processing
- **Model Ensembling**: Weighted prediction combination for improved accuracy

## Installation

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows 10/11, macOS, or Linux
- **GPU**: CUDA-compatible GPU (optional, but recommended)
- **RAM**: 8GB minimum, 16GB recommended
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
mediapipe>=0.10.0
Pillow>=10.0.0

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

### 1. Main Menu System

Launch the interactive menu system:

```bash
python main.py
```

**Menu Options:**
1. **Video Data Collection**: Collect training videos for custom signs
2. **PyTorch Model Training**: Train CNN+LSTM and Pose LSTM models
3. **Live Camera Recognition**: Real-time SASL recognition
4. **Model Management**: View outputs and manage trained models

### 2. Dataset Preparation

```
dataset/
├── sign1/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
├── sign2/
│   ├── video001.mp4
│   └── ...
└── ...
```

**Dataset Guidelines:**
- **Video Length**: 2-5 seconds per video
- **Resolution**: Minimum 480p, 720p recommended
- **Lighting**: Good, consistent lighting
- **Background**: Clean, contrasting background
- **Signer Position**: Full upper body visible
- **Examples per Class**: Minimum 20 videos, 50+ recommended

## Training System

### Advanced PyTorch Training Pipeline

The PyTorch training system provides comprehensive batch monitoring and professional-grade model training.

#### Key Features:
- **Real-time Progress Tracking**: tqdm progress bars with loss/accuracy
- **Dual Model Training**: CNN+LSTM and Pose LSTM models
- **Advanced Data Augmentation**: Brightness, contrast, rotation, noise
- **GPU Acceleration**: Automatic CUDA optimization
- **Early Stopping**: Prevent overfitting with patience mechanism
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

#### Training Output Example:

```
Starting PyTorch SASL Training
Device: cuda:0
Classes: 50
Sequence length: 30

Training Progress:
Epoch 1/100: 100%|██████████| 125/125 [02:15<00:00, 0.92it/s, loss=2.34, acc=45.2%]
Epoch 2/100: 100%|██████████| 125/125 [02:12<00:00, 0.94it/s, loss=1.89, acc=58.7%]
...

Training completed successfully!
Best CNN+LSTM model saved: best_sasl_cnn_lstm_model.pth
Best Pose LSTM model saved: best_sasl_pose_lstm_model.pth
```

## Live Recognition

### Real-time Camera Recognition

Launch live recognition:
```bash
python sasl_camera_recognition.py
```

#### Features:
- **Dual Model Ensemble**: CNN+LSTM + Pose LSTM predictions
- **Confidence Filtering**: Adjustable confidence thresholds
- **Top-3 Predictions**: Multiple prediction display
- **Visual Feedback**: MediaPipe landmark overlay
- **Prediction Smoothing**: Temporal smoothing for stable results

#### Controls:
- **'q'**: Quit recognition
- **'r'**: Reset prediction buffers

#### Performance Metrics:
- **Accuracy**: 85-95% on well-lit, clear signs
- **Speed**: 30 FPS real-time processing
- **Latency**: <100ms prediction time
- **Memory Usage**: ~2-4GB GPU/CPU memory

## Troubleshooting

### Common Issues & Solutions

#### 1. GPU/CUDA Issues
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**Solutions:**
- Install appropriate PyTorch CUDA version
- Update GPU drivers
- Check CUDA toolkit installation

#### 2. Memory Issues
```bash
# Reduce batch sizes in training
CNN_BATCH_SIZE = 2  # Instead of 4
POSE_BATCH_SIZE = 4  # Instead of 8
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip uninstall torch torchvision torchaudio timm
pip install torch torchvision torchaudio timm
```

#### 4. Camera Recognition Issues
- **Poor Recognition**: Improve lighting, ensure clear background
- **Slow Performance**: Reduce sequence length, use GPU
- **No Detection**: Check camera permissions, MediaPipe setup

---

<div align="center">

**Made with love for the SASL Community**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

*Empowering communication through AI*

</div>