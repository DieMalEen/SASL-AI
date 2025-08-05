# SASL Hand Detection System - User Guide

## Quick Start (Easiest Way)

### 1. Run the Interactive Guide
```bash
python quick_start.py
```
This interactive script will:
- Check system requirements
- Guide you through training
- Help you run real-time recognition
- Show you how to analyze videos

### 2. Or Use Direct Commands

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Train Enhanced Model (Recommended)
```bash
python hand_focused_model.py
```

#### Run Real-Time Recognition
```bash
python enhanced_camera.py
```

---

## 📋 Complete Program Guide

### Training Programs

#### 1. `hand_focused_CNN_LSTM.py` - Enhanced CNN+LSTM Model ⭐ **RECOMMENDED**
```bash
python hand_focused_CNN_LSTM.py
```
**What it does:**
- Uses MediaPipe hand detection during training
- Focuses on hand regions for better accuracy
- Uses bidirectional LSTM with attention mechanism
- Creates `hand_focused_sasl_model.pth` and `best_hand_focused_sasl_model.pth`
- **Expected 5-10% accuracy improvement**
- **Includes fallback support for standard CNN-LSTM training**

### Real-Time Recognition Programs

#### 1. `enhanced_camera.py` - Advanced Recognition ⭐ **RECOMMENDED**
```bash
python enhanced_camera.py
```
**Features:**
- Real-time hand detection and tracking
- Live gesture prediction with confidence scores
- Visual hand landmark overlay
- Performance monitoring (FPS)

**Controls:**
- `q` - Quit
- `h` - Toggle hand detection overlay
- `r` - Reset gesture buffer

#### 2. `camera.py` - Basic Recognition
```bash
python camera.py
```
**Features:**
- Basic real-time gesture recognition
- No hand tracking visualization
- Uses standard model only

### Analysis and Demo Programs

#### 1. `hand_tracking_demo.py` - Video Analysis ⭐ **NEW**
```bash
python hand_tracking_demo.py
```
**Options:**
1. **Show hand tracking video** - Live playback with overlays
2. **Save hand tracking video** - Export annotated video
3. **Extract hand regions** - Save hand images for analysis
4. **All of the above** - Complete analysis

**Output Files:**
- `hand_tracking_demo_[filename].mp4` - Annotated video
- `hand_regions_demo/` - Extracted hand images

#### 2. `quick_start.py` - Interactive Guide ⭐ **NEW**
```bash
python quick_start.py
```
**Features:**
- System requirement checker
- Guided training process
- Easy access to all programs
- Documentation viewer

---

## 🔧 Program Parameters and Configuration

### Hand Detection Settings
In `hand_detection.py`, you can adjust:
```python
HandDetector(
    static_image_mode=False,        # False for video, True for images
    max_num_hands=2,               # Maximum hands to detect
    min_detection_confidence=0.7,  # Detection threshold (0.5-1.0)
    min_tracking_confidence=0.5    # Tracking stability (0.5-1.0)
)
```

### Training Configuration
In `hand_focused_CNN_LSTM.py`, you can modify:
```python
ENABLE_VALIDATION = True           # Enable validation split
USE_HAND_DETECTION = True         # Toggle hand detection
num_epochs = 15                   # Training epochs
batch_size = 2                    # Batch size (reduce if GPU memory issues)
```

### Camera Settings
In `enhanced_camera.py`, you can adjust:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Camera width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Camera height
cap.set(cv2.CAP_PROP_FPS, 30)            # Camera FPS

buffer_size=16                     # Frames to collect before prediction
stability_threshold=5              # Frames for stable prediction
```

---

## 📊 Expected Performance

### Training Performance
- **Standard Model**: ~10-15 minutes on GPU, ~30-45 minutes on CPU
- **Hand-Focused Model**: ~15-25 minutes on GPU, ~45-75 minutes on CPU

### Real-Time Performance
- **Hand Detection**: 25+ FPS on standard hardware
- **Full Pipeline**: 15-30 FPS depending on hardware
- **Accuracy**: 5-10% improvement with hand detection

### Hardware Requirements
- **Minimum**: CPU-only, 8GB RAM
- **Recommended**: GPU with 4GB+ VRAM, 16GB RAM
- **Camera**: Any USB webcam

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. MediaPipe Installation Issues
```bash
# If installation fails
pip install --upgrade pip
pip install mediapipe --no-cache-dir

# Alternative installation
conda install mediapipe -c conda-forge
```

#### 2. Camera Not Working
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# Try different camera indices
# Change cv2.VideoCapture(0) to cv2.VideoCapture(1) or cv2.VideoCapture(2)
```

#### 3. GPU/CUDA Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Force CPU usage if needed
# Add this to your script: device = torch.device("cpu")
```

#### 4. Model Loading Errors
- Ensure you've trained a model first
- Check that `class_names.json` exists
- Verify model files exist in the correct directory

#### 5. Low FPS During Real-Time Recognition
- Reduce camera resolution
- Use CPU-only mode if GPU is overloaded
- Reduce batch size in training
- Close other applications

---

## 🎯 Best Practices

### For Training
1. **Use hand-focused model** for better accuracy
2. **Ensure good lighting** in your training videos
3. **Include diverse hand positions** in your dataset
4. **Train for enough epochs** (15+ recommended)
5. **Monitor validation accuracy** to avoid overfitting

### For Recognition
1. **Use enhanced camera** for best experience
2. **Ensure good lighting** during recognition
3. **Keep hands visible** in camera frame
4. **Allow buffer to fill** before expecting predictions
5. **Use stable hand positions** for better recognition

### For Best Results
1. **Consistent hand positions** between training and testing
2. **Good contrast** between hands and background
3. **Steady hand movements** during recognition
4. **Regular retraining** with new data

---

## 📚 Additional Resources

### Documentation Files
- `ReadMe` - Complete project documentation
- `HAND_DETECTION_ENHANCEMENT.md` - Detailed hand detection guide
- `requirements.txt` - Dependency list

### Generated Files
- `class_names.json` - Class name mappings
- `*.pth` - Trained model weights
- `hand_tracking_demo_*.mp4` - Demo videos
- `hand_regions_demo/` - Extracted hand images

### Key Code Modules
- `hand_focused_CNN_LSTM.py` - **Primary CNN+LSTM training script with hand detection**
- `enhanced_camera.py` - **Primary real-time recognition with hand tracking**
- `hand_detection.py` - Hand detection core functionality
- `model.py` - Basic model definitions (CNN backbone)
- `camera.py` - Basic real-time recognition (fallback)
- `quick_start.py` - Interactive setup and usage guide
- `hand_tracking_demo.py` - Video analysis and demonstration tool

**Note**: Redundant files (`cnn_lstm_model.py`, `camera_threaded.py`) have been removed. All functionality is now consolidated into the enhanced hand detection system.

Remember: The hand-focused approach significantly improves accuracy by focusing on the most relevant part of the image - the hands performing the sign language gestures!
