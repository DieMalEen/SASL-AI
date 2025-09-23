#!/usr/bin/env python3
"""
SASL Model Status Checker
========================

Quick utility to check if trained PyTorch models are available for camera recognition.
Shows model locations, file sizes, and training information.
"""

import json
from pathlib import Path
from datetime import datetime

def format_size(bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def main():
    print("SASL Model Status Checker")
    print("=" * 40)
    
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("No outputs directory found")
        print("Run training first: python video_based_sasl_training.py")
        return
    
    # Find training sessions
    training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
    
    if not training_dirs:
        print("No training sessions found")
        print("Run training first: python video_based_sasl_training.py")
        return
    
    print(f"Found {len(training_dirs)} training session(s):")
    print()
    
    for i, training_dir in enumerate(sorted(training_dirs, key=lambda x: x.name), 1):
        print(f"Session {i}: {training_dir.name}")
        
        # Parse timestamp from directory name
        try:
            timestamp_str = training_dir.name.split("_", 1)[1]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            print(f"   Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"   Date: Unknown")
        
        # Check models
        models_dir = training_dir / "models"
        results_dir = training_dir / "results"
        
        if models_dir.exists():
            cnn_model = models_dir / "best_sasl_cnn_lstm_model.pth"
            pose_model = models_dir / "best_sasl_pose_lstm_model.pth"
            
            if cnn_model.exists():
                size = format_size(cnn_model.stat().st_size)
                print(f"   CNN+LSTM Model: {size}")
            else:
                print(f"   CNN+LSTM Model: Missing")
            
            if pose_model.exists():
                size = format_size(pose_model.stat().st_size)
                print(f"   Pose LSTM Model: {size}")
            else:
                print(f"   Pose LSTM Model: Missing")
        else:
            print(f"   Models directory: Missing")
        
        # Check results
        if results_dir.exists():
            class_names_file = results_dir / "class_names.json"
            results_file = results_dir / "comprehensive_training_results.json"
            
            if class_names_file.exists():
                try:
                    with open(class_names_file, 'r') as f:
                        classes = json.load(f)
                    print(f"   Classes ({len(classes)}): {', '.join(classes)}")
                except:
                    print(f"   Classes file: Corrupted")
            else:
                print(f"   Classes file: Missing")
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    if 'best_accuracy' in results:
                        acc = results['best_accuracy'] * 100
                        print(f"   Best Accuracy: {acc:.1f}%")
                    if 'training_time' in results:
                        print(f"   Training Time: {results['training_time']}")
                except:
                    print(f"   Results file: Could not read")
        else:
            print(f"   Results directory: Missing")
        
        print()
    
    # Show latest session status
    latest_session = sorted(training_dirs, key=lambda x: x.name)[-1]
    models_dir = latest_session / "models"
    results_dir = latest_session / "results"
    
    print("Camera Recognition Status:")
    
    # Check if all required files exist
    required_files = [
        (models_dir / "best_sasl_cnn_lstm_model.pth", "CNN+LSTM Model"),
        (models_dir / "best_sasl_pose_lstm_model.pth", "Pose LSTM Model"),
        (results_dir / "class_names.json", "Class Names")
    ]
    
    all_ready = True
    for file_path, name in required_files:
        if file_path.exists():
            print(f"   {name}: Ready")
        else:
            print(f"   {name}: Missing")
            all_ready = False
    
    print()
    if all_ready:
        print("Status: READY FOR CAMERA RECOGNITION")
        print("Run: python sasl_camera_recognition.py")
        print("Or:  python launch_camera.py")
    else:
        print("Status: NOT READY")
        print("Complete training first: python video_based_sasl_training.py")

if __name__ == "__main__":
    main()