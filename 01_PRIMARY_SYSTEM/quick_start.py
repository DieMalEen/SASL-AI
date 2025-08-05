#!/usr/bin/env python3
"""
SASL Quick Start Script
This script helps you get started with the SASL hand detection system quickly.
"""

import os
import sys
import subprocess
import json

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'torchvision', 'cv2', 'PIL', 'numpy', 
        'sklearn', 'mediapipe'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    if not packages:
        return True
    
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    
    # Map package names for pip
    pip_names = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn'
    }
    
    pip_packages = [pip_names.get(pkg, pkg) for pkg in packages]
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'
        ] + pip_packages)
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install packages. Please install manually:")
        print(f"pip install {' '.join(pip_packages)}")
        return False

def check_dataset():
    """Check if dataset exists and is properly structured"""
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset folder not found: {dataset_path}")
        return False
    
    class_folders = [f for f in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, f))]
    
    if not class_folders:
        print(f"✗ No class folders found in {dataset_path}")
        return False
    
    video_count = 0
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        videos = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
        video_count += len(videos)
    
    print(f"✓ Dataset found: {len(class_folders)} classes, {video_count} videos")
    return True

def check_models():
    """Check which models are available"""
    models = {
        'sasl_model.pth': 'Standard CNN-LSTM model',
        'hand_focused_sasl_model.pth': 'Hand-focused CNN-LSTM model',
        'best_hand_focused_sasl_model.pth': 'Best hand-focused model'
    }
    
    available_models = []
    for model_file, description in models.items():
        if os.path.exists(model_file):
            print(f"✓ {description}: {model_file}")
            available_models.append(model_file)
        else:
            print(f"✗ {description}: {model_file}")
    
    return available_models

def main_menu():
    """Display main menu and handle user choice"""
    print("\n" + "="*60)
    print("SASL Hand Detection System - Quick Start")
    print("="*60)
    
    options = [
        ("1", "Check System Requirements", check_system),
        ("2", "Train Hand-Focused Model (RECOMMENDED)", train_hand_focused),
        ("3", "Run Real-Time Recognition", run_camera),
        ("4", "Demo Hand Tracking on Videos", run_demo),
        ("5", "View Documentation", show_docs),
        ("q", "Quit", quit_program)
    ]
    
    for key, description, _ in options:
        print(f"{key}. {description}")
    
    print("\nChoose an option:", end=" ")
    choice = input().strip().lower()
    
    # Find and execute the chosen option
    for key, _, func in options:
        if choice == key.lower():
            func()
            return
    
    print("Invalid choice. Please try again.")
    main_menu()

def check_system():
    """Check system requirements"""
    print("\n" + "-"*40)
    print("CHECKING SYSTEM REQUIREMENTS")
    print("-"*40)
    
    print("\nChecking Python packages...")
    missing = check_dependencies()
    
    if missing:
        install = input(f"\nInstall missing packages? (y/n): ").strip().lower()
        if install == 'y':
            if install_dependencies(missing):
                print("\n✓ All dependencies installed!")
            else:
                print("\n✗ Installation failed. Please install manually.")
                return
        else:
            print("Please install missing packages manually before proceeding.")
            return
    
    print("\nChecking dataset...")
    check_dataset()
    
    print("\nChecking trained models...")
    available_models = check_models()
    
    if not available_models:
        print("\n⚠️  No trained models found. You'll need to train a model first.")
    
    print(f"\n✓ System check complete!")
    input("Press Enter to continue...")
    main_menu()

def train_standard():
    """Legacy function - redirects to hand-focused training"""
    print("\n" + "-"*40)
    print("REDIRECTING TO ENHANCED TRAINING")
    print("-"*40)
    print("The standard model has been integrated into the hand-focused model.")
    print("The hand-focused model can train with or without hand detection.")
    print("Redirecting to enhanced training...")
    train_hand_focused()

def train_hand_focused():
    """Train the hand-focused CNN-LSTM model"""
    print("\n" + "-"*40)
    print("TRAINING HAND-FOCUSED CNN-LSTM MODEL")
    print("-"*40)
    
    if not check_dataset():
        input("Please fix dataset issues first. Press Enter to continue...")
        main_menu()
        return
    
    print("This will train an enhanced model with hand detection.")
    print("Training may take longer but provides better accuracy.")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        main_menu()
        return
    
    print("Starting enhanced CNN+LSTM training...")
    try:
        subprocess.run([sys.executable, "hand_focused_CNN_LSTM.py"], check=True)
        print("✓ Enhanced CNN+LSTM training completed successfully!")
    except subprocess.CalledProcessError:
        print("✗ Training failed. Check the error messages above.")
    except FileNotFoundError:
        print("✗ hand_focused_CNN_LSTM.py not found.")
    
    input("Press Enter to continue...")
    main_menu()

def run_camera():
    """Run real-time recognition"""
    print("\n" + "-"*40)
    print("REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("-"*40)
    
    available_models = check_models()
    if not available_models:
        print("✗ No trained models found. Please train a model first.")
        input("Press Enter to continue...")
        main_menu()
        return
    
    print("Available recognition modes:")
    print("1. Enhanced Camera with Hand Tracking (RECOMMENDED)")
    print("2. Standard Camera")
    
    choice = input("Choose mode (1/2): ").strip()
    
    if choice == "1":
        print("Starting enhanced camera with hand tracking...")
        print("Controls: 'q'=quit, 'h'=toggle hand overlay, 'r'=reset")
        try:
            subprocess.run([sys.executable, "enhanced_camera.py"], check=True)
        except subprocess.CalledProcessError:
            print("✗ Enhanced camera failed.")
        except FileNotFoundError:
            print("✗ enhanced_camera.py not found.")
    
    elif choice == "2":
        print("Starting standard camera...")
        try:
            subprocess.run([sys.executable, "camera.py"], check=True)
        except subprocess.CalledProcessError:
            print("✗ Camera failed.")
        except FileNotFoundError:
            print("✗ camera.py not found.")
    
    else:
        print("Invalid choice.")
    
    input("Press Enter to continue...")
    main_menu()

def run_demo():
    """Run hand tracking demo"""
    print("\n" + "-"*40)
    print("HAND TRACKING DEMONSTRATION")
    print("-"*40)
    
    if not check_dataset():
        input("Please fix dataset issues first. Press Enter to continue...")
        main_menu()
        return
    
    print("This will demonstrate hand tracking on your videos.")
    print("Starting demo...")
    
    try:
        subprocess.run([sys.executable, "hand_tracking_demo.py"], check=True)
        print("✓ Demo completed!")
    except subprocess.CalledProcessError:
        print("✗ Demo failed. Check the error messages above.")
    except FileNotFoundError:
        print("✗ hand_tracking_demo.py not found.")
    
    input("Press Enter to continue...")
    main_menu()

def show_docs():
    """Show documentation"""
    print("\n" + "-"*40)
    print("DOCUMENTATION")
    print("-"*40)
    
    docs = [
        ("ReadMe", "Main documentation"),
        ("HAND_DETECTION_ENHANCEMENT.md", "Hand detection guide")
    ]
    
    for doc_file, description in docs:
        if os.path.exists(doc_file):
            print(f"✓ {description}: {doc_file}")
        else:
            print(f"✗ {description}: {doc_file}")
    
    print(f"\nKey files and their purposes:")
    print(f"• hand_detection.py - Core hand detection module")
    print(f"• hand_focused_model.py - Enhanced training script")
    print(f"• enhanced_camera.py - Real-time recognition with hand tracking")
    print(f"• hand_tracking_demo.py - Video analysis tool")
    
    print(f"\nQuick commands:")
    print(f"• Train model: python hand_focused_model.py")
    print(f"• Real-time recognition: python enhanced_camera.py")
    print(f"• Demo hand tracking: python hand_tracking_demo.py")
    
    input("Press Enter to continue...")
    main_menu()

def quit_program():
    """Exit the program"""
    print("\nThank you for using SASL Hand Detection System!")
    sys.exit(0)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        quit_program()
