#!/usr/bin/env python3
"""
SASL-AI Main Menu System
========================

Complete SASL recognition system with:
1. Video data collection
2. CNN-only training (recommended) and dual model training (legacy)
3. CNN-only live recognition (recommended) and dual model recognition (legacy)
4. Model management and outputs

Navigate through the menu to access all features.
"""

import os
import sys
from pathlib import Path
import json
import subprocess

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the main banner"""
    print("=" * 70)
    print("                      SASL-AI RECOGNITION SYSTEM")
    print("                 South African Sign Language AI")
    print("=" * 70)
    print("        Video-based Transfer Learning & Real-time Recognition")
    print("=" * 70)

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = ['cv2', 'torch', 'mediapipe', 'numpy', 'sklearn', 'timm']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"!!! Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using:")
        print("pip install torch torchvision torchaudio opencv-python mediapipe numpy scikit-learn timm")
        return False
    return True

def get_dataset_stats():
    """Get statistics about the video dataset"""
    video_dataset_path = Path("video_dataset")
    
    if not video_dataset_path.exists():
        return {"classes": 0, "total_videos": 0}
    
    classes = [d for d in video_dataset_path.iterdir() if d.is_dir() and d.name != '.gitkeep']
    total_videos = 0
    
    for class_dir in classes:
        videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        total_videos += len(videos)
    
    return {"classes": len(classes), "total_videos": total_videos}

def get_trained_models_stats():
    """Get statistics about trained models"""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        return {"trained_models": 0, "last_training": "Never"}
    
    # Count training directories
    training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
    
    if not training_dirs:
        return {"trained_models": 0, "last_training": "Never"}
    
    # Get latest training timestamp
    latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
    timestamp_str = latest_training_dir.name.replace("training_", "")
    
    try:
        from datetime import datetime
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        last_training = timestamp.strftime("%Y-%m-%d %H:%M")
    except:
        last_training = timestamp_str
    
    return {"trained_models": len(training_dirs), "last_training": last_training}

def show_main_menu():
    """Display the main menu"""
    clear_screen()
    print_banner()
    
    # Get current stats
    dataset_stats = get_dataset_stats()
    model_stats = get_trained_models_stats()
    
    print(f"\n System Status:")
    print(f"   Video Classes: {dataset_stats['classes']}")
    print(f"   Total Videos: {dataset_stats['total_videos']}")
    print(f"   Trained Models: {model_stats['trained_models']}")
    print(f"   Last Training: {model_stats['last_training']}")
    
    print(f"\n Main Menu:")
    print(f"   1.  Collect Video Data")
    print(f"   2.  Train CNN-Only Models")
    print(f"   3.  Train Dual Models")
    print(f"   4.  CNN-Only Live Recognition")
    print(f"   5.  Dual Model Live Recognition")
    print(f"   6.  Batch Video Prediction")
    print(f"   7.  Manage Models & Outputs")
    print(f"   8.  System Information")
    print(f"   9.  Exit")
    print(f"\n" + "=" * 70)

def configure_cnn_training_parameters():
    """Configure CNN-only training parameters interactively"""
    clear_screen()
    print("CNN-Only Training Configuration")
    print("=" * 50)
    
    print("Configure your CNN-only training parameters:")
    print("(Press Enter for recommended values)\n")
    
    # Epochs configuration
    while True:
        try:
            epochs_input = input("Number of training epochs [recommended: 50]: ").strip()
            epochs = 50 if epochs_input == "" else int(epochs_input)
            if epochs > 0:
                break
            else:
                print("Epochs must be greater than 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Batch size configuration  
    while True:
        try:
            batch_input = input("Batch size [recommended: 8]: ").strip()
            batch_size = 8 if batch_input == "" else int(batch_input)
            if batch_size > 0:
                break
            else:
                print("Batch size must be greater than 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Data augmentation configuration
    print("\nData Augmentation Factor:")
    print("0 = No augmentation (fastest)")
    print("1 = 1x augmentation (doubles dataset, recommended)")
    print("2 = 2x augmentation (triples dataset)")
    print("3 = 3x augmentation (quadruples dataset)")
    print("Note: Higher augmentation improves robustness but increases training time")
    
    while True:
        try:
            aug_input = input("Augmentation factor (0-3) [recommended: 1]: ").strip()
            augmentation_factor = 1 if aug_input == "" else int(aug_input)
            if 0 <= augmentation_factor <= 3:
                break
            else:
                print("Augmentation factor must be between 0 and 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Learning rate configuration
    while True:
        try:
            lr_input = input("Learning rate [recommended: 0.001]: ").strip()
            learning_rate = 0.001 if lr_input == "" else float(lr_input)
            if learning_rate > 0:
                break
            else:
                print("Learning rate must be greater than 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Summary
    estimated_time = epochs * (1 + augmentation_factor) * 0.3  # Rough estimate
    print(f"\nTraining Configuration Summary:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Augmentation Factor: {augmentation_factor}x")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Estimated Time: {estimated_time:.0f}-{estimated_time*2:.0f} minutes")
    
    confirm = input(f"\nProceed with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        return None
    
    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'augmentation_factor': augmentation_factor,
        'learning_rate': learning_rate
    }

def collect_video_data():
    """Launch video data collection"""
    clear_screen()
    print("SASL Video Data Collection")
    print("=" * 50)
    print("This will launch the interactive video collector.")
    print("Press SPACE to start recording, SPACE again to stop and save!")
    print("Quality indicator shows when you're ready to record.")
    print("\nPress Enter to continue or 'q' to return to menu...")
    
    choice = input().strip().lower()
    if choice == 'q':
        return
    
    try:
        # Import and run video collector
        from sasl_video_collector import SASLVideoCollector
        
        collector = SASLVideoCollector(
            dataset_path="video_dataset",
            recording_duration=3.0,
            fps=30
        )
        collector.interactive_menu()
        
    except ImportError as e:
        print(f"ERROR: Could not import video collector: {e}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"ERROR: Error during video collection: {e}")
        input("Press Enter to continue...")

def train_cnn_only_models():
    """Launch CNN-only model training"""
    clear_screen()
    print("SASL CNN-Only Model Training")
    print("=" * 50)
    print("This trains only the CNN+LSTM model for better performance.")
    print("No pose dependency - more robust and faster!")
    
    # Check if video dataset exists
    video_dataset_path = Path("video_dataset")
    has_classes = video_dataset_path.exists() and any(d.is_dir() and d.name != '.gitkeep' for d in video_dataset_path.iterdir())
    if not has_classes:
        print("No video dataset found!")
        print("Please collect video data first using option 1.")
        input("Press Enter to continue...")
        return
    
    # Show dataset info
    stats = get_dataset_stats()
    print(f"\nDataset Information:")
    print(f"   Classes: {stats['classes']}")
    print(f"   Total Videos: {stats['total_videos']}")
    print(f"   Average per class: {stats['total_videos'] / max(stats['classes'], 1):.1f}")
    
    if stats['classes'] < 2:
        print("Need at least 2 classes to train models!")
        input("Press Enter to continue...")
        return
    
    if stats['total_videos'] < 10:
        print("!!! Warning: Very small dataset. Consider collecting more videos.")
        print("Recommended: At least 5-10 videos per class")
    
    print(f"\nTraining Configuration:")
    print(f"  1. Quick training (default settings)")
    print(f"  2. Custom configuration (configure parameters)")
    print(f"  3. Back to main menu")
    
    choice = input(f"\nEnter choice (1-3): ").strip()
    
    if choice == '3':
        return
    elif choice == '2':
        # Custom configuration
        config = configure_cnn_training_parameters()
        if config is None:
            return
    else:
        # Default configuration
        config = {
            'epochs': 50,
            'batch_size': 8,
            'augmentation_factor': 1,
            'learning_rate': 0.001
        }
        print(f"\nUsing default settings:")
        print(f"  - {config['epochs']} epochs")
        print(f"  - Batch size {config['batch_size']}")
        print(f"  - {config['augmentation_factor']}x augmentation")
        print(f"  - Learning rate {config['learning_rate']}")
        
        confirm = input(f"\nStart training with default settings? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    
    try:
        # Import and run CNN-only training
        from video_cnn_only_training import CNNOnlyVideoSASLTrainer
        
        print("\nInitializing CNN-only trainer with custom configuration...")
        trainer = CNNOnlyVideoSASLTrainer(
            video_dataset_path="video_dataset",
            sequence_length=30,
            input_size=(224, 224),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            augmentation_factor=config['augmentation_factor'],
            learning_rate=config['learning_rate']
        )
        
        print(f"Starting CNN-only training process...")
        print(f"  Configuration: {config['epochs']} epochs, batch size {config['batch_size']}, {config['augmentation_factor']}x augmentation")
        model = trainer.train_model()
        
        if model is not None:
            print(f"\nCNN-only training completed successfully!")
            print(f"  Configuration used: {config['epochs']} epochs, batch {config['batch_size']}, aug {config['augmentation_factor']}x")
            print(f"  Models saved in: outputs/training_[timestamp]/models/")
            print(f"  Results saved in: outputs/training_[timestamp]/results/")
            print(f"  Use option 4 for CNN-only live recognition!")
        else:
            print(f"\n✗ Training failed. Check your dataset.")
        
        input("Press Enter to continue...")
        
    except ImportError as e:
        print(f"ERROR: Could not import CNN-only trainer: {e}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"ERROR: Error during CNN-only training: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to continue...")

def train_dual_models():
    """Launch dual model training"""
    clear_screen()
    print("SASL Dual Model Training")
    print("=" * 50)
    print("WARNING: This trains both CNN+LSTM and Pose models.")
    print("The CNN-only approach (option 2) is recommended for better performance.")
    
    # Check if video dataset exists
    video_dataset_path = Path("video_dataset")
    has_classes = video_dataset_path.exists() and any(d.is_dir() and d.name != '.gitkeep' for d in video_dataset_path.iterdir())
    if not has_classes:
        print("No video dataset found!")
        print("Please collect video data first using option 1.")
        input("Press Enter to continue...")
        return
    
    # Show dataset info
    stats = get_dataset_stats()
    print(f"\nDataset Information:")
    print(f"   Classes: {stats['classes']}")
    print(f"   Total Videos: {stats['total_videos']}")
    print(f"   Average per class: {stats['total_videos'] / max(stats['classes'], 1):.1f}")
    
    if stats['classes'] < 2:
        print("Need at least 2 classes to train models!")
        input("Press Enter to continue...")
        return
    
    if stats['total_videos'] < 10:
        print("!!! Warning: Very small dataset. Consider collecting more videos.")
        print("Recommended: At least 5-10 videos per class")
    
    print(f"\nThis will train the legacy dual model system.")
    print(f"Note: CNN-only training (option 2) typically achieves better results.")
    
    choice = input(f"\nProceed with dual model training? (y/n): ").strip().lower()
    
    if choice != 'y':
        return
    
    try:
        # Import and run dual model training
        from video_based_sasl_training import VideoSASLTrainer
        
        print("\nInitializing dual model trainer...")
        trainer = VideoSASLTrainer(
            video_dataset_path="video_dataset",
            sequence_length=30,
            input_size=(224, 224),
            epochs=50,
            batch_size_cnn=4,
            batch_size_pose=8
        )
        
        print("Starting dual model training process...")
        cnn_lstm_model, pose_lstm_model = trainer.train_models()
        
        if cnn_lstm_model is not None:
            print(f"\nDual model training completed!")
            print(f"  Note: Consider using CNN-only training for better performance")
        else:
            print(f"\n✗ Training failed. Check your dataset.")
        
        input("Press Enter to continue...")
        
    except ImportError as e:
        print(f"ERROR: Could not import dual model trainer: {e}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"ERROR: Error during dual model training: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to continue...")

def live_cnn_only_recognition():
    """Launch CNN-only live camera recognition"""
    clear_screen()
    print("CNN-Only Live SASL Recognition")
    print("=" * 50)
    print("Fast, accurate recognition using only CNN+LSTM model!")
    
    # Check for the latest training outputs
    outputs_dir = Path("outputs")
    model_files = []
    
    if outputs_dir.exists():
        # Find the latest training directory
        training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
        if training_dirs:
            # Sort by name (timestamp) and get the latest
            latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            print(f"Found latest training session: {latest_training_dir.name}")
            
            # Check for models in the latest training directory
            models_dir = latest_training_dir / "models"
            results_dir = latest_training_dir / "results"
            
            if models_dir.exists() and results_dir.exists():
                cnn_model_path = models_dir / "best_sasl_cnn_lstm_model.pth"
                
                # Check for class names
                classes_path = results_dir / "class_names.json"
                if not classes_path.exists():
                    classes_path = results_dir / "pytorch_sasl_classes.json"
                
                if cnn_model_path.exists() and classes_path.exists():
                    model_files = [str(cnn_model_path), str(classes_path)]
                    print(f"Found CNN+LSTM model: {cnn_model_path}")
                    print(f"Found class names: {classes_path}")
                else:
                    print(f"Missing CNN-only model files in {models_dir}")
            else:
                print(f"Models or results directory not found in {latest_training_dir}")
        else:
            print("No training directories found in outputs/")
    
    # Fallback: Check for models in root directory or outputs/
    if not model_files:
        print("Checking for models in fallback locations...")
        fallback_files = [
            "best_sasl_cnn_lstm_model.pth",
            "class_names.json"
        ]
        
        # Check outputs directory first, then root
        for model_file in fallback_files:
            if (outputs_dir / model_file).exists():
                model_files.append(str(outputs_dir / model_file))
            elif Path(model_file).exists():
                model_files.append(model_file)
            else:
                model_files.append(None)
        
        # Check if all files found
        if None in model_files:
            missing_files = [f for f, path in zip(fallback_files, model_files) if path is None]
            print(f"ERROR: Missing CNN-only model files: {missing_files}")
            print("\nTo fix this issue:")
            print("1. Run CNN-only training first using option 2")
            print("2. Or collect training data using option 1")
            print("3. Make sure training completes successfully")
            input("Press Enter to continue...")
            return
        else:
            print("Found models in fallback locations")
    
    if not model_files or len(model_files) != 2:
        print("ERROR: Could not locate all required CNN-only model files")
        print("\nTo fix this issue:")
        print("1. Run CNN-only training first using option 2")
        print("2. Or collect training data using option 1") 
        print("3. Make sure training completes successfully")
        input("Press Enter to continue...")
        return
    
    try:
        with open(model_files[1], 'r') as f:
            classes = json.load(f)
        
        print(f"\nFOUND: Trained CNN-only model for {len(classes)} SASL classes:")
        for i, class_name in enumerate(classes, 1):
            print(f"   {i:2d}. {class_name}")
        
        print(f"\n Starting CNN-only live camera recognition...")
        print(f" Expected accuracy: 90-100%")
        print(f" Processing: Fast (CNN-only)")
        
        # Import and run CNN-only recognition
        from sasl_cnn_only_recognition import SASLCNNOnlyCameraRecognition
        
        recognition = SASLCNNOnlyCameraRecognition(
            model_path=model_files[0],
            classes_path=model_files[1]
        )
        
        recognition.run_live_recognition()
        
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        input("Press Enter to continue...")
    except ImportError as e:
        print(f"ERROR: Could not import CNN-only recognition: {e}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"ERROR: Error during CNN-only recognition: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to continue...")

def live_dual_model_recognition():
    """Launch dual model live camera recognition"""
    clear_screen()
    print("Dual Model Live SASL Recognition")
    print("=" * 50)
    print("WARNING: This uses both CNN+LSTM and Pose models.")
    print("The CNN-only approach (option 4) is recommended for better performance.")
    
    # Check for the latest training outputs
    outputs_dir = Path("outputs")
    model_files = []
    
    if outputs_dir.exists():
        # Find the latest training directory
        training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
        if training_dirs:
            # Sort by name (timestamp) and get the latest
            latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            print(f"Found latest training session: {latest_training_dir.name}")
            
            # Check for models in the latest training directory
            models_dir = latest_training_dir / "models"
            results_dir = latest_training_dir / "results"
            
            if models_dir.exists() and results_dir.exists():
                cnn_model_path = models_dir / "best_sasl_cnn_lstm_model.pth"
                pose_model_path = models_dir / "best_sasl_pose_lstm_model.pth"
                
                # Check for class names - try both pytorch_sasl_classes.json and class_names.json
                classes_path = results_dir / "pytorch_sasl_classes.json"
                if not classes_path.exists():
                    classes_path = results_dir / "class_names.json"
                
                if cnn_model_path.exists() and pose_model_path.exists() and classes_path.exists():
                    model_files = [str(cnn_model_path), str(pose_model_path), str(classes_path)]
                    print(f"Found CNN+LSTM model: {cnn_model_path}")
                    print(f"Found Pose LSTM model: {pose_model_path}")
                    print(f"Found class names: {classes_path}")
                else:
                    print(f"Missing model files in {models_dir}")
            else:
                print(f"Models or results directory not found in {latest_training_dir}")
        else:
            print("No training directories found in outputs/")
    
    # Fallback: Check for models in root directory or outputs/
    if not model_files:
        print("Checking for models in fallback locations...")
        fallback_files = [
            "best_sasl_cnn_lstm_model.pth",
            "best_sasl_pose_lstm_model.pth", 
            "pytorch_sasl_classes.json"
        ]
        
        # Check outputs directory first, then root
        for model_file in fallback_files:
            if (outputs_dir / model_file).exists():
                model_files.append(str(outputs_dir / model_file))
            elif Path(model_file).exists():
                model_files.append(model_file)
            else:
                model_files.append(None)
        
        # Check if all files found
        if None in model_files:
            missing_files = [f for f, path in zip(fallback_files, model_files) if path is None]
            print(f"ERROR: Missing PyTorch model files: {missing_files}")
            print("\nTo fix this issue:")
            print("1. Run training first using option 3")
            print("2. Or collect training data using option 1")
            print("3. Make sure training completes successfully")
            input("Press Enter to continue...")
            return
        else:
            print("Found models in fallback locations")
    
    if not model_files or len(model_files) != 3:
        print("ERROR: Could not locate all required model files")
        print("\nTo fix this issue:")
        print("1. Run training first using option 3")
        print("2. Or collect training data using option 1") 
        print("3. Make sure training completes successfully")
        input("Press Enter to continue...")
        return
    
    try:
        with open(model_files[2], 'r') as f:
            classes = json.load(f)
        
        print(f"FOUND: Trained models for {len(classes)} SASL classes:")
        for i, class_name in enumerate(classes, 1):
            print(f"   {i:2d}. {class_name}")
        
        print(f"\n Starting live dual model camera recognition...")
        print(f"Note: CNN-only recognition (option 4) typically performs better")
        
        # Import and run dual model recognition
        from sasl_camera_recognition import SASLCameraRecognition
        
        recognition = SASLCameraRecognition(
            cnn_lstm_model_path=model_files[0],
            pose_lstm_model_path=model_files[1],
            classes_path=model_files[2]
        )
        
        recognition.run_live_recognition()
        
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        input("Press Enter to continue...")
    except ImportError as e:
        print(f"ERROR: Could not import dual model recognition: {e}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"ERROR: Error during dual model recognition: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to continue...")

def batch_video_prediction():
    """Launch batch video prediction using standalone script"""
    clear_screen()
    print("Batch Video Prediction")
    print("=" * 50)
    print("Automatic batch prediction for videos in test_videos folder.")
    print("This will predict hand signs for multiple videos using your latest trained model.")
    
    try:
        import subprocess
        import sys
        
        # Check if batch_video_prediction.py exists
        script_path = Path("batch_video_prediction.py")
        if not script_path.exists():
            print("ERROR: batch_video_prediction.py script not found!")
            print("Make sure the script is in the same directory as main.py")
            input("Press Enter to continue...")
            return
        
        # Check if test_videos folder exists and has content
        test_videos_dir = Path("test_videos")
        if not test_videos_dir.exists():
            test_videos_dir.mkdir()
            print(f"Created test_videos folder: {test_videos_dir}")
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(test_videos_dir.glob(f"*{ext}"))
            video_files.extend(test_videos_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"\nNo videos found in {test_videos_dir}")
            print(f"Please add some video files to the test_videos folder first.")
            print(f"Supported formats: {', '.join(video_extensions)}")
            input("Press Enter to continue...")
            return
        
        print(f"\nFound {len(video_files)} video(s) in test_videos folder:")
        for video_file in video_files[:5]:  # Show first 5 files
            print(f"  - {video_file.name}")
        if len(video_files) > 5:
            print(f"  ... and {len(video_files) - 5} more")
        
        print(f"\nLaunching batch prediction script...")
        print("The script will automatically detect your latest trained model.")
        print()
        
        # Launch the standalone script
        result = subprocess.run([sys.executable, "batch_video_prediction.py"], 
                              cwd=Path.cwd())
        
        if result.returncode == 0:
            print("\nBatch prediction completed successfully!")
            print("Check the generated results file for detailed predictions.")
        else:
            print(f"\nBatch prediction script exited with code: {result.returncode}")
        
    except KeyboardInterrupt:
        print("\nBatch prediction interrupted by user.")
    except Exception as e:
        print(f"ERROR: Could not launch batch prediction script: {e}")
        print("You can run it manually with: python batch_video_prediction.py")
    
    input("\nPress Enter to return to main menu...")

def manage_models_and_outputs():
    """Manage models and output files"""
    clear_screen()
    print("Model & Output Management")
    print("=" * 50)
    
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("No outputs directory found.")
        print("Train some models first to see them here!")
        input("Press Enter to continue...")
        return
    
    # List training sessions
    training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
    
    if not training_dirs:
        print("No training sessions found in outputs directory.")
        input("Press Enter to continue...")
        return
    
    print(f"Found {len(training_dirs)} training session(s):")
    print()
    
    for i, training_dir in enumerate(sorted(training_dirs, key=lambda x: x.name), 1):
        timestamp_str = training_dir.name.replace("training_", "")
        
        try:
            from datetime import datetime
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp_str
        
        print(f"  {i}. {formatted_time}")
        
        # Check what files exist
        models_dir = training_dir / "models"
        results_dir = training_dir / "results"
        
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            print(f"     Models: {len(model_files)} file(s)")
            
            # Check for specific model types
            cnn_model = models_dir / "best_sasl_cnn_lstm_model.pth"
            pose_model = models_dir / "best_sasl_pose_lstm_model.pth"
            
            if cnn_model.exists():
                print(f"       CNN+LSTM model")
            if pose_model.exists():
                print(f"       Pose LSTM model")
        
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json")) + list(results_dir.glob("*.png"))
            print(f"     Results: {len(result_files)} file(s)")
        
        print()
    
    print("Management Options:")
    print("  1. View latest training results")
    print("  2. Clean old training sessions")
    print("  3. Back to main menu")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        # Show latest results
        latest_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
        results_dir = latest_dir / "results"
        
        if results_dir.exists():
            # Try to find and show training results
            result_files = list(results_dir.glob("*results*.json"))
            if result_files:
                try:
                    with open(result_files[0], 'r') as f:
                        results = json.load(f)
                    print(f"\nLatest Training Results:")
                    print(json.dumps(results, indent=2))
                except Exception as e:
                    print(f"Could not read results: {e}")
            else:
                print("No results file found")
        else:
            print("No results directory found")
        
        input("\nPress Enter to continue...")
    
    elif choice == '2':
        # Clean old sessions
        if len(training_dirs) > 1:
            print(f"\nFound {len(training_dirs)} training sessions.")
            print("Keep how many of the most recent sessions?")
            try:
                keep_count = int(input("Keep (1-10): ").strip())
                if 1 <= keep_count < len(training_dirs):
                    # Sort and keep only the most recent
                    sorted_dirs = sorted(training_dirs, key=lambda x: x.name)
                    dirs_to_remove = sorted_dirs[:-keep_count]
                    
                    print(f"\nThis will remove {len(dirs_to_remove)} old training session(s).")
                    confirm = input("Are you sure? (y/n): ").strip().lower()
                    
                    if confirm == 'y':
                        import shutil
                        for dir_to_remove in dirs_to_remove:
                            shutil.rmtree(dir_to_remove)
                            print(f"Removed: {dir_to_remove.name}")
                        print("Cleanup completed!")
                    else:
                        print("Cleanup cancelled.")
                else:
                    print("Invalid number")
            except ValueError:
                print("Invalid input")
        else:
            print("Only one training session found - nothing to clean")
        
        input("Press Enter to continue...")

def show_system_info():
    """Display system information"""
    clear_screen()
    print("System Information")
    print("=" * 50)
    
    # Python and system info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check key dependencies
    dependencies = {
        'torch': 'PyTorch (Deep Learning)',
        'cv2': 'OpenCV (Computer Vision)', 
        'mediapipe': 'MediaPipe (Pose Detection)',
        'numpy': 'NumPy (Numerical Computing)',
        'sklearn': 'Scikit-learn (Machine Learning)',
        'timm': 'TIMM (Pre-trained Models)'
    }
    
    print(f"\nDependency Status:")
    for module, description in dependencies.items():
        try:
            imported = __import__(module)
            if hasattr(imported, '__version__'):
                version = imported.__version__
            else:
                version = "Unknown"
            print(f"  {description}: v{version}")
        except ImportError:
            print(f"  ✗ {description}: Not installed")
    
    # Dataset info
    dataset_stats = get_dataset_stats()
    print(f"\nDataset Status:")
    print(f"  Classes: {dataset_stats['classes']}")
    print(f"  Total Videos: {dataset_stats['total_videos']}")
    
    # Model info
    model_stats = get_trained_models_stats()
    print(f"\nModel Status:")
    print(f"  Training Sessions: {model_stats['trained_models']}")
    print(f"  Last Training: {model_stats['last_training']}")
    
    # Hardware info
    try:
        import torch
        print(f"\nHardware:")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print(f"\nHardware: PyTorch not available")
    
    input("\nPress Enter to continue...")

def main():
    """Main application loop"""
    if not check_dependencies():
        print("\nPlease install missing dependencies before continuing.")
        input("Press Enter to exit...")
        return
    
    while True:
        show_main_menu()
        
        choice = input("Enter your choice (1-9): ").strip()
        
        if choice == '1':
            collect_video_data()
        elif choice == '2':
            train_cnn_only_models()
        elif choice == '3':
            train_dual_models()
        elif choice == '4':
            live_cnn_only_recognition()
        elif choice == '5':
            live_dual_model_recognition()
        elif choice == '6':
            batch_video_prediction()
        elif choice == '7':
            manage_models_and_outputs()
        elif choice == '8':
            show_system_info()
        elif choice == '9':
            clear_screen()
            print("Thank you for using SASL-AI Recognition System!")
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()