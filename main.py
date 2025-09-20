#!/usr/bin/env python3
"""
SASL-AI Main Menu System
========================

Complete SASL recognition system with:
1. Video data collection
2. Video-based training with transfer learning
3. Live camera recognition
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
    """Check if required dependencies are available - lightweight version"""
    # Skip heavy dependency checking at startup for speed
    # Dependencies will be checked when actually needed by each module
    return True

def check_dependencies_verbose():
    """Check if required dependencies are available with full verification"""
    required_modules = ['cv2', 'torch', 'mediapipe', 'numpy', 'sklearn', 'timm']
    missing_modules = []
    
    print("Checking dependencies...")
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"  ✗ {module}")
    
    if missing_modules:
        print(f"\n!!! Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using:")
        print("pip install torch torchvision torchaudio opencv-python mediapipe numpy scikit-learn timm")
        return False
    return True

def get_dataset_stats():
    """Get current dataset statistics"""
    video_dataset_path = Path("video_dataset")
    outputs_path = Path("outputs")
    
    stats = {
        'classes': 0,
        'total_videos': 0,
        'trained_models': 0,
        'last_training': 'Never'
    }
    
    # Count video classes and videos
    if video_dataset_path.exists():
        class_dirs = [d for d in video_dataset_path.iterdir() if d.is_dir() and d.name != '.gitkeep']
        stats['classes'] = len(class_dirs)
        
        for class_dir in class_dirs:
            video_files = len(list(class_dir.glob("*.mp4")))
            stats['total_videos'] += video_files
    
    # Check for trained models - updated for PyTorch and timestamped directories
    model_files = []
    
    # Check in timestamped training directories first
    if outputs_path.exists():
        training_dirs = [d for d in outputs_path.iterdir() if d.is_dir() and d.name.startswith("training_")]
        for training_dir in training_dirs:
            models_dir = training_dir / "models"
            if models_dir.exists():
                model_files.extend(list(models_dir.glob("*.pth")))
    
    # Fallback: check outputs/ and root for .pth files
    if outputs_path.exists():
        model_files.extend(list(outputs_path.glob("*.pth")))
    model_files.extend(list(Path(".").glob("*.pth")))
    
    stats['trained_models'] = len(model_files)
    
    # Check last training - updated for timestamped directories
    last_training = 'Never'
    
    # First check training directories for latest results
    if outputs_path.exists():
        training_dirs = [d for d in outputs_path.iterdir() if d.is_dir() and d.name.startswith("training_")]
        if training_dirs:
            # Get the latest training directory
            latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            
            # Extract date from directory name: training_YYYYMMDD_HHMMSS
            try:
                dir_name = latest_training_dir.name  # training_YYYYMMDD_HHMMSS
                date_part = dir_name.replace('training_', '').split('_')[0]  # YYYYMMDD
                # Convert YYYYMMDD to YYYY-MM-DD
                last_training = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            except:
                last_training = 'Recent'
    
    # Fallback: check old results file
    if last_training == 'Never':
        results_file = outputs_path / "video_sasl_results.json" if outputs_path.exists() else Path("video_sasl_results.json")
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    if 'training_date' in results:
                        last_training = results['training_date'][:10]  # Just the date part
            except:
                pass
    
    stats['last_training'] = last_training
    
    return stats

def show_main_menu():
    """Display the main menu"""
    clear_screen()
    print_banner()
    
    # Show current status
    stats = get_dataset_stats()
    print(f"\n Current Status:")
    print(f"   Classes: {stats['classes']}")
    print(f"   Videos: {stats['total_videos']}")
    print(f"   Trained Models: {stats['trained_models']}")
    print(f"   Last Training: {stats['last_training']}")
    
    print(f"\n Main Menu:")
    print(f"   1.  Collect Video Data")
    print(f"   2.  Train Models")
    print(f"   3.  Live Camera Recognition")
    print(f"   4.  Manage Models & Outputs")
    print(f"   5.  System Information")
    print(f"   6.  Exit")
    print(f"\n" + "=" * 70)

def collect_video_data():
    """Launch video data collection"""
    clear_screen()
    print("SASL Video Data Collection")
    print("=" * 50)
    print("This will launch the interactive video collector.")
    print("Press SPACE to start recording, SPACE again to stop and save!")
    print("Quality indicator shows when you're ready to record.")
    print("\nLaunching video collector...")
    
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
        print("Please check if all dependencies are installed.")
    except Exception as e:
        print(f"ERROR: Error during video collection: {e}")

def configure_training_parameters():
    """Configure training parameters interactively"""
    clear_screen()
    print("Training Configuration")
    print("=" * 50)
    
    print("Configure your training parameters:")
    print("(Press Enter for default values)\n")
    
    # Epochs configuration
    while True:
        try:
            epochs_input = input("Number of training epochs [default: 100]: ").strip()
            epochs = 100 if epochs_input == "" else int(epochs_input)
            if epochs > 0:
                break
            else:
                print("Epochs must be greater than 0")
        except ValueError:
            print("Please enter a valid number")
    
    # CNN Batch size configuration  
    while True:
        try:
            cnn_batch_input = input("CNN+LSTM batch size [default: 4]: ").strip()
            batch_size_cnn = 4 if cnn_batch_input == "" else int(cnn_batch_input)
            if batch_size_cnn > 0:
                break
            else:
                print("Batch size must be greater than 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Pose Batch size configuration
    while True:
        try:
            pose_batch_input = input("Pose LSTM batch size [default: 8]: ").strip()
            batch_size_pose = 8 if pose_batch_input == "" else int(pose_batch_input)
            if batch_size_pose > 0:
                break
            else:
                print("Batch size must be greater than 0")
        except ValueError:
            print("Please enter a valid number")
    
    # Data augmentation configuration
    print("\nData Augmentation:")
    print("0 = No augmentation")
    print("1 = 1x augmentation (doubles dataset)")  
    print("2 = 2x augmentation (triples dataset)")
    print("3 = 3x augmentation (quadruples dataset)")
    print("Note: Augmentation includes brightness, contrast, rotation, noise, and temporal variations")
    
    while True:
        try:
            aug_input = input("Augmentation factor [default: 0]: ").strip()
            augmentation_factor = 0 if aug_input == "" else int(aug_input)
            if augmentation_factor >= 0:
                break
            else:
                print("Augmentation factor must be 0 or greater")
        except ValueError:
            print("Please enter a valid number")
    
    # Summary
    print(f"\nTraining Configuration Summary:")
    print(f"  Epochs: {epochs}")
    print(f"  CNN+LSTM Batch Size: {batch_size_cnn}")
    print(f"  Pose LSTM Batch Size: {batch_size_pose}")
    print(f"  Data Augmentation: {augmentation_factor}x")
    if augmentation_factor > 0:
        print(f"  Dataset will be expanded by {augmentation_factor + 1}x")
    
    estimated_time = epochs * (2 + augmentation_factor) * 0.5  # Rough estimate
    print(f"  Estimated training time: {estimated_time:.0f}-{estimated_time*2:.0f} minutes")
    
    confirm = input(f"\nProceed with these settings? (y/n) [y]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        return {
            'epochs': epochs,
            'batch_size_cnn': batch_size_cnn, 
            'batch_size_pose': batch_size_pose,
            'augmentation_factor': augmentation_factor
        }
    else:
        return None

def train_models():
    """Launch model training"""
    clear_screen()
    print("SASL Model Training")
    print("=" * 50)
    
    # Check if video dataset exists
    video_dataset_path = Path("video_dataset")
    has_classes = video_dataset_path.exists() and any(d.is_dir() and d.name != '.gitkeep' for d in video_dataset_path.iterdir())
    if not has_classes:
        print("No video dataset found!")
        print("Please collect video data first using option 1.")
        return
    
    # Show dataset info
    stats = get_dataset_stats()
    print(f"Dataset Information:")
    print(f"   Classes: {stats['classes']}")
    print(f"   Total Videos: {stats['total_videos']}")
    print(f"   Average per class: {stats['total_videos'] / max(stats['classes'], 1):.1f}")
    
    if stats['classes'] < 2:
        print("Need at least 2 classes to train models!")
        return
    
    if stats['total_videos'] < 10:
        print("!!! Warning: Very small dataset. Consider collecting more videos.")
        print("Recommended: At least 5-10 videos per class")
    
    # Configure training parameters
    print(f"\nTraining Configuration:")
    print(f"  1. Use default settings (quick training)")
    print(f"  2. Configure custom parameters")
    print(f"  3. Back to main menu")
    
    choice = input(f"\nEnter choice (1-3): ").strip()
    
    if choice == '3':
        return
    elif choice == '2':
        # Custom configuration
        config = configure_training_parameters()
        if config is None:
            return
    else:
        # Default configuration
        config = {
            'epochs': 100,
            'batch_size_cnn': 4,
            'batch_size_pose': 8,
            'augmentation_factor': 0
        }
        print(f"\nUsing default configuration:")
        print(f"  Epochs: {config['epochs']}")
        print(f"  CNN Batch Size: {config['batch_size_cnn']}")
        print(f"  Pose Batch Size: {config['batch_size_pose']}")
        print(f"  Augmentation: {config['augmentation_factor']}x")
    
    print(f"\nReady to train video-based SASL models!")
    print(f"This will create:")
    print(f"   • CNN+LSTM model (visual features + temporal modeling)")
    print(f"   • Pose LSTM model (MediaPipe landmarks + temporal modeling)")
    print(f"   • Ensemble predictions combining both models")
    
    expected_dataset_size = stats['total_videos'] * (1 + config['augmentation_factor'])
    print(f"\nDataset size after augmentation: {expected_dataset_size} videos")
    
    estimated_time = config['epochs'] * (2 + config['augmentation_factor']) * 0.5
    print(f"Estimated training time: {estimated_time:.0f}-{estimated_time*2:.0f} minutes")
    
    print(f"\nStarting training in 3 seconds...")
    
    try:
        # Ensure outputs directory exists
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Import and run training with custom parameters
        from video_based_sasl_training import VideoSASLTrainer
        
        print("\nInitializing trainer with custom configuration...")
        trainer = VideoSASLTrainer(
            video_dataset_path="video_dataset",
            sequence_length=30,
            input_size=(224, 224),
            epochs=config['epochs'],
            batch_size_cnn=config['batch_size_cnn'],
            batch_size_pose=config['batch_size_pose'],
            augmentation_factor=config['augmentation_factor']
        )
        
        print("Starting training process...")
        cnn_lstm_model, pose_lstm_model = trainer.train_models()
        
        if cnn_lstm_model is not None:
            # Move model files to outputs directory
            import shutil
            
            model_files = [
                "best_sasl_cnn_lstm_model.h5",
                "best_sasl_pose_lstm_model.h5",
                "video_sasl_results.json",
                "video_sasl_classes.json"
            ]
            
            print(f"\n Moving model files to outputs directory...")
            for file in model_files:
                if Path(file).exists():
                    shutil.move(file, outputs_dir / file)
                    print(f"   MOVED: {file}")
            
            print(f"\n Training completed successfully!")
            print(f" Models saved in: outputs/")
            
        else:
            print(f"\n Training failed. Check your dataset.")
        
    except ImportError as e:
        print(f"ERROR: Could not import training system: {e}")
        print("Please check if all dependencies are installed.")
    except Exception as e:
        print(f"ERROR: Error during training: {e}")

def live_camera_recognition():
    """Launch live camera recognition"""
    clear_screen()
    print(" Live SASL Camera Recognition")
    print("=" * 50)
    
    # Check if trained models exist - updated to check training folders
    outputs_dir = Path("outputs")
    cnn_model_path = None
    pose_model_path = None
    classes_path = None
    
    # First check in timestamped training directories
    if outputs_dir.exists():
        training_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")]
        if training_dirs:
            # Get the latest training directory
            latest_training_dir = sorted(training_dirs, key=lambda x: x.name)[-1]
            models_dir = latest_training_dir / "models"
            results_dir = latest_training_dir / "results"
            
            if models_dir.exists() and results_dir.exists():
                cnn_model_path = models_dir / "best_sasl_cnn_lstm_model.pth"
                pose_model_path = models_dir / "best_sasl_pose_lstm_model.pth"
                # Try both class name file formats
                classes_path = results_dir / "pytorch_sasl_classes.json"
                if not classes_path.exists():
                    classes_path = results_dir / "class_names.json"
    
    # Fallback: Check in outputs/ and root directory  
    if not (cnn_model_path and cnn_model_path.exists()):
        cnn_model_path = outputs_dir / "best_sasl_cnn_lstm_model.pth"
        if not cnn_model_path.exists():
            cnn_model_path = Path("best_sasl_cnn_lstm_model.pth")
    
    if not (pose_model_path and pose_model_path.exists()):
        pose_model_path = outputs_dir / "best_sasl_pose_lstm_model.pth"
        if not pose_model_path.exists():
            pose_model_path = Path("best_sasl_pose_lstm_model.pth")
    
    if not (classes_path and classes_path.exists()):
        classes_path = outputs_dir / "pytorch_sasl_classes.json"
        if not classes_path.exists():
            classes_path = outputs_dir / "class_names.json"
            if not classes_path.exists():
                classes_path = Path("pytorch_sasl_classes.json")
                if not classes_path.exists():
                    classes_path = Path("class_names.json")
    
    if not (cnn_model_path.exists() and pose_model_path.exists() and classes_path.exists()):
        print("ERROR: No trained models found!")
        print("Please train models first using option 2.")
        print("\nSearched for models in:")
        print(f"  - Training directories: {outputs_dir}/training_*/models/")
        print(f"  - Outputs directory: {outputs_dir}")
        print(f"  - Root directory: current folder")
        return
    
    try:
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        
        print(f"FOUND: Trained models for {len(classes)} SASL classes:")
        for i, class_name in enumerate(classes, 1):
            print(f"   {i:2d}. {class_name}")
        
        print(f"\n Starting live camera recognition...")
        print(f"Controls:")
        print(f"   • Hold still and sign clearly")
        print(f"   • Press 'q' to quit")
        print(f"   • Press 'r' to reset prediction buffer")
        print(f"   • Press 'h' to toggle UI mode")
        print(f"   • Press 'o' to toggle overlays")
        print(f"   • Press 'c' for clean mode")
        
        print(f"\nInitializing camera recognition...")
        print(f"Loading models:")
        print(f"  CNN Model: {cnn_model_path}")
        print(f"  Pose Model: {pose_model_path}")
        print(f"  Classes: {classes_path}")
        
        # Launch camera recognition
        print(f"Importing camera recognition module...")
        from sasl_camera_recognition import SASLCameraRecognition
        
        print(f"Creating camera recognition instance...")
        camera = SASLCameraRecognition(
            cnn_model_path=str(cnn_model_path),
            pose_model_path=str(pose_model_path),
            classes_path=str(classes_path)
        )
        
        print(f"Starting live recognition...")
        camera.run_live_recognition()
        
    except ImportError as e:
        print(f"\n" + "="*50)
        print(f"IMPORT ERROR: Could not import camera system")
        print(f"Error: {e}")
        print(f"Please check if all dependencies are installed.")
        print(f"Use option 5 (System Information) to check dependencies.")
        print(f"="*50)
        input("Press Enter to return to main menu...")
    except Exception as e:
        print(f"\n" + "="*50)
        print(f"CAMERA ERROR: Error during camera recognition")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to show more details
        import traceback
        print(f"\nFull error details:")
        traceback.print_exc()
        
        print(f"\nPossible causes:")
        print(f"  - Camera not connected or in use by another app")
        print(f"  - Model files corrupted or incompatible")
        print(f"  - GPU/memory issues")
        print(f"  - Missing dependencies")
        print(f"="*50)
        input("Press Enter to return to main menu...")

def manage_models_outputs():
    """Manage models and outputs"""
    clear_screen()
    print(" Model & Output Management")
    print("=" * 50)
    
    outputs_dir = Path("outputs")
    
    # List model files
    model_files = []
    if outputs_dir.exists():
        model_files.extend(list(outputs_dir.glob("*.h5")))
        model_files.extend(list(outputs_dir.glob("*.json")))
    
    # Also check root directory
    root_files = list(Path(".").glob("*.h5")) + list(Path(".").glob("*sasl*.json"))
    
    print(f" Current Files:")
    print(f"\n In outputs/ directory:")
    if outputs_dir.exists():
        output_files = list(outputs_dir.iterdir())
        if output_files:
            for file in output_files:
                size = file.stat().st_size / (1024*1024) if file.is_file() else 0
                print(f"    {file.name} ({size:.1f} MB)")
        else:
            print(f"   (empty)")
    else:
        print(f"   (directory doesn't exist)")
    
    print(f"\n In root directory:")
    if root_files:
        for file in root_files:
            size = file.stat().st_size / (1024*1024)
            print(f"    {file.name} ({size:.1f} MB)")
    else:
        print(f"   (no SASL files)")
    
    print(f"\n Options:")
    print(f"   1. Create outputs directory")
    print(f"   2. Move root files to outputs")
    print(f"   3. Clean up old files")
    print(f"   4. Show model details")
    print(f"   5. Back to main menu")
    
    choice = input(f"\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        outputs_dir.mkdir(exist_ok=True)
        print(f"CREATED: outputs directory")
    
    elif choice == '2':
        outputs_dir.mkdir(exist_ok=True)
        import shutil
        moved_count = 0
        
        for file in root_files:
            try:
                shutil.move(str(file), outputs_dir / file.name)
                print(f"MOVED: {file.name}")
                moved_count += 1
            except Exception as e:
                print(f"ERROR: Error moving {file.name}: {e}")
        
        print(f"MOVED: {moved_count} files to outputs/")
    
    elif choice == '3':
        print(f"WARNING: This will delete old model and result files.")
        confirm = input(f"Type 'DELETE' to confirm: ").strip()
        if confirm == 'DELETE':
            deleted_count = 0
            for file in root_files:
                if file.name.endswith(('.h5', '.json')):
                    file.unlink()
                    deleted_count += 1
                    print(f"DELETED: {file.name}")
            print(f"DELETED: {deleted_count} files")
        else:
            print(f"ERROR: Cancelled")
    
    elif choice == '4':
        # Show model details
        results_files = list(outputs_dir.glob("*results.json")) if outputs_dir.exists() else []
        results_files.extend(list(Path(".").glob("*results.json")))
        
        if results_files:
            latest_results = max(results_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_results, 'r') as f:
                    results = json.load(f)
                
                print(f"\n Latest Training Results:")
                print(f"   Dataset size: {results.get('dataset_size', 'Unknown')}")
                print(f"   Classes: {results.get('num_classes', 'Unknown')}")
                print(f"   CNN+LSTM accuracy: {results.get('cnn_lstm_accuracy', 0):.3f}")
                print(f"   Pose LSTM accuracy: {results.get('pose_lstm_accuracy', 0):.3f}")
                print(f"   Ensemble accuracy: {results.get('ensemble_accuracy', 0):.3f}")
                print(f"   Training date: {results.get('training_date', 'Unknown')}")
                
            except Exception as e:
                print(f"ERROR: Error reading results: {e}")
        else:
            print(f" No training results found")

def show_system_info():
    """Show system information"""
    clear_screen()
    print(" System Information")
    print("=" * 50)
    
    print(f" Python: {sys.version}")
    print(f" Current directory: {Path.cwd()}")
    
    # Check dependencies
    print(f"\n Dependencies:")
    check_dependencies_verbose()
    
    # Additional module versions
    modules_to_check = {
        'timm': 'TIMM (PyTorch Image Models)',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn'
    }
    
    print(f"\n Module Versions:")
    for module, name in modules_to_check.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'Unknown version')
            print(f"   {name}: {version}")
        except ImportError:
            print(f"   {name}: Not installed")
    
    # Directory structure
    print(f"\n Project Structure:")
    important_paths = [
        ("video_dataset/", "Video dataset directory"),
        ("outputs/", "Model outputs directory"),
        ("video_based_sasl_training.py", "Training system"),
        ("sasl_video_collector.py", "Data collection"),
        ("sasl_camera_recognition.py", "Camera recognition"),
        ("main.py", "Main menu (this file)")
    ]
    
    for path, description in important_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                count = len(list(path_obj.iterdir()))
                print(f"   FOUND: {path} - {description} ({count} items)")
            else:
                size = path_obj.stat().st_size / 1024
                print(f"   FOUND: {path} - {description} ({size:.1f} KB)")
        else:
            print(f"   MISSING: {path} - {description} (missing)")
    
    # GPU check
    print(f"\n GPU Information:")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({memory:.1f} GB)")
            print(f"   Current device: {torch.cuda.current_device()}")
        else:
            print(f"   INFO: No GPU detected - using CPU")
    except:
        print(f"   ERROR: Could not check GPU status")
    
    print(f"\nSystem information complete.")

def create_gitignore():
    """Create or update .gitignore file"""
    gitignore_content = """# SASL-AI Project .gitignore

# Video dataset (large files)
video_dataset/
*.mp4
*.avi
*.mov
*.mkv
*.wmv

# Model outputs
outputs/
*.h5
*.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints

# Temporary files
*.tmp
*.log
"""
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        # Check if our content is already there
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
        
        if "# SASL-AI Project" not in existing_content:
            # Append our content
            with open(gitignore_path, 'a') as f:
                f.write("\n" + gitignore_content)
            print("UPDATED: existing .gitignore")
        else:
            print("INFO: .gitignore already configured for SASL-AI")
    else:
        # Create new .gitignore
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("CREATED: new .gitignore file")

def main():
    """Main menu loop"""
    # Setup
    Path("outputs").mkdir(exist_ok=True)
    create_gitignore()
    
    if not check_dependencies():
        print("\nERROR: Please install required dependencies before continuing.")
        print("Use System Information (option 5) to check dependencies.")
        return
    
    while True:
        try:
            show_main_menu()
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                collect_video_data()
            elif choice == '2':
                train_models()
            elif choice == '3':
                live_camera_recognition()
            elif choice == '4':
                manage_models_outputs()
            elif choice == '5':
                show_system_info()
            elif choice == '6':
                clear_screen()
                print(" Thank you for using SASL-AI!")
                print(" Keep building amazing sign language recognition!")
                break
            else:
                print("ERROR: Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            print("Returning to main menu...")

if __name__ == "__main__":
    main()