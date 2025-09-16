#!/usr/bin/env python3
"""
SASL Hand Detection System - Main Launcher
Single entry point for the organized SASL system with all functionality.
"""

import os
import sys
import subprocess

class SASLLauncher:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.setup_paths()
    
    def setup_paths(self):
        """Setup all necessary paths for the SASL system"""
        paths_to_add = [
            os.path.join(self.base_dir, '01_PRIMARY_SYSTEM'),
            os.path.join(self.base_dir, '02_FALLBACK_COMPATIBILITY'), 
            os.path.join(self.base_dir, '03_DATA_CONFIG'),
            os.path.join(self.base_dir, '04_DOCUMENTATION'),
            os.path.join(self.base_dir, '05_OUTPUT_GENERATED')
        ] 
        
        for path in paths_to_add:
            if path not in sys.path and os.path.exists(path):
                sys.path.insert(0, path)
    
    def show_menu(self):
        """Display main menu"""
        print("\nSASL Hand Detection System")
        print("=" * 50)
        print("Choose an option:")
        print("1. Performance Benchmark (Recommended First)")
        print("2. Train Hand-Focused CNN+LSTM Model")
        print("3. Real-Time Recognition with Hand Tracking") 
        print("4. Video Analysis & Hand Tracking Demo")
        print("5. System Information")
        print("q. Quit")
        print()
    
    def show_training_menu(self):
        """Display training options menu"""
        print("\nTraining Options")
        print("=" * 30)
        print("Choose training method:")
        print("1. GPU-Optimized Training (A10-12Q)")
        print("2. Back to Main Menu")
        print()
    
    def run_benchmark(self):
        """Run performance benchmark to help users choose training method"""
        print("\nPerformance Benchmark")
        print("=" * 50)
        print("This will test your system to recommend the best training method...")
        print("The benchmark takes about 1-2 minutes to complete.")
        
        confirm = input("\nRun benchmark? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            return
        
        script_path = os.path.join(self.base_dir, 'simple_gpu_benchmark.py')
        
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print("\n" + "="*60)
            print("RECOMMENDATIONS BASED ON BENCHMARK:")
            print("="*60)
            
            # Try to read the benchmark results to provide recommendations
            results_path = os.path.join(self.base_dir, "05_OUTPUT_GENERATED", "simple_benchmark_results.json")
            
            if os.path.exists(results_path):
                try:
                    import json
                    with open(results_path, 'r') as f:
                        data = json.load(f)
                    
                    results = data.get('results', [])
                    gpu_available = data.get('system_info', {}).get('cuda_available', False)
                    
                    if gpu_available and len(results) >= 2:
                        cpu_result = next((r for r in results if r['device'] == 'cpu'), None)
                        gpu_result = next((r for r in results if r['device'] == 'cuda'), None)
                        
                        if cpu_result and gpu_result:
                            speedup = cpu_result['avg_time_per_iteration'] / gpu_result['avg_time_per_iteration']
                            
                            if speedup > 5.0:
                                print(">> HIGHLY RECOMMENDED: GPU Training")
                                print(f"   Your GPU is {speedup:.1f}x faster than CPU!")
                                print("   Use Option 1 (GPU Training) for best performance")
                            elif speedup > 2.0:
                                print("+ RECOMMENDED: GPU Training")  
                                print(f"   Your GPU is {speedup:.1f}x faster than CPU")
                                print("   Use Option 1 (GPU Training)")
                            else:
                                print("* SUGGESTED: Hybrid Training")
                                print("   Moderate GPU speedup - try Option 3 (Hybrid)")
                        else:
                            print("| Use CPU Training (Option 2)")
                    else:
                        print("| No GPU detected - Use CPU Training (Option 2)")
                        
                except Exception as e:
                    print(f"Could not read benchmark results: {e}")
                    print("| Check the benchmark output above for recommendations")
            else:
                print("| Check the benchmark output above for recommendations")
                
        except FileNotFoundError:
            print(f"Error: Could not find benchmark script at {script_path}")
        except subprocess.CalledProcessError:
            print("Benchmark failed. You can still proceed with training.")
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
    
    def run_training_with_method(self, method):
        """Run training with GPU-optimized method for A10-12Q"""
        if method == 'cpu':
            description = 'GPU-Optimized Training (A10-12Q)'
            script_name = 'hand_focused_CNN_LSTM.py'
            recommended_batch_size = 8  # REDUCED for system stability
            recommended_epochs = 10  # REDUCED for faster completion
            print(f"\n{description}")
            print("=" * 60)
            print("🔥 GPU-Optimized Training - NVIDIA A10-12Q (Stability Mode)")
            print("   ⚡ Mixed precision training with conservative settings")
            print("   🧠 Lightweight ResNet18 backbone for system stability")
            print("   💾 8GB VRAM optimized batch processing")
            print("   🎯 Reduced augmentation to prevent system crashes")
            print(f"\n| Recommended Settings for {description}:")
            print(f"   Batch Size: {recommended_batch_size} (Conservative for system stability)")
            print(f"   Epochs: {recommended_epochs} (Reduced for faster completion)")
            print(f"\n* Training Configuration:")
            while True:
                try:
                    batch_input = input(f"Enter batch size (recommended: {recommended_batch_size}, press Enter for default): ").strip()
                    if batch_input == "":
                        batch_size = recommended_batch_size
                        break
                    else:
                        batch_size = int(batch_input)
                        if batch_size < 1:
                            print("X Batch size must be at least 1")
                            continue
                        if batch_size > 16:
                            print("! Warning: Large batch sizes may cause system crashes")
                        elif batch_size < 4:
                            print("! Note: Very small batch sizes may not utilize GPU well")
                        break
                except ValueError:
                    print("X Please enter a valid number")
            while True:
                try:
                    epoch_input = input(f"Enter number of epochs (recommended: {recommended_epochs}, press Enter for default): ").strip()
                    if epoch_input == "":
                        epochs = recommended_epochs
                        break
                    else:
                        epochs = int(epoch_input)
                        if epochs < 1:
                            print("X Epochs must be at least 1")
                            continue
                        if epochs > 25:
                            print("! Warning: Training with >25 epochs may take a very long time")
                        break
                except ValueError:
                    print("X Please enter a valid number")
            print(f"\n> Final Training Configuration:")
            print(f"   Method: {description}")
            print(f"   Batch Size: {batch_size}")
            print(f"   Epochs: {epochs}")
            # Updated time estimates for stability mode
            estimated_time = (2 + batch_size * 0.1) * epochs  # More conservative estimate
            if estimated_time < 60:
                time_str = f"{estimated_time:.0f} minutes"
            else:
                hours = estimated_time // 60
                minutes = estimated_time % 60
                time_str = f"{hours:.0f}h {minutes:.0f}m"
            print(f"   Estimated Time: {time_str} (Stability mode - conservative estimate)")
            print(f"\n🚀 Stability Mode Features:")
            print(f"   • Mixed Precision Training (controlled speed)")
            print(f"   • Single-threaded Data Loading (no multiprocessing crashes)")
            print(f"   • ResNet18 Backbone (lightweight, stable)")
            print(f"   • Reduced Augmentation (2x instead of 10x)")
            print(f"   • Conservative Memory Usage (system friendly)")
            print(f"   • Memory-optimized for system stability")
            print(f"\n💡 System Stability Notes:")
            print(f"   • Training now uses conservative settings to prevent crashes")
            print(f"   • Batch size reduced to prevent memory issues")
            print(f"   • Single-threaded loading prevents multiprocessing hangs")
            print("\n🎯 Training will begin shortly...")
            print("You can stop training at any time with Ctrl+C")
            print("The model will be saved automatically during training.")
            confirm = input(f"\nStart {description} with these settings? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                self.run_script_with_params(script_name, description, batch_size, epochs)
            else:
                print("Training cancelled.")
        else:
            print("Invalid training method selected.")
    
    def run_script(self, script_name, description):
        """Run a script from the PRIMARY_SYSTEM"""
        script_path = os.path.join(self.base_dir, '01_PRIMARY_SYSTEM', script_name)
        print(f"\n{description}")
        print("=" * 50)
        
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print(f"\n{description} completed successfully!")
        except FileNotFoundError:
            print(f"Error: Could not find {script_path}")
        except subprocess.CalledProcessError:
            print(f"{description} failed. Check error messages above.")
        except KeyboardInterrupt:
            print(f"\n{description} interrupted by user.")
    
    def run_script_with_params(self, script_name, description, batch_size, epochs):
        """Run a script from the PRIMARY_SYSTEM with custom parameters"""
        script_path = os.path.join(self.base_dir, '01_PRIMARY_SYSTEM', script_name)
        print(f"\n{description}")
        print("=" * 50)
        
        try:
            # Pass batch size and epochs as command line arguments
            subprocess.run([
                sys.executable, script_path, 
                '--batch_size', str(batch_size),
                '--epochs', str(epochs)
            ], check=True)
            print(f"\n{description} completed successfully!")
        except FileNotFoundError:
            print(f"Error: Could not find {script_path}")
        except subprocess.CalledProcessError:
            print(f"{description} failed. Check error messages above.")
        except KeyboardInterrupt:
            print(f"\n{description} interrupted by user.")
    
    def show_system_info(self):
        """Show system information and file structure"""
        print("\nSASL System Information")
        print("=" * 50)
        
        # Check directory structure
        directories = [
            ("Primary System", "01_PRIMARY_SYSTEM"),
            ("Fallback/Compatibility", "02_FALLBACK_COMPATIBILITY"),
            ("Data/Config", "03_DATA_CONFIG"),
            ("Documentation", "04_DOCUMENTATION"),
            ("Output/Generated", "05_OUTPUT_GENERATED")
        ]
        
        for desc, dir_name in directories:
            dir_path = os.path.join(self.base_dir, dir_name)
            status = "+" if os.path.exists(dir_path) else "X"
            print(f"{status} {desc}: {dir_name}")
            
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.py')]
                if files:
                    for file in files[:3]:  # Show first 3 files
                        print(f"    {file}")
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more files")
        
        # Check key files
        print(f"\nDataset: {'+' if os.path.exists('dataset') else 'X'}")
        
        # Check GPU availability
        try:
            import torch
            print(f"PyTorch: + (v{torch.__version__})")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU: + {gpu_name} ({gpu_memory:.1f}GB)")
                print("   Recommended: GPU Training")
            else:
                print("GPU: X (CUDA not available)")
                print("   Recommended: CPU Training")
        except ImportError:
            print("PyTorch: X (Not installed)")
        
        # Check dependencies
        try:
            import cv2, mediapipe
            print("OpenCV & MediaPipe: +")
        except ImportError as e:
            print(f"Missing dependencies: {e}")
    
    def run_camera_with_model_selection(self):
        """Run camera with model selection menu"""
        print("\nSASL Camera - Model Selection")
        print("=" * 40)
        output_dir = os.path.join(self.base_dir, "05_OUTPUT_GENERATED")
        
        # Check available models
        available_models = []
        model_files = {
            "gpu_optimized_sasl_model.pth": "GPU-Optimized Model (A10-12Q)",
            "focused_sasl_model.pth": "Focused Model (6 classes)",
            "simple_sasl_model.pth": "Simple Model"
        }
        
        print("Available models:")
        choice_map = {}
        choice_num = 1
        
        for model_file, description in model_files.items():
            model_path = os.path.join(output_dir, model_file)
            if os.path.exists(model_path):
                print(f"{choice_num}. {description}")
                choice_map[str(choice_num)] = (model_path, description)
                available_models.append((model_path, description))
                choice_num += 1
        
        if not available_models:
            print("No trained models found!")
            print("Please train a model first using option 2 (Training).")
            return
            
        print("0. Back to main menu")
        
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}, Enter for default): ").strip()
            if choice == "0":
                return
            elif choice == "" or choice == "1":
                # Use first available model as default
                model_path, description = available_models[0]
                print(f"Using {description}...")
                self.run_camera_with_specific_model(model_path, description)
            elif choice in choice_map:
                model_path, description = choice_map[choice]
                print(f"Using {description}...")
                self.run_camera_with_specific_model(model_path, description)
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    def run_camera_with_specific_model(self, model_path, description):
        """Run camera with a specific model by creating a custom launcher"""
        print(f"\n{description}")
        print("=" * 50)
        print("Starting camera with selected model...")
        print("Press 'q' to quit, 'h' to toggle hand detection, 'r' to reset")
        
        # Create a temporary script to force model selection
        # Properly escape the path for Windows
        escaped_model_path = model_path.replace('\\', '\\\\')
        temp_script_content = f'''#!/usr/bin/env python3
"""
Temporary launcher for enhanced camera with specific model
"""
import os
import sys

# Set environment variable to force specific model
os.environ['SASL_FORCE_MODEL_PATH'] = r"{escaped_model_path}"

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the enhanced camera
from enhanced_camera import run_enhanced_camera

if __name__ == "__main__":
    print("Forced model selection: {description}")
    print(r"Model path: {escaped_model_path}")
    run_enhanced_camera()
'''
        
        temp_script_path = os.path.join(self.base_dir, '01_PRIMARY_SYSTEM', 'temp_camera_launcher.py')
        
        try:
            # Write temporary script
            with open(temp_script_path, 'w') as f:
                f.write(temp_script_content)
            
            # Run the temporary script
            subprocess.run([sys.executable, temp_script_path], check=True)
            
        except subprocess.CalledProcessError:
            print(f"Camera failed to run with {description}")
        except KeyboardInterrupt:
            print(f"\nCamera interrupted by user")
        finally:
            # Clean up temporary script
            if os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                except:
                    pass  # Ignore cleanup errors
    
    def show_manual_fallback(self):
        """Show manual commands if interactive guide fails"""
        print("\nManual Commands:")
        print("=" * 30)
        print("GPU-Optimized Training: python 01_PRIMARY_SYSTEM/hand_focused_CNN_LSTM.py")
        print("Camera: python 01_PRIMARY_SYSTEM/enhanced_camera.py")
        print("Demo: python 01_PRIMARY_SYSTEM/hand_tracking_demo.py")
    
    def run(self):
        """Main run loop"""
        while True:
            try:
                self.show_menu()
                choice = input("Enter your choice: ").strip().lower()
                
                if choice == '1':
                    self.run_benchmark()
                
                elif choice == '2':
                    # Training submenu
                    while True:
                        self.show_training_menu()
                        train_choice = input("Enter your choice: ").strip()
                        
                        if train_choice == '1':
                            self.run_training_with_method('cpu')  # This will actually run GPU-optimized training
                            break
                        elif train_choice == '2':
                            break  # Back to main menu
                        else:
                            print("Invalid choice. Please try again.")
                
                elif choice == '3':
                    # Launch camera with model selection
                    self.run_camera_with_model_selection()
                
                elif choice == '4':
                    self.run_script('hand_tracking_demo.py', 'Video Analysis & Hand Tracking Demo')
                
                elif choice == '5':
                    self.show_system_info()
                
                elif choice in ['q', 'quit', 'exit']:
                    print("\nThank you for using SASL Hand Detection System!")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
                
                if choice not in ['5']:  # Don't pause for info screens
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\nThank you for using SASL Hand Detection System!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                self.show_manual_fallback()
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    launcher = SASLLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
