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
        print("1. GPU Training (Fastest - Recommended)")
        print("2. CPU Training (Slower but Compatible)")
        print("3. Hybrid CPU+GPU Training (Balanced)")
        print("4. Back to Main Menu")
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
        """Run training with specified method"""
        training_scripts = {
            'gpu': 'gpu_optimized_training.py',
            'cpu': 'hand_focused_CNN_LSTM.py', 
            'hybrid': 'hybrid_cpu_gpu_training.py'
        }
        
        descriptions = {
            'gpu': 'GPU-Optimized Training (Fastest)',
            'cpu': 'CPU Training (Original Method)',
            'hybrid': 'Hybrid CPU+GPU Training (Balanced)'
        }
        
        # Recommended settings for each method
        recommended_settings = {
            'gpu': {
                'batch_size': 8,
                'epochs': 15,
                'reason': 'Optimized for GTX 1650 4GB VRAM - higher batch size for GPU efficiency'
            },
            'cpu': {
                'batch_size': 2,
                'epochs': 20,
                'reason': 'Lower batch size to avoid memory issues, more epochs for convergence'
            },
            'hybrid': {
                'batch_size': 4,
                'epochs': 18,
                'reason': 'Balanced settings for CPU+GPU workload distribution'
            }
        }
        
        script_name = training_scripts.get(method)
        description = descriptions.get(method)
        
        if script_name:
            print(f"\n{description}")
            print("=" * 60)
            
            # Show method-specific information
            if method == 'gpu':
                print(">> This version is optimized for maximum GPU utilization!")
                print("   Expected GPU utilization: 70-90% (much higher than previous version)")
                print("   No MediaPipe bottlenecks - designed for speed and efficiency")
            elif method == 'cpu':
                print(">> CPU Training - Universal compatibility")
                print("   Works on all systems, slower but reliable")
                print("   Uses hand-focused training with MediaPipe detection")
            elif method == 'hybrid':
                print("+ Hybrid Training - Balanced performance")
                print("   CPU handles data processing, GPU handles model training")
                print("   Good compromise between speed and compatibility")
            
            # Display recommended settings
            rec = recommended_settings[method]
            print(f"\n| Recommended Settings for {description}:")
            print(f"   Batch Size: {rec['batch_size']} ({rec['reason']})")
            print(f"   Epochs: {rec['epochs']}")
            
            # Get user input for batch size
            print(f"\n* Training Configuration:")
            while True:
                try:
                    batch_input = input(f"Enter batch size (recommended: {rec['batch_size']}, press Enter for default): ").strip()
                    if batch_input == "":
                        batch_size = rec['batch_size']
                        break
                    else:
                        batch_size = int(batch_input)
                        if batch_size < 1:
                            print("X Batch size must be at least 1")
                            continue
                        if method == 'gpu' and batch_size > 16:
                            print("! Warning: Large batch sizes may cause GPU memory issues")
                        elif method == 'cpu' and batch_size > 4:
                            print("! Warning: Large batch sizes may cause CPU memory issues")
                        break
                except ValueError:
                    print("X Please enter a valid number")
            
            # Get user input for epochs
            while True:
                try:
                    epoch_input = input(f"Enter number of epochs (recommended: {rec['epochs']}, press Enter for default): ").strip()
                    if epoch_input == "":
                        epochs = rec['epochs']
                        break
                    else:
                        epochs = int(epoch_input)
                        if epochs < 1:
                            print("X Epochs must be at least 1")
                            continue
                        if epochs > 50:
                            print("! Warning: Training with >50 epochs may take a very long time")
                        break
                except ValueError:
                    print("X Please enter a valid number")
            
            # Display final configuration
            print(f"\n> Final Training Configuration:")
            print(f"   Method: {description}")
            print(f"   Batch Size: {batch_size}")
            print(f"   Epochs: {epochs}")
            
            # Estimate training time
            time_estimates = {
                'gpu': {
                    'base_time': 2,  # minutes per epoch
                    'batch_factor': 0.1  # additional minutes per batch size
                },
                'cpu': {
                    'base_time': 15,  # minutes per epoch
                    'batch_factor': 2  # additional minutes per batch size
                },
                'hybrid': {
                    'base_time': 6,  # minutes per epoch
                    'batch_factor': 0.5  # additional minutes per batch size
                }
            }
            
            est = time_estimates[method]
            estimated_time = (est['base_time'] + batch_size * est['batch_factor']) * epochs
            
            if estimated_time < 60:
                time_str = f"{estimated_time:.0f} minutes"
            else:
                hours = estimated_time // 60
                minutes = estimated_time % 60
                time_str = f"{hours:.0f}h {minutes:.0f}m"
            
            print(f"   Estimated Time: {time_str}")
            print("\nTraining will begin shortly...")
            print("You can stop training at any time with Ctrl+C")
            print("The model will be saved automatically during training.")
            
            confirm = input(f"\nStart {description} with these settings? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                # Pass parameters to the training script
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
        
        # Check for available models
        output_dir = os.path.join(self.base_dir, "05_OUTPUT_GENERATED")
        available_models = []
        
        model_files = [
            ("gpu_optimized_sasl_model.pth", "GPU-Optimized Model (FastCNNLSTM)"),
            ("hybrid_cpu_gpu_sasl_model.pth", "Hybrid CPU+GPU Model"),
            ("hand_focused_sasl_model.pth", "Hand-Focused Model"),
        ]
        
        for model_file, description in model_files:
            model_path = os.path.join(output_dir, model_file)
            if os.path.exists(model_path):
                available_models.append((model_file, description, model_path))
        
        if not available_models:
            print("No trained models found!")
            print("Please train a model first using option 2 (Training).")
            return
        
        print("Available models:")
        for i, (_, description, _) in enumerate(available_models, 1):
            print(f"{i}. {description}")
        print(f"{len(available_models) + 1}. Auto-detect best model (default)")
        print("0. Back to main menu")
        
        try:
            choice = input(f"\nSelect model (1-{len(available_models) + 1}, Enter for auto): ").strip()
            
            if choice == "0":
                return
            elif choice == "" or choice == str(len(available_models) + 1):
                # Auto-detect mode - use the enhanced camera's existing logic
                print("Using auto-detection mode...")
                self.run_script('enhanced_camera.py', 'Real-Time Recognition with Hand Tracking')
            elif choice.isdigit() and 1 <= int(choice) <= len(available_models):
                # Specific model selection
                selected_model = available_models[int(choice) - 1]
                model_file, description, model_path = selected_model
                
                print(f"Selected: {description}")
                print(f"Model: {model_file}")
                
                # Create a temporary script to run camera with specific model
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
        temp_script_content = f'''#!/usr/bin/env python3
"""
Temporary launcher for enhanced camera with specific model
"""
import os
import sys

# Set environment variable to force specific model
os.environ['SASL_FORCE_MODEL_PATH'] = r"{model_path}"

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the enhanced camera
from enhanced_camera import main

if __name__ == "__main__":
    print("Forced model selection: {description}")
    print("Model path: {model_path}")
    main()
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
        print("GPU Training: python 01_PRIMARY_SYSTEM/gpu_optimized_training.py")
        print("CPU Training: python 01_PRIMARY_SYSTEM/hand_focused_CNN_LSTM.py")
        print("Hybrid Training: python 01_PRIMARY_SYSTEM/hybrid_cpu_gpu_training.py")
        print("Camera: python 01_PRIMARY_SYSTEM/enhanced_camera.py")
        print("Demo: python 01_PRIMARY_SYSTEM/hand_tracking_demo.py")
        print("Benchmark: python simple_gpu_benchmark.py")
    
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
                            self.run_training_with_method('gpu')
                            break
                        elif train_choice == '2':
                            self.run_training_with_method('cpu')
                            break
                        elif train_choice == '3':
                            self.run_training_with_method('hybrid')
                            break
                        elif train_choice == '4':
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
