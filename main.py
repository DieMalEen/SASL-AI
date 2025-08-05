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
                                print("🚀 HIGHLY RECOMMENDED: GPU Training")
                                print(f"   Your GPU is {speedup:.1f}x faster than CPU!")
                                print("   Use Option 1 (GPU Training) for best performance")
                            elif speedup > 2.0:
                                print("✅ RECOMMENDED: GPU Training")  
                                print(f"   Your GPU is {speedup:.1f}x faster than CPU")
                                print("   Use Option 1 (GPU Training)")
                            else:
                                print("💡 SUGGESTED: Hybrid Training")
                                print("   Moderate GPU speedup - try Option 3 (Hybrid)")
                        else:
                            print("📊 Use CPU Training (Option 2)")
                    else:
                        print("📊 No GPU detected - Use CPU Training (Option 2)")
                        
                except Exception as e:
                    print(f"Could not read benchmark results: {e}")
                    print("📊 Check the benchmark output above for recommendations")
            else:
                print("📊 Check the benchmark output above for recommendations")
                
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
        
        script_name = training_scripts.get(method)
        description = descriptions.get(method)
        
        if script_name:
            print(f"\n{description}")
            print("=" * 60)
            print("Training will begin shortly...")
            print("You can stop training at any time with Ctrl+C")
            print("The model will be saved automatically during training.")
            
            confirm = input(f"\nStart {description}? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                self.run_script(script_name, description)
            else:
                print("Training cancelled.")
        else:
            print("Invalid training method selected.")
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
            status = "✓" if os.path.exists(dir_path) else "✗"
            print(f"{status} {desc}: {dir_name}")
            
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.py')]
                if files:
                    for file in files[:3]:  # Show first 3 files
                        print(f"    {file}")
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more files")
        
        # Check key files
        print(f"\nDataset: {'✓' if os.path.exists('dataset') else '✗'}")
        
        # Check GPU availability
        try:
            import torch
            print(f"PyTorch: ✓ (v{torch.__version__})")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU: ✓ {gpu_name} ({gpu_memory:.1f}GB)")
                print("   Recommended: GPU Training")
            else:
                print("GPU: ✗ (CUDA not available)")
                print("   Recommended: CPU Training")
        except ImportError:
            print("PyTorch: ✗ (Not installed)")
        
        # Check dependencies
        try:
            import cv2, mediapipe
            print("OpenCV & MediaPipe: ✓")
        except ImportError as e:
            print(f"Missing dependencies: {e}")
    
    def show_manual_fallback(self):
        """Show manual commands if interactive guide fails"""
        print("\nManual Commands:")
        print("=" * 30)
        print("GPU Training: python 01_PRIMARY_SYSTEM/gpu_optimized_training.py")
        print("CPU Training: python 01_PRIMARY_SYSTEM/hand_focused_CNN_LSTM.py")
        print("Hybrid Training: python 01_PRIMARY_SYSTEM/hybrid_cpu_gpu_training.py")
        print("Camera: python 01_PRIMARY_SYSTEM/enhanced_camera.py")
        print("GPU Camera: python 01_PRIMARY_SYSTEM/gpu_optimized_camera.py")
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
                    # Enhanced camera menu
                    print("\nCamera Options")
                    print("=" * 30)
                    print("1. Standard Camera (CPU)")
                    print("2. GPU-Optimized Camera (Faster)")
                    camera_choice = input("Choose camera type (1/2): ").strip()
                    
                    if camera_choice == '1':
                        self.run_script('enhanced_camera.py', 'Real-Time Recognition with Hand Tracking')
                    elif camera_choice == '2':
                        self.run_script('gpu_optimized_camera.py', 'GPU-Optimized Real-Time Recognition')
                    else:
                        print("Invalid choice.")
                
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
