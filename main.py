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
        print("1. Train Hand-Focused CNN+LSTM Model (RECOMMENDED)")
        print("2. Real-Time Recognition with Hand Tracking") 
        print("3. Video Analysis & Hand Tracking Demo")
        print("4. System Information")
        print("q. Quit")
        print()
    
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
    
    def show_system_info(self):
        """Show system information and file structure"""
        print("\n📁 SASL System Information")
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
            status = "Y" if os.path.exists(dir_path) else "N"
            print(f"{status} {desc}: {dir_name}")
            
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.endswith('.py')]
                if files:
                    for file in files[:3]:  # Show first 3 files
                        print(f"    {file}")
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more files")
        
        # Check key files
        print(f"\nDataset: {'Y' if os.path.exists('dataset') else 'N'}")
        
        # Check dependencies
        try:
            import torch, cv2, mediapipe
            print("Key dependencies installed")
        except ImportError as e:
            print(f"Missing dependencies: {e}")
    
    def show_manual_fallback(self):
        """Show manual commands if interactive guide fails"""
        print("\n🔧 Manual Commands:")
        print("=" * 30)
        print("Training: python 01_PRIMARY_SYSTEM/hand_focused_CNN_LSTM.py")
        print("Camera: python 01_PRIMARY_SYSTEM/enhanced_camera.py")
        print("Demo: python 01_PRIMARY_SYSTEM/hand_tracking_demo.py")
        print("Basic Camera: python 02_FALLBACK_COMPATIBILITY/camera.py")
    
    def run(self):
        """Main run loop"""
        while True:
            try:
                self.show_menu()
                choice = input("Enter your choice: ").strip().lower()
                
                if choice == '1':
                    self.run_script('hand_focused_CNN_LSTM.py', 'Training Hand-Focused CNN+LSTM Model')
                
                elif choice == '2':
                    self.run_script('enhanced_camera.py', 'Real-Time Recognition with Hand Tracking')
                
                elif choice == '3':
                    self.run_script('hand_tracking_demo.py', 'Video Analysis & Hand Tracking Demo')
                
                elif choice == '4':
                    self.show_system_info()
                
                elif choice in ['q', 'quit', 'exit']:
                    print("\nThank you for using SASL Hand Detection System!")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
                
                if choice not in ['4']:  # Don't pause for info screens
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\nThank you for using SASL Hand Detection System!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    launcher = SASLLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
