#!/usr/bin/env python3
"""
Test script for CNN-only training configuration
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_configuration():
    """Test the configuration function"""
    try:
        from main import configure_cnn_training_parameters
        
        print("Testing CNN-only training configuration...")
        print("Note: This will prompt for input. Use Enter for defaults.")
        
        config = configure_cnn_training_parameters()
        
        if config:
            print(f"\n✓ Configuration successful!")
            print(f"Configuration received: {config}")
            
            # Validate configuration
            required_keys = ['epochs', 'batch_size', 'augmentation_factor', 'learning_rate']
            if all(key in config for key in required_keys):
                print("✓ All required parameters present")
                
                # Validate ranges
                if config['epochs'] > 0:
                    print("✓ Epochs valid")
                if config['batch_size'] > 0:
                    print("✓ Batch size valid")
                if 0 <= config['augmentation_factor'] <= 3:
                    print("✓ Augmentation factor valid")
                if config['learning_rate'] > 0:
                    print("✓ Learning rate valid")
                
                print("\n✅ Configuration test passed!")
            else:
                print("❌ Missing required parameters")
        else:
            print("ℹ️ Configuration cancelled by user")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_configuration()