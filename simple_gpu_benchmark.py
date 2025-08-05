#!/usr/bin/env python3
"""
Simple GPU vs CPU Benchmark for SASL Training
Tests basic PyTorch operations to estimate performance gains
"""

import torch
import torch.nn as nn
import time
import json
import os

def simple_benchmark():
    """Simple benchmark without torchvision dependencies"""
    print("Simple GPU vs CPU Benchmark for SASL Training")
    
    # Check system capabilities
    print(f"\nSystem Information:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Simple CNN-like operations
    def create_test_model(device):
        """Create a simple model for testing"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 20)
        ).to(device)
        return model
    
    def test_device(device_name, batch_size=4, num_iterations=100):
        """Test performance on a specific device"""
        device = torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu')
        print(f"\n🔬 Testing {device_name.upper()}...")
        
        model = create_test_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create test data (simulating video frames)
        test_data = torch.randn(batch_size, 3, 224, 224, device=device)
        test_labels = torch.randint(0, 20, (batch_size,), device=device)
        
        # Warmup
        for _ in range(10):
            outputs = model(test_data)
            loss = criterion(outputs, test_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Actual test
        start_time = time.time()
        
        for i in range(num_iterations):
            outputs = model(test_data)
            loss = criterion(outputs, test_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 20 == 0:
                print(f"   Iteration {i}/{num_iterations}")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time_per_iteration = total_time / num_iterations
        
        memory_usage = "N/A"
        if device.type == 'cuda':
            memory_usage = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            torch.cuda.reset_peak_memory_stats()
        
        return {
            'device': device_name,
            'total_time': total_time,
            'avg_time_per_iteration': avg_time_per_iteration,
            'iterations_per_second': num_iterations / total_time,
            'memory_usage': memory_usage
        }
    
    # Run tests
    results = []
    
    # Test CPU
    cpu_result = test_device('cpu', batch_size=4, num_iterations=50)
    results.append(cpu_result)
    
    # Test GPU if available
    if torch.cuda.is_available():
        gpu_result = test_device('cuda', batch_size=4, num_iterations=50)
        results.append(gpu_result)
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\n   {result['device'].upper()} Performance:")
        print(f"   Total Time: {result['total_time']:.2f} seconds")
        print(f"   Avg Time/Iteration: {result['avg_time_per_iteration']:.4f} seconds")
        print(f"   Iterations/Second: {result['iterations_per_second']:.1f}")
        print(f"   Memory Usage: {result['memory_usage']}")
    
    # Calculate speedup
    if len(results) > 1:
        cpu_result = next(r for r in results if r['device'] == 'cpu')
        gpu_result = next(r for r in results if r['device'] == 'cuda')
        
        speedup = cpu_result['avg_time_per_iteration'] / gpu_result['avg_time_per_iteration']
        print(f"\n   GPU SPEEDUP: {speedup:.1f}x faster than CPU")
        
        # Provide detailed recommendations
        print(f"\n   TRAINING RECOMMENDATIONS:")
        if speedup > 8.0:
            print(f"   HIGHLY RECOMMENDED: GPU Training")
            print(f"   Your GPU provides excellent {speedup:.1f}x speedup!")
            print(f"   Expected training time: ~1-2 hours instead of 8-15 hours")
            print(f"   Use: GPU Training option in main menu")
        elif speedup > 3.0:
            print(f"   RECOMMENDED: GPU Training")  
            print(f"   Your GPU provides good {speedup:.1f}x speedup")
            print(f"   Expected training time: ~2-4 hours instead of 8-15 hours")
            print(f"   Use: GPU Training option in main menu")
        elif speedup > 1.5:
            print(f"   SUGGESTED: Hybrid Training")
            print(f"   Moderate {speedup:.1f}x GPU speedup")
            print(f"   Hybrid mode balances CPU and GPU workload")
            print(f"   Use: Hybrid CPU+GPU Training option")
        else:
            print(f"   SUGGESTED: CPU Training")
            print(f"   Limited GPU benefit ({speedup:.1f}x speedup)")
            print(f"   CPU training may be more stable")
    
    print("="*60)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if script_dir.endswith('01_PRIMARY_SYSTEM') else script_dir
    output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "simple_benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Final recommendation summary
    if torch.cuda.is_available() and len(results) > 1:
        cpu_result = next(r for r in results if r['device'] == 'cpu')
        gpu_result = next(r for r in results if r['device'] == 'cuda')
        speedup = cpu_result['avg_time_per_iteration'] / gpu_result['avg_time_per_iteration']
        
        print(f"\nQUICK RECOMMENDATION:")
        if speedup > 8.0:
            print(f"   Use GPU Training (Menu Option 2 → 1)")
        elif speedup > 3.0:
            print(f"   Use GPU Training (Menu Option 2 → 1)")
        elif speedup > 1.5:
            print(f"   Use Hybrid Training (Menu Option 2 → 3)")
        else:
            print(f"   Use CPU Training (Menu Option 2 → 2)")
    else:
        print(f"\nQUICK RECOMMENDATION:")
        print(f"   Use CPU Training (Menu Option 2 → 2)")
    
    return results

if __name__ == '__main__':
    simple_benchmark()
