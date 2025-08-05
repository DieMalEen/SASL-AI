# Suppress MediaPipe verbose logging (must be before any imports)
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress MediaPipe specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")

import cv2
import numpy as np
import sys
from hand_detection import HandDetector
import time

def demonstrate_hand_tracking(video_path, output_path=None, show_video=True):
    """
    Demonstrate hand tracking on a video file
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        show_video: Whether to display video in real-time
    """
    print(f"Processing video: {video_path}")
    
    # Initialize hand detector
    hand_detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path is provided
    if output_path:
        # Try different codecs for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Warning: Could not initialize video writer with XVID codec")
            print(f"Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"Error: Could not initialize video writer. Video will not be saved.")
                output_path = None  # Disable video saving
    
    frame_count = 0
    hands_detected_count = 0
    processing_times = []
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Detect hands
        hands_data = hand_detector.detect_hands(frame)
        
        # Draw hands on frame
        annotated_frame = hand_detector.draw_hands(
            frame, hands_data, 
            draw_landmarks=True, 
            draw_bbox=True
        )
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        if hands_data:
            hands_detected_count += 1
            
            # Add hand information text
            for i, hand_data in enumerate(hands_data):
                info_text = f"Hand {i+1}: {hand_data['hand_label']} ({hand_data['confidence']:.2f})"
                y_pos = 30 + i * 30
                cv2.putText(annotated_frame, info_text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame, info_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        detection_rate = (hands_detected_count / frame_count) * 100
        rate_text = f"Hand Detection Rate: {detection_rate:.1f}%"
        cv2.putText(annotated_frame, rate_text, (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame if requested
        if show_video:
            cv2.imshow('Hand Tracking Demo', annotated_frame)
            
            # Control playback speed
            key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to pause
                cv2.waitKey(0)
        
        # Write frame to output video
        if output_path:
            out.write(annotated_frame)
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            avg_processing_time = np.mean(processing_times[-30:])
            print(f"Progress: {progress:.1f}% - Avg processing time: {avg_processing_time:.3f}s")
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    hand_detector.close()
    
    # Print summary
    print("\n" + "="*50)
    print("HAND TRACKING SUMMARY")
    print("="*50)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with hands detected: {hands_detected_count}")
    print(f"Hand detection rate: {(hands_detected_count/frame_count)*100:.1f}%")
    print(f"Average processing time per frame: {np.mean(processing_times):.3f}s")
    print(f"Estimated FPS: {1/np.mean(processing_times):.1f}")
    
    if output_path:
        print(f"Output video saved to: {output_path}")

def extract_hand_regions_demo(video_path, output_dir="hand_regions_demo"):
    """
    Extract and save hand regions from video frames
    """
    print(f"Extracting hand regions from: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    hand_detector = HandDetector()
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    hand_regions_saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 10th frame to avoid too many images
        if frame_count % 10 != 0:
            continue
        
        # Detect hands
        hands_data = hand_detector.detect_hands(frame)
        
        if hands_data:
            # Extract hand regions
            hand_regions = hand_detector.extract_hand_regions(frame, hands_data)
            
            for i, region_data in enumerate(hand_regions):
                hand_region = region_data['region']
                hand_label = region_data['hand_label']
                confidence = region_data['confidence']
                
                # Save hand region
                filename = f"frame_{frame_count:04d}_hand_{i}_{hand_label}_{confidence:.2f}.jpg"
                filepath = os.path.join(output_dir, filename)
                success = cv2.imwrite(filepath, hand_region)
                if success:
                    hand_regions_saved += 1
    
    cap.release()
    hand_detector.close()
    
    print(f"Extracted {hand_regions_saved} hand regions to {output_dir}")

def find_sample_video():
    """
    Find a sample video from the dataset
    """
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print("Dataset folder not found!")
        return None
    
    # Look for videos in dataset folders
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith('.mp4'):
                    video_path = os.path.join(class_path, file)
                    print(f"Found sample video: {video_path}")
                    return video_path
    
    print("No video files found in dataset!")
    return None

def main():
    """
    Main demo function
    """
    print("SASL Hand Detection Demo")
    print("="*40)
    
    # Try to install mediapipe if not available
    try:
        import mediapipe as mp
        print("MediaPipe is available!")
    except ImportError:
        print("MediaPipe not found. Installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mediapipe'])
            print("MediaPipe installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install MediaPipe. Please install it manually:")
            print("pip install mediapipe")
            return
    
    # Find a sample video
    sample_video = find_sample_video()
    
    if not sample_video:
        print("Please provide a video path manually")
        return
    
    # Choose what to demonstrate
    print("\nChoose demo option:")
    print("1. Show hand tracking video")
    print("2. Save hand tracking video")
    print("3. Extract hand regions")
    print("4. All of the above")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    if choice in ['1', '4']:
        print("\nStarting real-time hand tracking demo...")
        print("Press 'q' to quit, SPACE to pause")
        demonstrate_hand_tracking(sample_video, show_video=True)
    
    if choice in ['2', '4']:
        # Create output directory using absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
        os.makedirs(output_dir, exist_ok=True)
        output_video = os.path.join(output_dir, f"hand_tracking_demo_{os.path.basename(sample_video)}")
        print(f"\nSaving hand tracking video to {output_video}...")
        demonstrate_hand_tracking(sample_video, output_path=output_video, show_video=False)
    
    if choice in ['3', '4']:
        print("\nExtracting hand regions...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "05_OUTPUT_GENERATED")
        os.makedirs(output_dir, exist_ok=True)
        hand_regions_dir = os.path.join(output_dir, "hand_regions_demo")
        extract_hand_regions_demo(sample_video, output_dir=hand_regions_dir)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
