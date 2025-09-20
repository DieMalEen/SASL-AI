#!/usr/bin/env python3
"""
SASL Video Data Collector
==========================

This tool helps you collect video sequences for SASL training by:
1. Recording video sequences of each sign (3-5 seconds each)
2. Real-time quality checking with MediaPipe
3. Automatic file organization
4. Preview and review recorded videos

Perfect for building a comprehensive SASL video dataset quickly!
"""

import cv2
import os
import numpy as np
import mediapipe as mp
from pathlib import Path
import time
from datetime import datetime
import json

class SASLVideoCollector:
    """
    Interactive video data collection tool for SASL signs
    """
    
    def __init__(self, dataset_path="video_dataset", recording_duration=3.0, fps=30, 
                 background_removal=True, bg_removal_method="transparent"):
        """
        Initialize the video collector
        
        Args:
            dataset_path: Path to save video dataset
            recording_duration: Duration of each video in seconds
            fps: Frames per second for recording
            background_removal: Enable background removal during recording
            bg_removal_method: Method for background removal ("solid_color", "blur", "transparent", "remove")
        """
        self.dataset_path = Path(dataset_path)
        self.recording_duration = recording_duration
        self.fps = fps
        self.current_class = None
        
        # Background removal settings
        self.background_removal = background_removal
        self.bg_removal_method = bg_removal_method
        self.background_color = (0, 255, 0)  # Green screen default
        
        # Create dataset directory
        self.dataset_path.mkdir(exist_ok=True)
        
        # Initialize MediaPipe for quality checking and segmentation
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize selfie segmentation for background removal
        if self.background_removal:
            self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 0 for general, 1 for landscape
            )
        
        # Load existing classes
        self.load_existing_classes()
        
        print(f"SASL Video Collector initialized:")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Recording duration: {recording_duration} seconds")
        print(f"  Target FPS: {fps}")
        print(f"  Background removal: {background_removal}")
        if background_removal:
            print(f"  Background method: {bg_removal_method}")
    
    def load_existing_classes(self):
        """Load existing SASL classes from dataset"""
        self.existing_classes = []
        if self.dataset_path.exists():
            self.existing_classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir() and d.name != '.gitkeep']
            self.existing_classes.sort()
        
        if self.existing_classes:
            print(f"\nFound {len(self.existing_classes)} existing SASL classes:")
            for i, class_name in enumerate(self.existing_classes[:10], 1):
                video_count = len([f for f in (self.dataset_path / class_name).glob("*.mp4")])
                print(f"  {i:2d}. {class_name} ({video_count} videos)")
            
            if len(self.existing_classes) > 10:
                print(f"  ... and {len(self.existing_classes) - 10} more classes")
        else:
            print("\nNo existing classes found. Ready to create new ones!")
    
    def apply_background_removal(self, frame):
        """
        Apply background removal to frame using MediaPipe selfie segmentation
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with background removed/modified
        """
        if not self.background_removal:
            return frame
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get segmentation mask
        results = self.selfie_segmentation.process(rgb_frame)
        
        # Create mask (1 for person, 0 for background)
        mask = results.segmentation_mask
        
        # Apply different background removal methods
        if self.bg_removal_method == "solid_color":
            # Replace background with solid color
            condition = mask > 0.5  # Threshold for person vs background
            background = np.full_like(frame, self.background_color, dtype=np.uint8)
            output_frame = np.where(condition[..., None], frame, background)
            
        elif self.bg_removal_method == "blur":
            # Blur the background
            condition = mask > 0.5
            blurred_background = cv2.GaussianBlur(frame, (21, 21), 0)
            output_frame = np.where(condition[..., None], frame, blurred_background)
            
        elif self.bg_removal_method == "transparent":
            # Create 4-channel image with alpha transparency
            condition = mask > 0.5
            alpha = (mask * 255).astype(np.uint8)
            output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            output_frame[..., 3] = alpha
            
        elif self.bg_removal_method == "remove":
            # Set background to black
            condition = mask > 0.5
            output_frame = np.where(condition[..., None], frame, 0)
            
        else:
            # Default: return original frame
            output_frame = frame
            
        return output_frame
    
    def toggle_background_removal_method(self):
        """Cycle through background removal methods"""
        methods = ["solid_color", "blur", "transparent", "remove"]
        current_idx = methods.index(self.bg_removal_method)
        next_idx = (current_idx + 1) % len(methods)
        self.bg_removal_method = methods[next_idx]
        
        method_names = {
            "solid_color": "Solid Color Background",
            "blur": "Blurred Background", 
            "transparent": "Transparent Background",
            "remove": "Black Background"
        }
        
        return method_names[self.bg_removal_method]
    
    def set_background_color(self, color):
        """Set background color for solid_color method"""
        self.background_color = color

    def check_sign_quality(self, frame):
        """
        Check if the frame contains good sign language data
        Returns quality score and feedback
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Check for hands and pose
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        quality_score = 0
        feedback = []
        
        # Hand detection quality (up to 60 points)
        if hand_results.multi_hand_landmarks:
            num_hands = len(hand_results.multi_hand_landmarks)
            quality_score += min(num_hands * 30, 60)
            feedback.append(f"GOOD: {num_hands} hand(s) detected")
            
            # Check hand visibility and spread
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                spread = np.std(landmarks) * 1000
                if spread > 40:
                    quality_score += 5
                    feedback.append("GOOD: Hand articulation")
        else:
            feedback.append("ERROR: No hands detected")
        
        # Pose detection quality (up to 25 points)
        if pose_results.pose_landmarks:
            quality_score += 15
            feedback.append("GOOD: Body pose detected")
            
            # Check upper body visibility
            upper_landmarks = pose_results.pose_landmarks.landmark[11:17]
            visible_count = sum(1 for lm in upper_landmarks if lm.visibility > 0.5)
            if visible_count >= 4:
                quality_score += 10
                feedback.append("GOOD: Upper body visible")
        else:
            feedback.append("ERROR: No body pose detected")
        
        # Lighting and contrast (up to 15 points)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        if contrast > 30:
            quality_score += 8
            feedback.append("GOOD: Good contrast")
        else:
            feedback.append("WARNING: Low contrast")
        
        if 50 < brightness < 200:
            quality_score += 7
            feedback.append("GOOD: Good lighting")
        else:
            feedback.append("WARNING: Check lighting")
        
        return min(quality_score, 100), feedback
    
    def draw_recording_overlay(self, frame, recording=False, countdown=None, quality_score=0, feedback=None, clean_recording=True):
        """Draw recording indicators and quality information on frame"""
        h, w = frame.shape[:2]
        
        # If clean_recording is True, don't draw anything - return clean frame
        if clean_recording and recording:
            return frame
        
        # Recording indicator
        if recording:
            # Red recording circle
            cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Countdown
        if countdown is not None:
            countdown_text = f"Recording in: {countdown:.1f}s"
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Background for countdown
            cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Quality bar
        bar_width = 300
        bar_height = 20
        bar_x = 20
        bar_y = 20
        
        # Background
        cv2.rectangle(frame, (bar_x-2, bar_y-2), (bar_x+bar_width+2, bar_y+bar_height+2), (0, 0, 0), -1)
        
        # Quality bar
        fill_width = int(bar_width * quality_score / 100)
        if quality_score > 80:
            color = (0, 255, 0)  # Green
        elif quality_score > 60:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Quality text
        cv2.putText(frame, f"Quality: {quality_score}%", (bar_x, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Feedback
        if feedback:
            y_offset = 60
            for line in feedback[:5]:
                cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        return frame
    
    def record_video_sequence(self, class_name, video_number):
        """
        Record a single video sequence for a SASL class
        
        Returns:
            bool: True if recording was successful
        """
        class_dir = self.dataset_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{video_number:03d}_{timestamp}.mp4"
        filepath = class_dir / filename
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nRecording setup:")
        print(f"  Class: {class_name}")
        print(f"  Video: {video_number}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {actual_fps}")
        print(f"  Duration: {self.recording_duration}s")
        print(f"  Output: {filename}")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filepath), fourcc, actual_fps, (width, height))
        
        # Recording states
        WAITING = 0
        RECORDING = 1
        FINISHED = 2
        
        state = WAITING
        recording_start = None
        frames_recorded = 0
        
        print("\nControls:")
        print("  SPACE - Start recording (always available, press again to stop and save)")
        print("  'r' - Restart (cancel current recording)")
        print("  'q' or ESC - Quit and close window immediately")
        print("  's' - Skip this video")
        print("  Quality score is shown for guidance, but recording is always allowed")
        print("  During recording: All UI elements are hidden for clean recording")
        print("  After recording: Choose whether to continue or stop")
        if self.background_removal:
            print("  'b' - Toggle background removal method")
            print("  'c' - Change background color (for solid color mode)")
            print(f"  Current background method: {self.bg_removal_method}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Apply background removal if enabled
            if self.background_removal:
                processed_frame = self.apply_background_removal(frame)
                # For transparency mode, convert back to BGR for display/saving
                if self.bg_removal_method == "transparent" and processed_frame.shape[2] == 4:
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR)
                else:
                    display_frame = processed_frame.copy()
            else:
                display_frame = frame.copy()
                processed_frame = frame.copy()
            
            # Check quality using original frame (before background removal)
            quality_score, feedback = self.check_sign_quality(frame)
            
            current_time = time.time()
            
            # State machine
            if state == WAITING:
                display_frame = self.draw_recording_overlay(display_frame, False, None, quality_score, feedback)
                
                # Instructions
                cv2.putText(display_frame, f"Class: {class_name} - Video {video_number}", 
                           (20, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Background removal info
                if self.background_removal:
                    cv2.putText(display_frame, f"Background: {self.bg_removal_method.title()}", 
                               (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show recording readiness message (quality score is informational only)
                cv2.putText(display_frame, "READY - Press SPACE to start recording", 
                           (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add quality guidance (but don't require it)
                if quality_score < 70:
                    cv2.putText(display_frame, "TIP: Better positioning improves quality", 
                               (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            elif state == RECORDING:
                elapsed = current_time - recording_start
                
                # Use clean recording mode - no overlays on display frame for clean visual
                display_frame = self.draw_recording_overlay(display_frame, True, None, quality_score, feedback, clean_recording=True)
                
                # No text overlays during recording for clean display
                # (All UI elements hidden during recording)
                
                # Write processed frame to video (with background removal if enabled)
                if self.bg_removal_method == "transparent":
                    # Convert BGRA back to BGR for video writing
                    save_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR) if processed_frame.shape[2] == 4 else processed_frame
                else:
                    save_frame = processed_frame
                    
                out.write(save_frame)
                frames_recorded += 1
            
            elif state == FINISHED:
                cv2.putText(display_frame, "Recording Complete!", 
                           (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press any key to continue", 
                           (width//2 - 120, height//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f'SASL Video Collection - {class_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if state == WAITING:
                if key == ord(' '):
                    # Start recording immediately (no quality threshold required)
                    state = RECORDING
                    recording_start = current_time
                    frames_recorded = 0
                    print(f"Recording started! Quality: {quality_score}% - All UI hidden for clean recording. Press SPACE again to stop and save.")
                elif key == ord('s'):
                    print("Skipping this video")
                    break
                elif key == ord('b') and self.background_removal:
                    new_method = self.toggle_background_removal_method()
                    print(f"Background removal method changed to: {new_method}")
                elif key == ord('c') and self.background_removal and self.bg_removal_method == "solid_color":
                    # Cycle through preset colors
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255)]
                    color_names = ["Green", "Blue", "Red", "Yellow", "Magenta", "Cyan", "Black", "White"]
                    try:
                        current_idx = colors.index(tuple(self.background_color))
                        next_idx = (current_idx + 1) % len(colors)
                    except ValueError:
                        next_idx = 0
                    self.set_background_color(colors[next_idx])
                    print(f"Background color changed to: {color_names[next_idx]}")
            elif state == RECORDING:
                if key == ord(' '):
                    # Stop recording and save
                    state = FINISHED
                    print(f"Recording stopped! Saved {frames_recorded} frames ({elapsed:.1f}s)")
                elif key == ord('r'):
                    state = WAITING
                    print("Recording cancelled and restarted")
            elif state == FINISHED:
                if key != 255:  # Any key pressed
                    break
            
            if key == ord('q') or key == 27:  # 'q' key or ESC key
                print("Quit requested - closing window")
                if filepath.exists():
                    filepath.unlink()
                    print("Incomplete recording deleted")
                cap.release()
                if 'out' in locals():
                    out.release()
                cv2.destroyAllWindows()
                return False
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if state == FINISHED:
            print(f"Video saved: {filename}")
            return True
        else:
            if filepath.exists():
                filepath.unlink()
            return False
    
    def collect_class_videos(self, class_name, target_count=10):
        """
        Collect multiple videos for a specific SASL class
        """
        class_dir = self.dataset_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Count existing videos
        existing_videos = list(class_dir.glob("*.mp4"))
        existing_count = len(existing_videos)
        
        print(f"\n{'='*50}")
        print(f"COLLECTING VIDEOS FOR: {class_name}")
        print(f"{'='*50}")
        print(f"Existing videos: {existing_count}")
        print(f"Target: {target_count} total videos")
        print(f"Need to record: {max(0, target_count - existing_count)} more videos")
        
        if existing_count >= target_count:
            print(f"Target already reached for '{class_name}'!")
            return existing_count
        
        # Record new videos
        video_number = existing_count + 1
        recorded_count = 0
        
        while existing_count + recorded_count < target_count:
            print(f"\nPreparing to record video {video_number} of {target_count}")
            
            success = self.record_video_sequence(class_name, video_number)
            
            if success:
                recorded_count += 1
                video_number += 1
                print(f"Progress: {existing_count + recorded_count}/{target_count} videos")
                
                # Check if user wants to continue
                if existing_count + recorded_count < target_count:
                    print(f"\nVideo {existing_count + recorded_count}/{target_count} recorded successfully!")
                    continue_choice = input("Continue recording? (y/n/q): ").strip().lower()
                    if continue_choice in ['n', 'no']:
                        print("Recording session ended by user")
                        break
                    elif continue_choice in ['q', 'quit']:
                        print("Quitting video collection")
                        break
                    # Default 'y' or any other key continues
            else:
                print("Recording failed or cancelled - ending session")
                break
        
        final_count = existing_count + recorded_count
        print(f"\nFinal count for '{class_name}': {final_count} videos")
        return final_count
    
    def interactive_menu(self):
        """
        Interactive menu for video collection
        """
        print("\n" + "="*60)
        print("SASL VIDEO DATA COLLECTOR")
        print("="*60)
        print(f"Dataset location: {self.dataset_path}")
        print(f"Current classes: {len(self.existing_classes)}")
        
        while True:
            print(f"\nOptions:")
            print(f"1. Record videos for existing class")
            print(f"2. Create new class and record videos")
            print(f"3. Batch recording (all classes)")
            print(f"4. Preview existing videos")
            print(f"5. Show dataset statistics")
            print(f"6. Delete class")
            print(f"7. Quit")
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                self.record_existing_class()
            elif choice == '2':
                self.create_new_class()
            elif choice == '3':
                self.batch_recording()
            elif choice == '4':
                self.preview_videos()
            elif choice == '5':
                self.show_statistics()
            elif choice == '6':
                self.delete_class()
            elif choice == '7':
                print("Quitting SASL Video Collector...")
                cv2.destroyAllWindows()  # Close any remaining windows
                break
            else:
                print("Invalid choice. Please enter 1-7.")
        
        print("Video collection complete!")
    
    def record_existing_class(self):
        """Record videos for an existing class"""
        if not self.existing_classes:
            print("No existing classes found!")
            return
        
        print(f"\nExisting classes:")
        for i, class_name in enumerate(self.existing_classes, 1):
            video_count = len(list((self.dataset_path / class_name).glob("*.mp4")))
            print(f"  {i:2d}. {class_name} ({video_count} videos)")
        
        try:
            choice = int(input(f"\nSelect class (1-{len(self.existing_classes)}): "))
            if 1 <= choice <= len(self.existing_classes):
                class_name = self.existing_classes[choice - 1]
                target = int(input(f"Target number of videos for '{class_name}' (default 15): ") or "15")
                self.collect_class_videos(class_name, target)
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")
    
    def create_new_class(self):
        """Create a new SASL class and record videos"""
        class_name = input("Enter new SASL class name: ").strip()
        if not class_name:
            print("Class name cannot be empty!")
            return
        
        if class_name in self.existing_classes:
            print(f"Class '{class_name}' already exists!")
            return
        
        target = int(input(f"Target number of videos for '{class_name}' (default 15): ") or "15")
        final_count = self.collect_class_videos(class_name, target)
        
        if final_count > 0:
            self.existing_classes.append(class_name)
            self.existing_classes.sort()
            print(f"Created new class '{class_name}' with {final_count} videos")
    
    def batch_recording(self):
        """Record videos for all classes in batch"""
        if not self.existing_classes:
            print("No existing classes found!")
            return
        
        target = int(input("Target videos per class (default 15): ") or "15")
        
        for class_name in self.existing_classes:
            current_count = len(list((self.dataset_path / class_name).glob("*.mp4")))
            if current_count < target:
                print(f"\n{'='*50}")
                print(f"PROCESSING CLASS: {class_name}")
                print(f"Current: {current_count}, Target: {target}")
                print(f"{'='*50}")
                self.collect_class_videos(class_name, target)
            else:
                print(f"Skipping '{class_name}' - already has {current_count} videos")
    
    def preview_videos(self):
        """Preview existing videos"""
        if not self.existing_classes:
            print("No classes found!")
            return
        
        print(f"\nSelect class to preview:")
        for i, class_name in enumerate(self.existing_classes, 1):
            video_count = len(list((self.dataset_path / class_name).glob("*.mp4")))
            print(f"  {i:2d}. {class_name} ({video_count} videos)")
        
        try:
            choice = int(input(f"\nSelect class (1-{len(self.existing_classes)}): "))
            if 1 <= choice <= len(self.existing_classes):
                class_name = self.existing_classes[choice - 1]
                self.preview_class_videos(class_name)
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")
    
    def preview_class_videos(self, class_name):
        """Preview videos for a specific class"""
        class_dir = self.dataset_path / class_name
        video_files = list(class_dir.glob("*.mp4"))
        
        if not video_files:
            print(f"No videos found for class '{class_name}'")
            return
        
        print(f"\nPreviewing {len(video_files)} videos for '{class_name}'")
        print("Controls: SPACE=next video, 'q'=quit preview")
        
        for video_file in video_files:
            print(f"Playing: {video_file.name}")
            
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"Could not open {video_file.name}")
                continue
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                
                cv2.imshow(f'Preview: {class_name} - {video_file.name}', frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):  # Next video
                    break
                elif key == ord('q'):  # Quit preview
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            cap.release()
        
        cv2.destroyAllWindows()
    
    def show_statistics(self):
        """Show dataset statistics"""
        print(f"\n{'='*60}")
        print("SASL VIDEO DATASET STATISTICS")
        print("="*60)
        
        total_videos = 0
        total_duration = 0
        class_stats = []
        
        for class_name in self.existing_classes:
            video_files = list((self.dataset_path / class_name).glob("*.mp4"))
            count = len(video_files)
            duration = count * self.recording_duration
            
            total_videos += count
            total_duration += duration
            class_stats.append((class_name, count, duration))
        
        class_stats.sort(key=lambda x: x[1])  # Sort by video count
        
        print(f"Total classes: {len(self.existing_classes)}")
        print(f"Total videos: {total_videos}")
        print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"Average videos per class: {total_videos/max(len(self.existing_classes), 1):.1f}")
        
        print(f"\nClass breakdown:")
        for class_name, count, duration in class_stats:
            print(f"  {class_name}: {count} videos ({duration:.1f}s)")
    
    def delete_class(self):
        """Delete a class and all its videos"""
        if not self.existing_classes:
            print("No classes found!")
            return
        
        print(f"\nExisting classes:")
        for i, class_name in enumerate(self.existing_classes, 1):
            video_count = len(list((self.dataset_path / class_name).glob("*.mp4")))
            print(f"  {i:2d}. {class_name} ({video_count} videos)")
        
        try:
            choice = int(input(f"\nSelect class to DELETE (1-{len(self.existing_classes)}): "))
            if 1 <= choice <= len(self.existing_classes):
                class_name = self.existing_classes[choice - 1]
                
                confirm = input(f"DELETE ALL videos for '{class_name}'? Type 'DELETE' to confirm: ")
                if confirm == 'DELETE':
                    class_dir = self.dataset_path / class_name
                    if class_dir.exists():
                        import shutil
                        shutil.rmtree(class_dir)
                        self.existing_classes.remove(class_name)
                        print(f"Deleted class '{class_name}' and all its videos")
                    else:
                        print(f"Class directory not found")
                else:
                    print("Deletion cancelled")
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")

def main():
    """Main function"""
    print("SASL Video Data Collector with Background Removal")
    print("Build your SASL video dataset efficiently!")
    
    # Ask user about background removal preferences
    print("\nBackground Removal Options:")
    print("1. Enable background removal (recommended for better training)")
    print("2. Disable background removal (record with original background)")
    
    choice = input("Choose option (1/2) [1]: ").strip() or "1"
    background_removal = choice == "1"
    
    bg_method = "transparent"  # Default method - transparent background
    if background_removal:
        print("\nBackground Removal Methods:")
        print("1. Solid color background (green screen effect)")
        print("2. Blurred background (background stays but blurred)")
        print("3. Black background (background becomes black)")
        print("4. Transparent background (default - for advanced editing)")
        
        method_choice = input("Choose method (1-4) [4]: ").strip() or "4"
        methods = ["solid_color", "blur", "remove", "transparent"]
        bg_method = methods[int(method_choice) - 1]
    
    # Initialize collector with background removal settings
    collector = SASLVideoCollector(
        dataset_path="video_dataset",
        recording_duration=3.0,  # 3-second videos
        fps=30,
        background_removal=background_removal,
        bg_removal_method=bg_method
    )
    
    if background_removal:
        print(f"\n✅ Background removal enabled with '{bg_method}' method")
        print("During recording:")
        print("  - Press 'b' to switch between removal methods")
        print("  - Press 'c' to change background color (solid color mode)")
        print("  - Background removal helps models focus on sign language gestures")
    else:
        print("\n❌ Background removal disabled - recording with original background")
    
    # Run interactive menu
    collector.interactive_menu()
    
    print("\nDataset ready for video-based training!")
    print("Use 'python video_based_sasl_training.py' to train your models.")

if __name__ == "__main__":
    main()