import sys
import pkg_resources
print(sys.executable)  # Should show your cv_env path
import pkg_resources
for pkg in ['numpy', 'opencv-python', 'ultralytics']:
    try:
        print(f"{pkg}: {pkg_resources.get_distribution(pkg).version}")
    except:
        print(f"{pkg}: Not installed")

import  cv2
print(cv2.__version__)

import yolov5

# Load pretrained YOLOv5s model
model = yolov5.load('yolov5s.pt')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yolov5.models.common")

import warnings
import yolov5
import cv2
import numpy as np
from datetime import timedelta

# Suppress all warnings
warnings.filterwarnings("ignore")

def process_video(video_path, process_duration=None):
    """Process video with optional duration limit"""
    
    # Load YOLOv5s model
    model = yolov5.load('yolov5s.pt')
    
    # Set model parameters for vehicle detection
    model.conf = 0.4  # Confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = [2, 5, 7]  # Car (2), bus (5), truck (7) in COCO dataset

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate max frames if duration specified
    max_frames = int(fps * process_duration) if process_duration else None
    
    # Create output filename based on processing choice
    output_video = ('first_'+str(process_duration)+'sec_detection.avi' 
                    if process_duration else 'full_video_detection.avi')
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Detection log
    detection_log = []
    frame_count = 0
    
    try:
        while cap.isOpened():
            # Check if we've reached the requested duration
            if max_frames and frame_count >= max_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current timestamp (MM:SS)
            current_time_sec = frame_count / fps
            timestamp = str(timedelta(seconds=current_time_sec)).split(".")[0]
            
            # Create writable copy
            frame_copy = np.ascontiguousarray(frame.copy())
            
            # Convert to RGB for YOLOv5
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
            # Perform inference
            results = model(frame_rgb)
            
            # Get all detections
            detections = results.pandas().xyxy[0]
            
            # Log all detections with timestamp
            for _, detection in detections.iterrows():
                detection_log.append({
                    'timestamp': timestamp,
                    'frame': frame_count,
                    'class_id': int(detection['class']),
                    'class_name': detection['name'],
                    'confidence': float(detection['confidence']),
                    'xmin': float(detection['xmin']),
                    'ymin': float(detection['ymin']),
                    'xmax': float(detection['xmax']),
                    'ymax': float(detection['ymax'])
                })
            
            # Create display frame with all detections
            display_frame = frame_copy.copy()
            if len(results.xyxy[0]) > 0:
                results.ims = [np.ascontiguousarray(frame_rgb.copy())]
                rendered = results.render()[0]
                display_frame = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            
            # Add timestamp
            cv2.putText(display_frame, timestamp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(display_frame)
            
            frame_count += 1
            
            # Print progress every second
            if frame_count % int(fps) == 0:
                print(f"Processed {frame_count//int(fps)} seconds ({frame_count} frames)")
    
    except Exception as e:
        print(f"Error processing frame {frame_count}: {str(e)}")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        
        # Create output log filename
        log_filename = ('first_'+str(process_duration)+'sec_detection_log.txt' 
                       if process_duration else 'full_video_detection_log.txt')
        
        # Save detailed classification log
        with open(log_filename, 'w') as f:
            # Write header
            f.write("Timestamp | Frame | ClassID | ClassName | Confidence | BBox Coordinates (xmin,ymin,xmax,ymax)\n")
            f.write("-" * 100 + "\n")
            
            # Write each detection
            for log in detection_log:
                f.write(
                    f"{log['timestamp']:<10} | "
                    f"{log['frame']:<6} | "
                    f"{log['class_id']:<8} | "
                    f"{log['class_name']:<10} | "
                    f"{log['confidence']:.4f} | "
                    f"[{log['xmin']:.1f},{log['ymin']:.1f},{log['xmax']:.1f},{log['ymax']:.1f}]\n"
                )
        
        # Print summary
        print("\nProcessing complete. Output files created:")
        print(f"- {output_video} (annotated video)")
        print(f"- {log_filename} (detailed detection log)")
        
        unique_classes = set(log['class_name'] for log in detection_log)
        print("\nDetected classes:", ", ".join(unique_classes))
        print(f"Total detections: {len(detection_log)}")
        print(f"Total duration processed: {frame_count/fps:.2f} seconds")

 # Main program
if __name__ == "__main__":
    video_path = 'cvtest.avi'
    
    print("Video Processing Options:")
    print("1. Process entire video")
    print("2. Process first N seconds")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        print("\nProcessing entire video...")
        process_video(video_path)
    elif choice == '2':
        try:
            n_seconds = int(input("Enter number of seconds to process: "))
            print(f"\nProcessing first {n_seconds} seconds...")
            process_video(video_path, n_seconds)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    else:
        print("Invalid choice. Please enter 1 or 2.")       