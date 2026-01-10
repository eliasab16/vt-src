# This script draws bounding boxes on a recorded video using YOLOv11 local inference
#
# Usage:
#   python yolo11_video_inference.py --model path/to/best.pt --input path/to/input.mp4 --display
#   python yolo11_video_inference.py --model path/to/best.pt --input video.mp4 --output result.mp4

import cv2
import os
import time
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Process recorded video with YOLOv11 bounding box detection')
parser.add_argument('--model', '-m', required=True, help='Path to trained YOLO model (.pt file)')
parser.add_argument('--input', '-i', required=True, help='Input video file path')
parser.add_argument('--output', '-o', default=None, help='Output video file path (optional, only saves if provided)')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
parser.add_argument('--display', '-d', action='store_true', help='Show live video preview while processing')
args = parser.parse_args()

model_path = args.model
input_video_path = args.input
output_video_path = args.output
confidence_threshold = args.conf
display_video = args.display

print(f"Loading model: {model_path}")
model = YOLO(model_path)

def process_video_file(input_video_path, output_video_path):
    """Process a single video file with YOLOv11 bounding box detection"""
    
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps:.2f} fps")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv11 inference
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        
        # Draw bounding boxes (green)
        display_frame = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Draw green rectangle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"wire: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                detection_count += 1
        
        if out:
            out.write(display_frame)
        
        frame_count += 1
        
        if display_video:
            cv2.imshow("YOLOv11 Bounding Box Output", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if out:
        out.release()
    
    return frame_count, detection_count

print(f"Processing: {input_video_path}")
if output_video_path:
    print(f"Output: {output_video_path}")
print(f"Confidence: {confidence_threshold}")

start_time = time.time()
frame_count, detection_count = process_video_file(input_video_path, output_video_path)
end_time = time.time()

if display_video:
    cv2.destroyAllWindows()

processing_time = end_time - start_time
minutes = int(processing_time // 60)
seconds = processing_time % 60

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Frames: {frame_count}")
print(f"Detections: {detection_count}")
print(f"Avg detections/frame: {detection_count / frame_count:.2f}")
print(f"Time: {minutes}m {seconds:.1f}s")
if output_video_path:
    print(f"Output: {output_video_path}")
print(f"{'='*60}")
