# This script converts a YOLO segmentation PyTorch model to CoreML and runs inference on a video
# CoreML runs on Apple Neural Engine for faster inference on Apple Silicon
# Draws only the mask border (no fill) with a green 4-pixel outline
#
# Usage:
#   python yolo_coreml_segment_video.py --model path/to/best.pt --input video.mp4 --display
#   python yolo_coreml_segment_video.py --model path/to/best.pt --input video.mp4 --output result.mp4

import cv2
import os
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Run YOLO CoreML segmentation inference on a video file')
parser.add_argument('--model', '-m', required=True, help='Path to YOLO segmentation model (.pt file, will convert to CoreML)')
parser.add_argument('--input', '-i', required=True, help='Input video file path')
parser.add_argument('--output', '-o', default=None, help='Output video path (optional, only saves if provided)')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
parser.add_argument('--display', '-d', action='store_true', help='Show live video preview while processing')
parser.add_argument('--border-thickness', '-b', type=int, default=4, help='Mask border thickness in pixels (default: 4)')
parser.add_argument('--no-text', action='store_true', help='Hide all text overlays (inference time, labels)')
args = parser.parse_args()

model_path = args.model
input_video_path = args.input
output_video_path = args.output
confidence_threshold = args.conf
display_video = args.display
border_thickness = args.border_thickness
no_text = args.no_text

# Magenta color (BGR format) - same as bounding box script
BORDER_COLOR = (255, 0, 255)

# Convert to CoreML if needed
model_path_obj = Path(model_path)
if model_path_obj.suffix == '.pt':
    coreml_path = model_path_obj.with_suffix('.mlpackage')
    if not coreml_path.exists():
        print(f"Converting {model_path} to CoreML...")
        pt_model = YOLO(model_path, task='segment')
        # Export with nms=False to preserve mask data
        pt_model.export(format='coreml', nms=False)
        print(f"CoreML model saved to: {coreml_path}")
    else:
        print(f"Using existing CoreML model: {coreml_path}")
    model_path = str(coreml_path)

print(f"Loading CoreML model: {model_path}")
model = YOLO(model_path, task='segment')


def draw_segmentation_borders(frame, masks, thickness=4, color=(0, 255, 0)):
    """Draw only the borders of segmentation masks on the frame.
    
    Args:
        frame: The original video frame (will be modified in-place)
        masks: The masks object from YOLO results
        thickness: Border line thickness in pixels
        color: BGR color tuple for the border
    
    Returns:
        The frame with mask borders drawn
    """
    if masks is None or masks.xy is None:
        return frame
    
    # Use masks.xy which provides polygon coordinates already in frame space
    for polygon in masks.xy:
        if len(polygon) > 0:
            # Convert to integer points for cv2
            pts = polygon.astype(np.int32).reshape((-1, 1, 2))
            # Draw only the polygon border (closed polyline)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    
    return frame


def process_video_file(input_video_path, output_video_path):
    """Process a single video file with YOLO CoreML segmentation inference"""
    
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
    total_inference_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference and measure time
        start_time = time.time()
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Draw segmentation mask borders (green, no fill)
        display_frame = frame.copy()
        for r in results:
            if r.masks is not None:
                draw_segmentation_borders(display_frame, r.masks, 
                                         thickness=border_thickness, 
                                         color=BORDER_COLOR)
                detection_count += len(r.masks.data)
            
            # Also draw confidence labels for each detection (unless --no-text)
            if not no_text and r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Draw label at top of the detection
                    label = f"seg: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), BORDER_COLOR, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw inference time on frame (unless --no-text)
        if not no_text:
            fps_text = f"Inference: {inference_time*1000:.1f}ms ({1/inference_time:.1f} FPS)"
            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BORDER_COLOR, 2)
        
        if out:
            out.write(display_frame)
        
        frame_count += 1
        
        if display_video:
            cv2.imshow("YOLO CoreML Segmentation", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if out:
        out.release()
    
    return frame_count, detection_count, total_inference_time


print(f"Processing: {input_video_path}")
if output_video_path:
    print(f"Output: {output_video_path}")
print(f"Confidence: {confidence_threshold}")
print(f"Border thickness: {border_thickness}px")

frame_count, detection_count, total_inference_time = process_video_file(input_video_path, output_video_path)

if display_video:
    cv2.destroyAllWindows()

avg_inference_time = total_inference_time / frame_count * 1000  # ms
avg_fps = frame_count / total_inference_time

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Frames: {frame_count}")
print(f"Segmentation detections: {detection_count}")
print(f"Avg detections/frame: {detection_count / frame_count:.2f}")
print(f"Avg inference: {avg_inference_time:.1f}ms ({avg_fps:.1f} FPS)")
if output_video_path:
    print(f"Output: {output_video_path}")
print(f"{'='*60}")
