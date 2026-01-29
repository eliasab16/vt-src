#!/usr/bin/env python3
"""
CUDA-accelerated YOLO segmentation for video annotation
For use on hosted GPU environments (vast.ai, etc.)

Usage:
    python yolo_cuda_segment_video.py --model best.pt --input video.mp4 --output segmented.mp4
"""

import cv2
import os
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

parser = argparse.ArgumentParser(description='CUDA YOLO segmentation inference')
parser.add_argument('--model', '-m', required=True, help='Path to YOLO segmentation .pt model')
parser.add_argument('--input', '-i', required=True, help='Input video file path')
parser.add_argument('--output', '-o', required=True, help='Output video path')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--border-thickness', '-b', type=int, default=4, help='Mask border thickness')
parser.add_argument('--no-text', action='store_true', help='Hide text overlays')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Magenta color (BGR format)
BORDER_COLOR = (255, 0, 255)

# Load model
print(f"Loading YOLO model: {args.model}")
model = YOLO(args.model, task='segment')
model.to(DEVICE)

def draw_segmentation_borders(frame, masks, thickness=4, color=(255, 0, 255)):
    """Draw only the borders of segmentation masks on the frame."""
    if masks is None or masks.xy is None:
        return frame
    
    for polygon in masks.xy:
        if len(polygon) > 0:
            pts = polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    
    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps:.2f} fps ({total_frames} frames)")
    
    frame_count = 0
    detection_count = 0
    total_time = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        results = model.predict(frame, conf=args.conf, verbose=False, device=DEVICE)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        display_frame = frame.copy()
        for r in results:
            if r.masks is not None:
                draw_segmentation_borders(display_frame, r.masks, 
                                         thickness=args.border_thickness, 
                                         color=BORDER_COLOR)
                detection_count += len(r.masks.data)
        
        if not args.no_text:
            fps_text = f"Frame {frame_count}/{total_frames} | {inference_time*1000:.1f}ms"
            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BORDER_COLOR, 2)
        
        out.write(display_frame)
        frame_count += 1
        
        if frame_count % 500 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    return frame_count, detection_count, total_time

print(f"\nProcessing: {args.input}")
print(f"Output: {args.output}")
print(f"Confidence: {args.conf}")
print(f"Border thickness: {args.border_thickness}px")

frame_count, detection_count, total_time = process_video(args.input, args.output)

avg_fps = frame_count / total_time
print(f"\n{'='*60}")
print(f"CUDA Segmentation Complete")
print(f"{'='*60}")
print(f"Frames: {frame_count}")
print(f"Segmentation detections: {detection_count}")
print(f"Avg detections/frame: {detection_count / frame_count:.2f}")
print(f"Total Time: {total_time:.1f}s")
print(f"Average: {total_time/frame_count*1000:.1f}ms ({avg_fps:.1f} FPS)")
print(f"{'='*60}")
