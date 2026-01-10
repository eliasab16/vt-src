# Streaming wire detection pipeline for real-time video processing
# Two-stage pipeline: Bbox Detector + MobileNetV2 color classification
#
# Usage:
#   from vt_perception.detectors import WireDetectionPipeline
#   pipeline = WireDetectionPipeline(target_colors=['red'], bbox_model_type='yolo')
#   annotated_frame = pipeline(frame)

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from dataclasses import dataclass
from typing import Optional

from ..bbox_detectors import BboxDetector, BboxDetection, RFDETRDetector, YOLODetector


# ============ DEFAULT CONFIGURATION ============
DEFAULT_RFDETR_MODEL_PATH = '/Users/elisd/Desktop/vult/models/trained_models/bounding_box_rfdetr_jan7/rfdetr_jan7/checkpoint_best_total.pth'
DEFAULT_YOLO_MODEL_PATH = '/Users/elisd/Desktop/vult/models/trained_models/bounding_box_yolov11_jan9/best.pt'
DEFAULT_COLOR_MODEL_PATH = '/Users/elisd/Desktop/vult/models/trained_models/wire_color_detection_imagenet_jan6/imagenet_wire_color_detection_model_jan6.pt'
DEFAULT_BBOX_THRESHOLD = 0.7
DEFAULT_COLOR_THRESHOLD = 0.8
DEFAULT_FRAME_STRIDE = 2
DEFAULT_BBOX_PADDING = 10
IMG_SIZE = 224
AVAILABLE_COLORS = ['green', 'red', 'white', 'yellow']
# ===============================================


@dataclass
class Detection:
    """Single detection result."""
    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    bbox_confidence: float
    color_confidence: float


class WireDetectionPipeline:
    """
    Streaming wire detection pipeline for real-time video processing.
    
    Two-stage detection:
      1. Bounding box detection (RF-DETR or YOLOv11)
      2. MobileNetV2 color classification
    
    Can be used in multiple ways:
      - Simple: pipeline(frame) returns annotated frame with automatic stride handling
      - Flexible: pipeline.detect() + pipeline.draw_detections() for manual control
    """
    
    def __init__(
        self,
        target_colors: list[str],
        bbox_detector: Optional[BboxDetector] = None,
        bbox_model_type: str = 'yolo',  # 'yolo' or 'rfdetr'
        bbox_model_path: Optional[str] = None,
        bbox_threshold: float = DEFAULT_BBOX_THRESHOLD,
        color_model_path: str = DEFAULT_COLOR_MODEL_PATH,
        color_threshold: float = DEFAULT_COLOR_THRESHOLD,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        bbox_padding: int = DEFAULT_BBOX_PADDING,
        device: Optional[str] = None
    ):
        """
        Initialize the wire detection pipeline.
        
        Args:
            target_colors: List of colors to detect (e.g., ['red', 'white'])
            bbox_detector: Pre-configured BboxDetector instance (overrides bbox_model_type/path)
            bbox_model_type: Type of detector if not providing bbox_detector ('yolo' or 'rfdetr')
            bbox_model_path: Path to bbox model (uses default for type if None)
            bbox_threshold: Confidence threshold for bounding box detection
            color_model_path: Path to color classification checkpoint
            color_threshold: Confidence threshold for color classification
            frame_stride: Run inference every Nth frame (1 = every frame)
            bbox_padding: Pixels to pad around bounding boxes
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        # Validate colors
        self.target_colors = [c.lower() for c in target_colors]
        for c in self.target_colors:
            if c not in AVAILABLE_COLORS:
                raise ValueError(f"Invalid color '{c}'. Choose from: {', '.join(AVAILABLE_COLORS)}")
        
        # Store configuration
        self.color_threshold = color_threshold
        self.frame_stride = frame_stride
        self.bbox_padding = bbox_padding
        
        # Setup device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Image transform for color classification
        self.eval_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Setup bbox detector
        if bbox_detector is not None:
            self.bbox_detector = bbox_detector
        else:
            # Create detector based on type
            if bbox_model_type.lower() == 'yolo':
                model_path = bbox_model_path or DEFAULT_YOLO_MODEL_PATH
                self.bbox_detector = YOLODetector(model_path=model_path, threshold=bbox_threshold)
            elif bbox_model_type.lower() == 'rfdetr':
                model_path = bbox_model_path or DEFAULT_RFDETR_MODEL_PATH
                self.bbox_detector = RFDETRDetector(model_path=model_path, threshold=bbox_threshold)
            else:
                raise ValueError(f"Unknown bbox_model_type: {bbox_model_type}. Use 'yolo' or 'rfdetr'")
        
        # Load color model
        self.color_model, self.color_classes = self._load_color_model(color_model_path)
        
        # Per-camera state: keyed by camera_id
        self._frame_counts: dict[str, int] = {}
        self._cached_detections: dict[str, list[Detection]] = {}
    
    def _load_color_model(self, path: str):
        """Load MobileNetV2 color classification model."""
        print(f"Loading color model from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        classes = checkpoint['classes']
        num_classes = len(classes)
        
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Color classes: {classes}")
        return model, classes
    
    def _predict_color(self, image_crop: np.ndarray) -> tuple[str, float]:
        """Predict color for a cropped image (RGB numpy array)."""
        # Frame is already RGB from lerobot
        image_pil = Image.fromarray(image_crop)
        image_tensor = self.eval_transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.color_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        return self.color_classes[predicted.item()], confidence.item()
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run full inference on a frame.
        
        Args:
            frame: RGB numpy array (lerobot format)
            
        Returns:
            List of Detection objects that match target colors
        """
        height, width = frame.shape[:2]
        
        # Stage 1: Bounding box detection using pluggable detector
        # Note: Detector expects BGR, but lerobot provides RGB
        # Convert to BGR for detector
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bbox_detections = self.bbox_detector.detect(bgr_frame)
        
        # Stage 2: Color classification for each bbox
        detections = []
        for bbox in bbox_detections:
            # Pad for cropping (to avoid cutting off wire edges)
            crop_x1 = max(0, bbox.x1 - 10)
            crop_y1 = max(0, bbox.y1 - 10)
            crop_x2 = min(width, bbox.x2 + 10)
            crop_y2 = min(height, bbox.y2 + 10)
            
            # Use RGB frame for color classification
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue
            
            # Color classification
            predicted_color, color_confidence = self._predict_color(crop)
            
            if predicted_color in self.target_colors and color_confidence >= self.color_threshold:
                # Apply display padding
                px1 = max(0, bbox.x1 - self.bbox_padding)
                py1 = max(0, bbox.y1 - self.bbox_padding)
                px2 = min(width, bbox.x2 + self.bbox_padding)
                py2 = min(height, bbox.y2 + self.bbox_padding)
                
                detections.append(Detection(
                    x1=px1, y1=py1, x2=px2, y2=py2,
                    color=predicted_color,
                    bbox_confidence=bbox.confidence,
                    color_confidence=color_confidence
                ))
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Optional[list[Detection]] = None
    ) -> np.ndarray:
        """
        Draw detection boxes on a frame.
        
        Args:
            frame: RGB numpy array
            detections: List of Detection objects (uses cached if None)
            
        Returns:
            Annotated frame (copy of input)
        """
        output = frame.copy()
        dets = detections if detections is not None else self._cached_detections
        
        for det in dets:
            # Draw bounding box (green in RGB)
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.color} | bbox:{det.bbox_confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (det.x1, det.y1 - label_size[1] - 10),
                         (det.x1 + label_size[0], det.y1), (0, 255, 0), -1)
            cv2.putText(output, label, (det.x1, det.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return output
    
    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "default",
        run_inference: bool = True
    ) -> tuple[np.ndarray, list[Detection]]:
        """
        Process a frame with explicit control over inference.
        
        Args:
            frame: RGB numpy array
            camera_id: Identifier for the camera source
            run_inference: If True, run detection. If False, use cached detections.
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        if run_inference:
            detections = self.detect(frame)
            self._cached_detections[camera_id] = detections
        else:
            detections = self._cached_detections.get(camera_id, [])
        
        annotated = self.draw_detections(frame, detections)
        return annotated, detections
    
    def __call__(self, frame: np.ndarray, camera_id: str = "default") -> np.ndarray:
        """
        Process a frame with automatic stride handling.
        
        This is the simplest interface - just call pipeline(frame, camera_id) repeatedly.
        Handles frame stride and caching per-camera.
        
        Args:
            frame: RGB numpy array
            camera_id: Identifier for the camera source
            
        Returns:
            Annotated frame with detection boxes
        """
        # Get per-camera frame count
        frame_count = self._frame_counts.get(camera_id, 0)
        
        # Determine if we should run inference based on stride
        run_inference = (frame_count % self.frame_stride == 0)
        self._frame_counts[camera_id] = frame_count + 1
        
        annotated, _ = self.process_frame(frame, camera_id=camera_id, run_inference=run_inference)
        return annotated
    
    def reset(self, camera_id: str | None = None):
        """Reset internal state (frame count, cache).
        
        Args:
            camera_id: If provided, reset only that camera's state. If None, reset all.
        """
        if camera_id is None:
            self._frame_counts = {}
            self._cached_detections = {}
        else:
            self._frame_counts.pop(camera_id, None)
            self._cached_detections.pop(camera_id, None)
    
    def get_detections(self, camera_id: str = "default") -> list[Detection]:
        """Get the most recent cached detections for a camera."""
        return self._cached_detections.get(camera_id, [])
