# Streaming wire detection pipeline for real-time video processing
# Two-stage pipeline: YOLO bbox detection + MobileNet color classification
# Uses CoreML on Apple Silicon for ~4x faster inference
#
# Usage:
#   from vt_perception.detectors import WireDetectionPipeline
#   pipeline = WireDetectionPipeline(target_colors=['red'])
#   annotated_frame = pipeline(frame)

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import coremltools as ct
from ultralytics import YOLO


# ============ DEFAULT CONFIGURATION ============
DEFAULT_YOLO_MODEL_PATH = '/Users/elisd/Desktop/vult/models.nosync/trained_models/jan9/bounding_box_yolov11_jan9/best.pt'
DEFAULT_COLOR_PYTORCH_PATH = '/Users/elisd/Desktop/vult/models.nosync/trained_models/jan13/wire_color_detection_jan13/best_model.pt'
DEFAULT_BBOX_THRESHOLD = 0.7
DEFAULT_COLOR_THRESHOLD = 0.8
DEFAULT_FRAME_STRIDE = 2
DEFAULT_BBOX_PADDING = 10
IMG_SIZE = 224
AVAILABLE_COLORS = ['green', 'red', 'white', 'yellow']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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
    Uses CoreML for all inference (~58 FPS on Apple Silicon).
    
    Two-stage detection:
      1. YOLO bounding box detection (CoreML)
      2. MobileNet color classification (CoreML)
    
    Usage:
      - Simple: pipeline(frame) returns annotated frame with automatic stride handling
      - Flexible: pipeline.detect() + pipeline.draw_detections() for manual control
    """
    
    def __init__(
        self,
        target_colors: list[str],
        bbox_model_path: Optional[str] = None,
        bbox_threshold: float = DEFAULT_BBOX_THRESHOLD,
        color_model_path: Optional[str] = None,
        color_threshold: float = DEFAULT_COLOR_THRESHOLD,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        bbox_padding: int = DEFAULT_BBOX_PADDING,
        bbox_thickness: int = 4,
        bbox_color: tuple[int, int, int] = (255, 0, 255),  # Magenta BGR
    ):
        """
        Initialize the wire detection pipeline.
        
        Args:
            target_colors: List of colors to detect (e.g., ['red', 'white'])
            bbox_model_path: Path to YOLO model (.pt or .mlpackage)
            bbox_threshold: Confidence threshold for bounding box detection
            color_model_path: Path to color classification checkpoint (.pt)
            color_threshold: Confidence threshold for color classification
            frame_stride: Run inference every Nth frame (1 = every frame)
            bbox_padding: Pixels to pad around bounding boxes
            bbox_thickness: Line thickness for bounding box drawing (in pixels)
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
        self.bbox_threshold = bbox_threshold
        self.bbox_thickness = bbox_thickness
        self.bbox_color = bbox_color
        
        # Load yolo model (CoreML)
        self.yolo_model = self._load_yolo_coreml(bbox_model_path or DEFAULT_YOLO_MODEL_PATH)
        
        # Load color model (CoreML)
        self.color_model, self.color_classes = self._load_color_coreml(
            color_model_path or DEFAULT_COLOR_PYTORCH_PATH
        )
        
        # Per-camera state
        self._frame_counts: dict[str, int] = {}
        self._cached_detections: dict[str, list[Detection]] = {}
    
    def _load_yolo_coreml(self, model_path: str):
        """Load YOLO model, converting to CoreML if needed."""
        model_path_obj = Path(model_path)
        
        # Convert to CoreML if .pt file
        if model_path_obj.suffix == '.pt':
            coreml_path = model_path_obj.with_suffix('.mlpackage')
            if not coreml_path.exists():
                print(f"Converting yolo to CoreML: {model_path}")
                pt_model = YOLO(model_path)
                pt_model.export(format='coreml', nms=True)
                print(f"CoreML model saved to: {coreml_path}")
            model_path = str(coreml_path)
        
        print(f"Loading yolo CoreML model: {model_path}")
        return YOLO(model_path)
    
    def _load_color_coreml(self, pytorch_path: str):
        """Load MobileNet color model, converting to CoreML if needed."""
        pytorch_path_obj = Path(pytorch_path)
        coreml_path = pytorch_path_obj.parent / 'color_model.mlpackage'
        
        # Load classes from PyTorch checkpoint
        checkpoint = torch.load(pytorch_path, map_location='cpu')
        classes = checkpoint['classes']
        
        # Convert to CoreML if needed
        if not coreml_path.exists():
            print(f"Converting color model to CoreML: {pytorch_path}")
            num_classes = len(classes)
            
            pt_model = models.mobilenet_v2(weights=None)
            in_features = pt_model.classifier[1].in_features
            pt_model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
            pt_model.load_state_dict(checkpoint['model_state_dict'])
            pt_model.eval()
            
            example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            traced_model = torch.jit.trace(pt_model, example_input)
            
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=(1, 3, IMG_SIZE, IMG_SIZE))],
                convert_to="mlprogram"
            )
            coreml_model.save(str(coreml_path))
            print(f"CoreML color model saved to: {coreml_path}")
        
        print(f"Loading color CoreML model: {coreml_path}")
        model = ct.models.MLModel(str(coreml_path))
        print(f"Color classes: {classes}")
        
        return model, classes
    
    def _predict_color(self, image_crop: np.ndarray) -> tuple[str, float]:
        """Predict color for a cropped image (RGB numpy array)."""
        image_resized = cv2.resize(image_crop, (IMG_SIZE, IMG_SIZE))
        
        # Normalize (ImageNet stats)
        image_float = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        
        # CHW format, add batch dimension
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_chw, axis=0).astype(np.float32)
        
        # Run inference
        prediction = self.color_model.predict({"input": image_batch})
        
        output_key = list(prediction.keys())[0]
        logits = prediction[output_key]
        
        # Softmax and argmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[0, predicted_idx]
        
        return self.color_classes[predicted_idx], float(confidence)
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run full inference on a frame.
        
        Args:
            frame: RGB numpy array (lerobot format)
            
        Returns:
            List containing at most one Detection (highest bbox confidence after color filter)
        """
        height, width = frame.shape[:2]
        
        # Stage 1: YOLO bounding box detection
        # Detector expects BGR, lerobot provides RGB
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.yolo_model.predict(bgr_frame, conf=self.bbox_threshold, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_conf = float(box.conf[0])
                
                # Crop for color classification (with padding)
                crop_x1 = max(0, x1 - 10)
                crop_y1 = max(0, y1 - 10)
                crop_x2 = min(width, x2 + 10)
                crop_y2 = min(height, y2 + 10)
                
                # Use RGB frame for color classification
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue
                
                # Stage 2: Color classification
                predicted_color, color_confidence = self._predict_color(crop)
                
                if predicted_color in self.target_colors and color_confidence >= self.color_threshold:
                    # Apply display padding
                    px1 = max(0, x1 - self.bbox_padding)
                    py1 = max(0, y1 - self.bbox_padding)
                    px2 = min(width, x2 + self.bbox_padding)
                    py2 = min(height, y2 + self.bbox_padding)
                    
                    detections.append(Detection(
                        x1=px1, y1=py1, x2=px2, y2=py2,
                        color=predicted_color,
                        bbox_confidence=bbox_conf,
                        color_confidence=color_confidence
                    ))
        
        # Return only the highest bbox confidence detection
        if detections:
            best = max(detections, key=lambda d: d.bbox_confidence)
            return [best]
        return []
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Optional[list[Detection]] = None
    ) -> np.ndarray:
        """Draw detection boxes on a frame."""
        output = frame.copy()
        dets = detections if detections is not None else []
        
        for det in dets:
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), self.bbox_color, self.bbox_thickness)
        
        return output
    
    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "default",
        run_inference: bool = True
    ) -> tuple[np.ndarray, list[Detection]]:
        """Process a frame with explicit control over inference."""
        if run_inference:
            detections = self.detect(frame)
            self._cached_detections[camera_id] = detections
        else:
            detections = self._cached_detections.get(camera_id, [])
        
        annotated = self.draw_detections(frame, detections)
        return annotated, detections
    
    def __call__(self, frame: np.ndarray, camera_id: str = "default") -> np.ndarray:
        """Process a frame with automatic stride handling."""
        frame_count = self._frame_counts.get(camera_id, 0)
        run_inference = (frame_count % self.frame_stride == 0)
        self._frame_counts[camera_id] = frame_count + 1
        
        annotated, _ = self.process_frame(frame, camera_id=camera_id, run_inference=run_inference)
        return annotated
    
    def reset(self, camera_id: str | None = None):
        """Reset internal state (frame count, cache)."""
        if camera_id is None:
            self._frame_counts = {}
            self._cached_detections = {}
        else:
            self._frame_counts.pop(camera_id, None)
            self._cached_detections.pop(camera_id, None)
    
    def get_detections(self, camera_id: str = "default") -> list[Detection]:
        """Get the most recent cached detections for a camera."""
        return self._cached_detections.get(camera_id, [])
    
    def set_target_colors(self, colors: list[str]) -> None:
        """Dynamically update target colors for detection.
        
        Args:
            colors: List of colors to detect (e.g., ['red', 'white'])
            
        Raises:
            ValueError: If any color is not in AVAILABLE_COLORS
        """
        validated = [c.lower() for c in colors]
        for c in validated:
            if c not in AVAILABLE_COLORS:
                raise ValueError(f"Invalid color '{c}'. Choose from: {', '.join(AVAILABLE_COLORS)}")
        self.target_colors = validated
        self.reset()  # Clear cached detections
