# Breaker segmentation pipeline for real-time video processing
# Uses YOLO segmentation model with CoreML on Apple Silicon
# Draws only mask borders (no fill) for visual feedback

from typing import Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import cv2

# Default configuration values
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_BORDER_THICKNESS = 4
DEFAULT_BORDER_COLOR = (255, 0, 255)  # Magenta (BGR)
DEFAULT_FRAME_STRIDE = 1


@dataclass
class SegmentationResult:
    """Single segmentation result."""
    mask_polygon: np.ndarray  # Polygon coordinates (N, 2)
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class BreakerSegmentationPipeline:
    """Streaming breaker segmentation pipeline for real-time video processing.
    Uses CoreML for inference on Apple Silicon.
    
    Draws only mask borders (no fill) for visual feedback during recording/inference.
    
    Args:
        model_path: Path to YOLO segmentation model (.pt or .mlpackage)
        conf_threshold: Confidence threshold for detections
        border_thickness: Thickness of mask border in pixels
        border_color: BGR color tuple for border
        frame_stride: Process every Nth frame (1 = every frame)
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        border_thickness: int = DEFAULT_BORDER_THICKNESS,
        border_color: tuple[int, int, int] = DEFAULT_BORDER_COLOR,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
    ):
        self.conf_threshold = conf_threshold
        self.border_thickness = border_thickness
        self.border_color = border_color
        self.frame_stride = frame_stride
        
        # Load model (convert to CoreML if needed)
        self.model = self._load_yolo_coreml(model_path)
        
        # Per-camera state for frame stride
        self._frame_counts: dict[str, int] = {}
        self._cached_results: dict[str, list[SegmentationResult]] = {}
    
    def _load_yolo_coreml(self, model_path: str):
        """Load YOLO model, converting to CoreML if needed."""
        from ultralytics import YOLO
        
        model_path_obj = Path(model_path)
        
        # If .pt file, convert to CoreML
        if model_path_obj.suffix == '.pt':
            coreml_path = model_path_obj.with_suffix('.mlpackage')
            if not coreml_path.exists():
                print(f"Converting {model_path} to CoreML for Apple Silicon acceleration...")
                pt_model = YOLO(model_path, task='segment')
                pt_model.export(format='coreml', nms=False)
                print(f"CoreML model saved to: {coreml_path}")
            model_path = str(coreml_path)
        
        return YOLO(model_path, task='segment')
    
    def detect(self, frame: np.ndarray) -> list[SegmentationResult]:
        """
        Run segmentation inference on a frame.
        
        Args:
            frame: RGB numpy array (lerobot format)
            
        Returns:
            List of SegmentationResult objects
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.masks is None or r.masks.xy is None:
                continue
            
            # Process each mask
            for i, polygon in enumerate(r.masks.xy):
                if len(polygon) == 0:
                    continue
                
                # Get corresponding box and confidence
                if r.boxes is not None and i < len(r.boxes):
                    box = r.boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    detections.append(SegmentationResult(
                        mask_polygon=polygon,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))
        
        return detections
    
    def draw_segmentation_borders(
        self, 
        frame: np.ndarray, 
        detections: list[SegmentationResult]
    ) -> np.ndarray:
        """Draw mask borders on frame.
        
        Args:
            frame: Frame to draw on (RGB numpy array from lerobot)
            detections: List of segmentation results
            
        Returns:
            Modified frame
        """
        # Convert border_color from BGR to RGB since lerobot passes RGB frames
        # border_color is stored as BGR (cv2 convention) but frame is RGB
        rgb_color = (self.border_color[2], self.border_color[1], self.border_color[0])
        
        for detection in detections:
            # Convert polygon to integer points for cv2
            pts = detection.mask_polygon.astype(np.int32).reshape((-1, 1, 2))
            # Draw only the polygon border (closed polyline)
            cv2.polylines(
                frame, 
                [pts], 
                isClosed=True, 
                color=rgb_color, 
                thickness=self.border_thickness
            )
        
        return frame
    
    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "default",
        run_inference: bool = True
    ) -> np.ndarray:
        """Process a frame with explicit control over inference.
        
        Args:
            frame: RGB numpy array
            camera_id: Camera identifier for per-camera state
            run_inference: Whether to run inference or use cached results
            
        Returns:
            Frame with segmentation borders drawn
        """
        if run_inference:
            detections = self.detect(frame)
            self._cached_results[camera_id] = detections
        else:
            detections = self._cached_results.get(camera_id, [])
        
        return self.draw_segmentation_borders(frame.copy(), detections)
    
    def __call__(self, frame: np.ndarray, camera_id: str = "default") -> np.ndarray:
        """Process a frame with automatic stride handling.
        
        Args:
            frame: RGB numpy array
            camera_id: Camera identifier
            
        Returns:
            Frame with segmentation borders drawn
        """
        # Initialize frame count for this camera
        if camera_id not in self._frame_counts:
            self._frame_counts[camera_id] = 0
        
        # Determine if we should run inference this frame
        run_inference = (self._frame_counts[camera_id] % self.frame_stride) == 0
        self._frame_counts[camera_id] += 1
        
        return self.process_frame(frame, camera_id, run_inference)
    
    def reset(self, camera_id: str | None = None) -> None:
        """Reset internal state (frame count, cache).
        
        Args:
            camera_id: Specific camera to reset, or None for all cameras
        """
        if camera_id is None:
            self._frame_counts.clear()
            self._cached_results.clear()
        else:
            self._frame_counts.pop(camera_id, None)
            self._cached_results.pop(camera_id, None)
    
    def get_detections(self, camera_id: str = "default") -> list[SegmentationResult]:
        """Get the most recent cached detections for a camera."""
        return self._cached_results.get(camera_id, [])
