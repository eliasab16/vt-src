# This wraps WireDetectionPipeline as a frame processor to be compliant with the expected interface
# for frame processing in lerobot scripts 

from typing import Any
import numpy as np
from numpy.typing import NDArray

from ..frame_processor import FrameProcessorRegistry


@FrameProcessorRegistry.register("wire_detection")
class WireDetectionProcessor:
    """Frame processor that applies wire detection with color filtering.
    
    Config options:
        target_colors: List of colors to detect (e.g., ["red", "white"])
        bbox_model_type: Detector type - "yolo" (faster) or "rfdetr" (default: "yolo")
        bbox_model_path: Custom path to bbox model (optional)
        frame_stride: Run inference every Nth frame (default: 2)
        bbox_threshold: Bounding box confidence threshold (default: 0.7)
        color_threshold: Color confidence threshold (default: 0.8)
        bbox_padding: Pixels to pad around boxes (default: 10)
        cameras: List of camera IDs to process (default: all cameras)
                 Camera ID format: "OpenCVCamera(index)" e.g. ["OpenCVCamera(0)"]
    """
    
    def __init__(self, pipeline, enabled_cameras: list[str] | None = None):
        """Initialize with a WireDetectionPipeline instance."""
        self.pipeline = pipeline
        self.enabled_cameras = enabled_cameras  # None means all cameras
    
    def process(self, frame: NDArray[Any], camera_id: str = "default") -> NDArray[Any]:
        """process frame through wire detection pipeline
        
        Args:
            frame: RGB numpy array
            camera_id: Identifier for the camera source (for per-camera state)
        """
        # Skip processing if camera is not in enabled list
        if self.enabled_cameras is not None and camera_id not in self.enabled_cameras:
            return frame
        
        return self.pipeline(frame, camera_id)
    
    @classmethod
    def from_config(cls, config: dict) -> "WireDetectionProcessor":
        """create processor from configuration dict."""
        # Import detector lazily to avoid loading models at module import time
        from ..detectors.wire_detection import WireDetectionPipeline
        
        pipeline = WireDetectionPipeline(
            target_colors=config.get("target_colors", ["red"]),
            bbox_model_type=config.get("bbox_model_type", "yolo"),
            bbox_model_path=config.get("bbox_model_path", None),
            frame_stride=config.get("frame_stride", 2),
            bbox_threshold=config.get("bbox_threshold", 0.7),
            color_threshold=config.get("color_threshold", 0.8),
            bbox_padding=config.get("bbox_padding", 10)
        )
        
        enabled_cameras = config.get("cameras", None)
        
        return cls(pipeline, enabled_cameras)

