# Breaker segmentation frame processor for lerobot integration
# Wraps BreakerSegmentationPipeline as a frame processor

from typing import Any
import numpy as np
from numpy.typing import NDArray

from ..frame_processor import FrameProcessorRegistry


@FrameProcessorRegistry.register("breaker_segmentation")
class BreakerSegmentationProcessor:
    """Frame processor that applies breaker segmentation with border drawing.
    
    Uses CoreML for inference on Apple Silicon.
    
    Config options:
        model_path: Path to YOLO segmentation model (.pt or .mlpackage)
        conf_threshold: Confidence threshold (default: 0.5)
        border_thickness: Border thickness in pixels (default: 4)
        border_color: BGR color tuple (default: [255, 0, 255] magenta)
        frame_stride: Run inference every Nth frame (default: 1)
        cameras: List of camera IDs to enable, or dict with per-camera settings
                 Example: ["OpenCVCamera(0)", "OpenCVCamera(1)"]
                 Or: {"OpenCVCamera(0)": {"border_thickness": 6}}
    """
    
    def __init__(self, pipeline, enabled_cameras: list[str] | None = None,
                 camera_settings: dict[str, dict] | None = None):
        """Initialize with a BreakerSegmentationPipeline instance."""
        self.pipeline = pipeline
        self.enabled_cameras = enabled_cameras
        self.camera_settings = camera_settings or {}
    
    def process(self, frame: NDArray[Any], camera_id: str = "default") -> NDArray[Any]:
        """Process frame through breaker segmentation pipeline."""
        if self.enabled_cameras is not None and camera_id not in self.enabled_cameras:
            return frame
        
        # Get per-camera settings if configured
        camera_config = self.camera_settings.get(camera_id, {})
        
        # Temporarily override pipeline settings for this camera if needed
        if camera_config:
            original_settings = {}
            for key, value in camera_config.items():
                if hasattr(self.pipeline, key):
                    original_settings[key] = getattr(self.pipeline, key)
                    setattr(self.pipeline, key, value)
            
            result = self.pipeline(frame, camera_id)
            
            # Restore original settings
            for key, value in original_settings.items():
                setattr(self.pipeline, key, value)
            
            return result
        
        return self.pipeline(frame, camera_id)
    
    @classmethod
    def from_config(cls, config: dict) -> "BreakerSegmentationProcessor":
        """Create processor from configuration dict."""
        from ..detectors.breaker_segmentation import BreakerSegmentationPipeline
        
        # Validate required config
        if "model_path" not in config:
            raise ValueError("breaker_segmentation config requires 'model_path'")
        
        # Parse cameras config (can be list or dict with per-camera settings)
        cameras_config = config.get("cameras", None)
        enabled_cameras = None
        camera_settings = {}
        
        if cameras_config is not None:
            if isinstance(cameras_config, list):
                enabled_cameras = cameras_config
            elif isinstance(cameras_config, dict):
                enabled_cameras = list(cameras_config.keys())
                camera_settings = cameras_config
        
        # Create pipeline with global settings
        pipeline = BreakerSegmentationPipeline(
            model_path=config["model_path"],
            conf_threshold=config.get("conf_threshold", 0.5),
            border_thickness=config.get("border_thickness", 4),
            border_color=tuple(config.get("border_color", [255, 0, 255])),
            frame_stride=config.get("frame_stride", 1)
        )
        
        return cls(pipeline, enabled_cameras, camera_settings)
