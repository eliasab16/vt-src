# Wire detection frame processor for lerobot integration
# Wraps WireDetectionPipeline as a frame processor

from typing import Any
import numpy as np
from numpy.typing import NDArray

from ..frame_processor import FrameProcessorRegistry


@FrameProcessorRegistry.register("wire_detection")
class WireDetectionProcessor:
    """Frame processor that applies wire detection with color filtering.
    
    Uses CoreML for all inference (~58 FPS on Apple Silicon).
    
    Config options:
        target_colors: List of colors to detect (e.g., ["red", "white"])
        frame_stride: Run inference every Nth frame (default: 2)
        bbox_threshold: Bounding box confidence threshold (default: 0.7)
        color_threshold: Color confidence threshold (default: 0.8)
        bbox_padding: Default pixels to pad around boxes (default: 10)
        cameras: Dict of camera configs with per-camera padding, or list of camera IDs
                 Example: {"OpenCVCamera(0)": {"bbox_padding": 5}, "OpenCVCamera(1)": {"bbox_padding": 15}}
                 Or simple: ["OpenCVCamera(0)", "OpenCVCamera(1)"]
    """
    
    def __init__(self, pipeline, enabled_cameras: list[str] | None = None,
                 camera_padding: dict[str, int] | None = None):
        """Initialize with a WireDetectionPipeline instance."""
        self.pipeline = pipeline
        self.enabled_cameras = enabled_cameras
        self.camera_padding = camera_padding or {}
    
    def process(self, frame: NDArray[Any], camera_id: str = "default") -> NDArray[Any]:
        """Process frame through wire detection pipeline."""
        if self.enabled_cameras is not None and camera_id not in self.enabled_cameras:
            return frame
        
        # Get per-camera padding if configured
        padding = self.camera_padding.get(camera_id)
        if padding is not None:
            # Temporarily override pipeline padding for this camera
            original_padding = self.pipeline.bbox_padding
            self.pipeline.bbox_padding = padding
            result = self.pipeline(frame, camera_id)
            self.pipeline.bbox_padding = original_padding
            return result
        
        return self.pipeline(frame, camera_id)
    
    def set_target_colors(self, colors: list[str]) -> None:
        """Update target colors on the underlying pipeline."""
        self.pipeline.set_target_colors(colors)
    
    @classmethod
    def from_config(cls, config: dict) -> "WireDetectionProcessor":
        """Create processor from configuration dict."""
        from ..detectors.wire_detection import WireDetectionPipeline
        
        # Parse cameras config (can be list or dict with per-camera settings)
        cameras_config = config.get("cameras", None)
        enabled_cameras = None
        camera_padding = {}
        
        if cameras_config is not None:
            if isinstance(cameras_config, list):
                enabled_cameras = cameras_config
            elif isinstance(cameras_config, dict):
                enabled_cameras = list(cameras_config.keys())
                for cam_id, cam_config in cameras_config.items():
                    if isinstance(cam_config, dict) and "bbox_padding" in cam_config:
                        camera_padding[cam_id] = cam_config["bbox_padding"]
        
        pipeline = WireDetectionPipeline(
            target_colors=config.get("target_colors", ["red"]),
            frame_stride=config.get("frame_stride", 2),
            bbox_threshold=config.get("bbox_threshold", 0.7),
            color_threshold=config.get("color_threshold", 0.8),
            bbox_padding=config.get("bbox_padding", 10),
            bbox_thickness=config.get("bbox_thickness", 6),
            bbox_color=tuple(config.get("bbox_color", [255, 0, 255]))  # Magenta BGR
        )
        
        return cls(pipeline, enabled_cameras, camera_padding)

