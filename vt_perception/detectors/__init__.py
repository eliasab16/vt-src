# Detectors module
from .wire_detection import WireDetectionPipeline
from .breaker_segmentation import BreakerSegmentationPipeline

__all__ = ["WireDetectionPipeline", "BreakerSegmentationPipeline"]
