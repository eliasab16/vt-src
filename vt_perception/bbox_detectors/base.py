# Base protocol for bounding box detectors
#
# All bbox detectors should return detections in a common format:
#   list of (x1, y1, x2, y2, confidence) tuples

from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np


@dataclass
class BboxDetection:
    """Single bounding box detection result."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@runtime_checkable
class BboxDetector(Protocol):
    """Protocol for bounding box detectors.
    
    All implementations must provide:
        - detect(frame) -> list of BboxDetection
        - threshold property (get/set)
    """
    
    @property
    def threshold(self) -> float:
        """Get confidence threshold."""
        ...
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set confidence threshold."""
        ...
    
    def detect(self, frame: np.ndarray) -> list[BboxDetection]:
        """
        Run detection on a frame.
        
        Args:
            frame: BGR numpy array (OpenCV format)
            
        Returns:
            List of BboxDetection objects
        """
        ...
