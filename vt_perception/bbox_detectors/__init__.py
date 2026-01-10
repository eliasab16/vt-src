# Bounding Box Detectors
#
# Abstract detector pattern for swappable bbox detection models.
# All detectors implement the BboxDetector protocol and return BboxDetection objects.

from .base import BboxDetector, BboxDetection
from .rfdetr import RFDETRDetector
from .yolo import YOLODetector

__all__ = [
    'BboxDetector',
    'BboxDetection',
    'RFDETRDetector',
    'YOLODetector',
]
