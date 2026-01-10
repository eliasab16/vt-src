# rf-detr bounding box detector implementation

import numpy as np
import cv2
from PIL import Image
from rfdetr import RFDETRBase

from .base import BboxDetector, BboxDetection


class RFDETRDetector:
    """RF-DETR based bounding box detector."""
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.7,
    ):
        """
        Initialize RF-DETR detector.
        
        Args:
            model_path: Path to RF-DETR checkpoint (.pth file)
            threshold: Confidence threshold for detections
        """
        self._threshold = threshold
        self._model_path = model_path
        
        print(f"Loading RF-DETR model from: {model_path}")
        self._model = RFDETRBase(pretrain_weights=model_path)
        print("RF-DETR loaded successfully")
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value
    
    def detect(self, frame: np.ndarray) -> list[BboxDetection]:
        """
        Run RF-DETR detection on a frame.
        
        Args:
            frame: BGR numpy array (OpenCV format)
            
        Returns:
            List of BboxDetection objects
        """
        # Convert BGR to RGB for RF-DETR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run detection
        raw_detections = self._model.predict(pil_image, threshold=self._threshold)
        
        # Convert to common format
        detections = []
        for i in range(len(raw_detections.xyxy)):
            x1, y1, x2, y2 = raw_detections.xyxy[i].astype(int)
            confidence = float(raw_detections.confidence[i])
            
            detections.append(BboxDetection(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                confidence=confidence
            ))
        
        return detections
