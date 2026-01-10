# yolov11 bounding box detector implementation

import numpy as np
from ultralytics import YOLO

from .base import BboxDetector, BboxDetection


class YOLODetector:
    """yolov11-based bounding box detector."""
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.7,
    ):
        """initialize YOLOv11 detector.
        
        Args:
            model_path: Path to YOLO model (.pt file)
            threshold: Confidence threshold for detections
        """
        self._threshold = threshold
        self._model_path = model_path
        
        print(f"Loading YOLOv11 model from: {model_path}")
        self._model = YOLO(model_path)
        print("YOLOv11 loaded successfully")
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value
    
    def detect(self, frame: np.ndarray) -> list[BboxDetection]:
        """run YOLOv11 detection on a frame.
        
        Args:
            frame: BGR numpy array (OpenCV format)
            
        Returns:
            List of BboxDetection objects
        """
        # run detection (yolo handles BGR input)
        results = self._model.predict(frame, conf=self._threshold, verbose=False)
        
        # convert to common format
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                detections.append(BboxDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence
                ))
        
        return detections
