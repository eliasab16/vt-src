# vt_perception package
# Frame processing and inference utilities for camera streams
# Uses CoreML for all inference on Apple Silicon

from .frame_processor import FrameProcessor, FrameProcessorRegistry

__all__ = [
    "FrameProcessor",
    "FrameProcessorRegistry",
]
