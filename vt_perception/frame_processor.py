# Frame Processor System
# Provides a pluggable architecture for applying inference models to camera frames

from typing import Protocol, Any
import numpy as np
from numpy.typing import NDArray


class FrameProcessor(Protocol):
    """Protocol defining the interface for frame processors.
    
    All frame processors must implement:
    - process(frame) -> modified_frame
    - from_config(config) -> processor instance (class method)
    
    Example implementation:
        class MyProcessor:
            def __init__(self, threshold: float):
                self.threshold = threshold
            
            def process(self, frame):
                # Apply some transformation
                return transformed_frame
            
            @classmethod
            def from_config(cls, config: dict) -> "MyProcessor":
                # Parse config dict and create instance
                threshold = config.get("threshold", 0.5)
                return cls(threshold)
    """
    
    def process(self, frame: NDArray[Any], camera_id: str = "default") -> NDArray[Any]:
        """Process a frame and return the modified frame.
        
        Args:
            frame: Input frame (RGB numpy array)
            camera_id: Identifier for the camera source (for per-camera state)
        """
        ...
    
    @classmethod
    def from_config(cls, config: dict) -> "FrameProcessor":
        """Factory method to create a processor from a configuration dict."""
        ...


class FrameProcessorRegistry:
    """Singleton registry for frame processors.
    
    Usage:
        @FrameProcessorRegistry.register("my_processor")
        class MyProcessor:
            ...
        
        FrameProcessorRegistry.initialize("my_processor", {"param": "value"})
        processed = FrameProcessorRegistry.process(frame)
    """
    
    _processors: dict[str, type] = {}
    _active_processor: FrameProcessor | None = None
    _initialized: bool = False
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor class by name."""
        def decorator(processor_class: type) -> type:
            cls._processors[name] = processor_class
            return processor_class
        return decorator
    
    @classmethod
    def register_class(cls, name: str, processor_class: type) -> None:
        """Directly register an existing processor class by name."""
        cls._processors[name] = processor_class
    
    @classmethod
    def available_processors(cls) -> list[str]:
        """Return list of registered processor names."""
        return list(cls._processors.keys())
    
    @classmethod
    def initialize(cls, name: str, config: dict) -> "FrameProcessor":
        """Initialize and set the active processor."""
        if name not in cls._processors:
            raise ValueError(
                f"Unknown processor '{name}'. "
                f"Available: {', '.join(cls.available_processors())}"
            )
        
        processor_class = cls._processors[name]
        cls._active_processor = processor_class.from_config(config)
        cls._initialized = True
        return cls._active_processor
    
    @classmethod
    def process(cls, frame: NDArray[Any], camera_id: str = "default") -> NDArray[Any]:
        """Process frame with active processor. Returns unchanged if none active.
        
        Args:
            frame: Input frame (RGB numpy array)
            camera_id: Identifier for the camera source (for per-camera state)
        """
        if cls._active_processor is None:
            return frame
        return cls._active_processor.process(frame, camera_id)
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if a processor has been initialized."""
        return cls._initialized
    
    @classmethod
    def reset(cls) -> None:
        """Reset the active processor."""
        cls._active_processor = None
        cls._initialized = False


# Auto-register processors when this module is imported
def _auto_register_processors():
    """Import processor modules to trigger registration."""
    try:
        from .processors import wire_detection_processor  # noqa: F401
    except ImportError:
        pass


_auto_register_processors()
