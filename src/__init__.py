"""
Initialization file for the src package.
"""

__version__ = "2.0.0"
__author__ = "Nipun Keshan"
__email__ = "your-email@domain.com"
__description__ = "Real-Time AI Scene Description System"

# Import main components for easy access
from .models.model_manager import ModelManager, ModelOutput
from .processors.video_processor import VideoProcessor
from .utils.config_loader import ConfigLoader
from .utils.metrics import MetricsCollector

__all__ = [
    "ModelManager",
    "ModelOutput", 
    "VideoProcessor",
    "ConfigLoader",
    "MetricsCollector"
]
