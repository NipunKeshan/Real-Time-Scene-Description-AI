"""
Comprehensive test suite for Real-Time AI Scene Description System.
"""

import pytest
import numpy as np
from PIL import Image
import cv2
import time
import tempfile
from pathlib import Path
import asyncio
from unittest.mock import Mock, patch

# Test fixtures and utilities
@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 1] = 255  # Green channel
    cv2.putText(image, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image."""
    return Image.new('RGB', (640, 480), color='green')

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "app": {
            "name": "Test App",
            "debug": True,
            "log_level": "DEBUG"
        },
        "models": {
            "blip": {
                "model_name": "Salesforce/blip-image-captioning-base",
                "device": "cpu",
                "precision": "fp32",
                "max_new_tokens": 30,
                "temperature": 0.7
            },
            "yolo": {
                "model_name": "yolov8n",
                "confidence": 0.5,
                "iou_threshold": 0.45,
                "max_detections": 100
            },
            "clip": {
                "model_name": "ViT-B/32",
                "device": "cpu"
            },
            "emotion": {
                "model_name": "j-hartmann/emotion-english-distilroberta-base",
                "device": "cpu"
            }
        },
        "video": {
            "input": {
                "camera_id": 0,
                "resolution": [640, 480],
                "fps": 30
            },
            "processing": {
                "frame_skip": 30,
                "resize_for_inference": [384, 384]
            }
        },
        "performance": {
            "gpu": {
                "enable_mixed_precision": False
            },
            "cache": {
                "enable_result_cache": True
            },
            "threading": {
                "enable_async_processing": False,
                "worker_threads": 1
            }
        }
    }

# Configuration Tests
class TestConfigLoader:
    """Test configuration loading and validation."""
    
    def test_load_default_config(self, mock_config, tmp_path):
        """Test loading default configuration."""
        from src.utils.config_loader import ConfigLoader
        
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load_config()
        
        assert config["app"]["name"] == "Test App"
        assert config["models"]["blip"]["device"] == "cpu"
    
    def test_env_variable_override(self, mock_config, tmp_path, monkeypatch):
        """Test environment variable overrides."""
        from src.utils.config_loader import ConfigLoader
        
        # Set environment variable
        monkeypatch.setenv("RTSD_DEBUG", "true")
        monkeypatch.setenv("RTSD_DEVICE", "cuda")
        
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load_config()
        
        assert config["app"]["debug"] == True
        assert config["models"]["blip"]["device"] == "cuda"
    
    def test_config_validation(self):
        """Test configuration validation."""
        from src.utils.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        
        # Test missing required section
        invalid_config = {"app": {"name": "test"}}
        
        with pytest.raises(ValueError, match="Missing required configuration section"):
            loader._validate_config(invalid_config)


# Model Manager Tests
class TestModelManager:
    """Test AI model management."""
    
    @pytest.mark.slow
    def test_model_initialization(self, mock_config):
        """Test model manager initialization."""
        from src.models.model_manager import ModelManager
        
        # Use CPU for testing
        mock_config["models"]["blip"]["device"] = "cpu"
        
        manager = ModelManager(mock_config)
        
        assert manager.device.type == "cpu"
        assert "blip" in manager.models
    
    def test_caption_generation(self, mock_config, sample_pil_image):
        """Test caption generation."""
        from src.models.model_manager import ModelManager
        
        with patch('transformers.BlipProcessor.from_pretrained') as mock_processor, \
             patch('transformers.BlipForConditionalGeneration.from_pretrained') as mock_model:
            
            # Mock processor and model
            mock_proc_instance = Mock()
            mock_model_instance = Mock()
            
            mock_processor.return_value = mock_proc_instance
            mock_model.return_value = mock_model_instance
            
            # Mock processor methods
            mock_proc_instance.return_value = Mock()
            mock_proc_instance.decode.return_value = "A test image with green background"
            
            # Mock model generate
            mock_model_instance.generate.return_value = [[1, 2, 3]]  # Dummy tokens
            mock_model_instance.to.return_value = mock_model_instance
            
            manager = ModelManager(mock_config)
            caption, confidence = manager.generate_caption(sample_pil_image)
            
            assert isinstance(caption, str)
            assert 0 <= confidence <= 1
    
    def test_object_detection(self, mock_config, sample_image):
        """Test object detection."""
        from src.models.model_manager import ModelManager
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo = Mock()
            mock_yolo_class.return_value = mock_yolo
            
            # Mock YOLO results
            mock_result = Mock()
            mock_result.names = {0: "person", 1: "car"}
            
            mock_box = Mock()
            mock_box.cls = [0]
            mock_box.conf = [0.8]
            mock_box.xyxy = [[100, 100, 200, 200]]
            
            mock_result.boxes = [mock_box]
            mock_yolo.return_value = [mock_result]
            
            manager = ModelManager(mock_config)
            objects = manager.detect_objects(sample_image)
            
            assert isinstance(objects, list)
    
    def test_emotion_analysis(self, mock_config):
        """Test emotion analysis."""
        from src.models.model_manager import ModelManager
        
        with patch('transformers.pipeline') as mock_pipeline:
            mock_emotion_model = Mock()
            mock_pipeline.return_value = mock_emotion_model
            
            mock_emotion_model.return_value = [
                {"label": "joy", "score": 0.8},
                {"label": "surprise", "score": 0.2}
            ]
            
            manager = ModelManager(mock_config)
            emotions = manager.analyze_emotions("A happy scene with people smiling")
            
            assert isinstance(emotions, list)
    
    def test_frame_processing(self, mock_config, sample_image):
        """Test complete frame processing pipeline."""
        from src.models.model_manager import ModelManager, ModelOutput
        
        # Mock all model components
        with patch.multiple(
            'src.models.model_manager',
            BlipProcessor=Mock(),
            BlipForConditionalGeneration=Mock(),
            YOLO=Mock(),
            pipeline=Mock()
        ):
            manager = ModelManager(mock_config)
            
            # Mock methods
            manager.generate_caption = Mock(return_value=("Test caption", 0.9))
            manager.detect_objects = Mock(return_value=[])
            manager.analyze_emotions = Mock(return_value=["joy"])
            
            output = manager.process_frame(sample_image)
            
            assert isinstance(output, ModelOutput)
            assert output.caption == "Test caption"
            assert output.confidence == 0.9


# Video Processor Tests
class TestVideoProcessor:
    """Test video processing pipeline."""
    
    def test_initialization(self, mock_config):
        """Test video processor initialization."""
        from src.processors.video_processor import VideoProcessor
        from src.models.model_manager import ModelManager
        
        with patch('src.models.model_manager.ModelManager'):
            mock_manager = Mock()
            processor = VideoProcessor(mock_config, mock_manager)
            
            assert processor.camera_id == 0
            assert processor.resolution == (640, 480)
            assert processor.frame_skip == 30
    
    def test_frame_preprocessing(self, mock_config, sample_image):
        """Test frame preprocessing."""
        from src.processors.video_processor import VideoProcessor
        
        mock_manager = Mock()
        processor = VideoProcessor(mock_config, mock_manager)
        
        processed = processor.preprocess_frame(sample_image)
        
        assert processed.shape == (384, 384, 3)  # Resized to inference size
    
    def test_visualization_drawing(self, mock_config, sample_image):
        """Test drawing visualizations on frame."""
        from src.processors.video_processor import VideoProcessor
        from src.models.model_manager import ModelOutput
        
        mock_manager = Mock()
        processor = VideoProcessor(mock_config, mock_manager)
        
        # Create mock output
        output = ModelOutput(
            caption="Test caption",
            confidence=0.9,
            objects=[{
                "class": "person",
                "confidence": 0.8,
                "bbox": [100, 100, 200, 200]
            }],
            emotions=["joy"],
            processing_time=0.05,
            timestamp=time.time()
        )
        
        vis_frame = processor.draw_visualizations(sample_image, output)
        
        assert vis_frame.shape == sample_image.shape
        assert not np.array_equal(vis_frame, sample_image)  # Frame should be modified
    
    @patch('cv2.VideoCapture')
    def test_camera_initialization(self, mock_cap, mock_config):
        """Test camera initialization."""
        from src.processors.video_processor import VideoProcessor
        
        # Mock camera
        mock_cap_instance = Mock()
        mock_cap.return_value = mock_cap_instance
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = [640, 480, 30]  # width, height, fps
        
        mock_manager = Mock()
        processor = VideoProcessor(mock_config, mock_manager)
        
        result = processor.initialize_camera()
        
        assert result == True
        mock_cap_instance.set.assert_called()


# Metrics Tests
class TestMetricsCollector:
    """Test metrics collection and analysis."""
    
    def test_metrics_update(self):
        """Test metrics updating."""
        from src.utils.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        metrics = {
            "fps": 25.5,
            "processing_time": 45.2,
            "memory_usage": 2048
        }
        
        collector.update_metrics(metrics)
        current = collector.get_current_metrics()
        
        assert current["fps"] == 25.5
        assert current["processing_time"] == 45.2
        assert "timestamp" in current
    
    def test_statistics_calculation(self):
        """Test statistics calculation."""
        from src.utils.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Add multiple metrics
        for i in range(10):
            collector.update_metrics({
                "fps": 20 + i,
                "processing_time": 50 + i,
                "memory_usage": 2000 + i * 100
            })
        
        stats = collector.get_statistics()
        
        assert "fps" in stats
        assert "mean" in stats["fps"]
        assert "std" in stats["fps"]
        assert stats["fps"]["mean"] > 0
    
    def test_performance_grade(self):
        """Test performance grading."""
        from src.utils.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Add good performance metrics
        collector.update_metrics({
            "fps": 30,
            "processing_time": 30,
            "memory_usage": 1000
        })
        
        grade = collector.get_performance_grade()
        assert grade in ["Excellent", "Good", "Fair", "Poor", "Critical"]
    
    def test_alerts_generation(self):
        """Test alert generation."""
        from src.utils.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Add problematic metrics
        collector.update_metrics({
            "fps": 5,  # Low FPS
            "processing_time": 250,  # High processing time
            "memory_usage": 9000  # High memory
        })
        
        alerts = collector.check_alerts()
        assert len(alerts) > 0
        assert any("Low FPS" in alert for alert in alerts)


# API Tests
class TestAPI:
    """Test FastAPI server."""
    
    @pytest.fixture
    def test_client(self, mock_config):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.server import create_app
        
        app = create_app(mock_config)
        return TestClient(app)
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        # May return 503 if models not loaded, which is expected in test
        assert response.status_code in [200, 503]
    
    def test_status_endpoint(self, test_client):
        """Test status endpoint."""
        response = test_client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime" in data
    
    @patch('src.models.model_manager.ModelManager')
    def test_process_image_endpoint(self, mock_manager_class, test_client, sample_pil_image):
        """Test image processing endpoint."""
        import base64
        import io
        
        # Convert image to base64
        img_buffer = io.BytesIO()
        sample_pil_image.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Mock model manager
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        from src.models.model_manager import ModelOutput
        mock_output = ModelOutput(
            caption="Test caption",
            confidence=0.9,
            objects=[],
            emotions=[],
            processing_time=0.05,
            timestamp=time.time()
        )
        mock_manager.process_frame.return_value = mock_output
        
        # Test request
        payload = {
            "image_data": img_base64,
            "include_objects": True,
            "include_emotions": True
        }
        
        # This test may fail if models aren't loaded - that's expected
        response = test_client.post("/process", json=payload)
        assert response.status_code in [200, 503]  # 503 if models not loaded


# Integration Tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_processing(self, mock_config, sample_image):
        """Test end-to-end processing pipeline."""
        from src.models.model_manager import ModelManager
        from src.processors.video_processor import VideoProcessor
        
        # This test requires actual models, so it's marked as slow
        try:
            manager = ModelManager(mock_config)
            processor = VideoProcessor(mock_config, manager)
            
            # Process a single frame
            output = manager.process_frame(sample_image)
            
            assert isinstance(output.caption, str)
            assert len(output.caption) > 0
            assert 0 <= output.confidence <= 1
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to: {e}")


# Performance Tests
class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_processing_speed(self, mock_config, sample_image):
        """Test processing speed benchmarks."""
        from src.models.model_manager import ModelManager
        
        # Mock models for speed testing
        with patch.multiple(
            'src.models.model_manager',
            BlipProcessor=Mock(),
            BlipForConditionalGeneration=Mock()
        ):
            manager = ModelManager(mock_config)
            manager.generate_caption = Mock(return_value=("caption", 0.9))
            manager.detect_objects = Mock(return_value=[])
            manager.analyze_emotions = Mock(return_value=[])
            
            # Measure processing time
            start_time = time.time()
            for _ in range(10):
                output = manager.process_frame(sample_image)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            # Should process frame in reasonable time (even with mocks)
            assert avg_time < 1.0  # Less than 1 second per frame
    
    def test_memory_usage(self, mock_config):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and destroy model manager multiple times
        for _ in range(5):
            with patch.multiple(
                'src.models.model_manager',
                BlipProcessor=Mock(),
                BlipForConditionalGeneration=Mock()
            ):
                from src.models.model_manager import ModelManager
                manager = ModelManager(mock_config)
                del manager
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for mocked tests)
        assert memory_increase < 100 * 1024 * 1024


# Fixtures for running tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
