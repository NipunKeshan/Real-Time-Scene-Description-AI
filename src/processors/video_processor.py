"""
Advanced video processing pipeline with real-time optimization.
Handles camera input, frame processing, and output visualization.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from ..models.model_manager import ModelManager, ModelOutput


@dataclass
class FrameMetrics:
    """Metrics for frame processing performance."""
    frame_id: int
    timestamp: float
    processing_time: float
    queue_size: int
    fps: float


class VideoProcessor:
    """Advanced video processing with real-time optimization."""
    
    def __init__(self, config: Dict, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        
        # Video capture settings
        self.camera_id = config["video"]["input"]["camera_id"]
        self.resolution = tuple(config["video"]["input"]["resolution"])
        self.target_fps = config["video"]["input"]["fps"]
        
        # Processing settings
        self.frame_skip = config["video"]["processing"]["frame_skip"]
        self.inference_size = tuple(config["video"]["processing"]["resize_for_inference"])
        
        # State variables
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.last_caption = ""
        self.last_objects = []
        self.fps_counter = FPSCounter()
        
        # Threading for async processing
        self.enable_async = config["performance"]["threading"]["enable_async_processing"]
        self.processing_queue = Queue(maxsize=config["performance"]["threading"]["queue_size"])
        self.result_queue = Queue()
        self.worker_threads = []
        
        # Visualization settings
        self.display_config = config["ui"]["display"]
        
        logger.info("VideoProcessor initialized")
    
    def initialize_camera(self) -> bool:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Enable GPU decode if available
            if self.config["video"]["processing"]["enable_gpu_decode"]:
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                except:
                    logger.warning("GPU decode not available, using CPU")
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def start_processing_threads(self):
        """Start async processing threads."""
        if not self.enable_async:
            return
        
        num_threads = self.config["performance"]["threading"]["worker_threads"]
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=self._processing_worker,
                name=f"ProcessingThread-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"Started {num_threads} processing threads")
    
    def _processing_worker(self):
        """Worker thread for async frame processing."""
        while self.is_running:
            try:
                frame_data = self.processing_queue.get(timeout=1.0)
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, frame_id, timestamp = frame_data
                
                # Process frame with AI models
                start_time = time.time()
                result = self.model_manager.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Put result in result queue
                self.result_queue.put((frame_id, result, processing_time))
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal inference."""
        # Resize for inference if needed
        if frame.shape[:2] != self.inference_size[::-1]:
            frame = cv2.resize(frame, self.inference_size)
        
        # Color space conversion if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def draw_visualizations(self, frame: np.ndarray, output: ModelOutput) -> np.ndarray:
        """Draw AI predictions on frame."""
        vis_frame = frame.copy()
        
        # Draw caption
        if output.caption:
            self._draw_caption(vis_frame, output.caption, output.confidence)
        
        # Draw object detection boxes
        if output.objects:
            self._draw_objects(vis_frame, output.objects)
        
        # Draw emotions
        if output.emotions:
            self._draw_emotions(vis_frame, output.emotions)
        
        # Draw performance metrics
        self._draw_metrics(vis_frame, output)
        
        return vis_frame
    
    def _draw_caption(self, frame: np.ndarray, caption: str, confidence: float):
        """Draw caption with confidence score."""
        pos = tuple(self.display_config["caption_position"])
        font = getattr(cv2, self.display_config["caption_font"])
        scale = self.display_config["caption_scale"]
        color = tuple(self.display_config["caption_color"])
        thickness = self.display_config["caption_thickness"]
        
        # Prepare text with confidence
        text = f"{caption} ({confidence:.2f})"
        
        # Add background if enabled
        if self.display_config.get("label_background", True):
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(
                frame,
                (pos[0] - 5, pos[1] - text_height - 10),
                (pos[0] + text_width + 5, pos[1] + 5),
                (0, 0, 0),
                -1
            )
        
        cv2.putText(frame, text, pos, font, scale, color, thickness)
    
    def _draw_objects(self, frame: np.ndarray, objects: List[Dict]):
        """Draw object detection bounding boxes."""
        bbox_color = tuple(self.display_config["bbox_color"])
        bbox_thickness = self.display_config["bbox_thickness"]
        
        for obj in objects:
            bbox = obj["bbox"]
            class_name = obj["class"]
            confidence = obj["confidence"]
            
            # Draw bounding box
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(frame, pt1, pt2, bbox_color, bbox_thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(
                frame,
                (pt1[0], pt1[1] - label_size[1] - 10),
                (pt1[0] + label_size[0], pt1[1]),
                bbox_color,
                -1
            )
            
            cv2.putText(
                frame, label,
                (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
    
    def _draw_emotions(self, frame: np.ndarray, emotions: List[str]):
        """Draw detected emotions."""
        if not emotions:
            return
        
        y_offset = 70
        for i, emotion in enumerate(emotions[:3]):  # Show top 3 emotions
            cv2.putText(
                frame, f"Emotion: {emotion}",
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2
            )
    
    def _draw_metrics(self, frame: np.ndarray, output: ModelOutput):
        """Draw performance metrics."""
        if not self.config["video"]["output"]["display_fps"]:
            return
        
        fps = self.fps_counter.get_fps()
        processing_time = output.processing_time * 1000  # Convert to ms
        
        metrics_text = [
            f"FPS: {fps:.1f}",
            f"Processing: {processing_time:.1f}ms",
            f"Objects: {len(output.objects)}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(
                frame, text,
                (frame.shape[1] - 200, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2
            )
    
    def process_video_stream(self, callback: Optional[Callable] = None):
        """Main video processing loop."""
        if not self.initialize_camera():
            return
        
        self.is_running = True
        self.start_processing_threads()
        
        logger.info("Starting video processing...")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                self.frame_count += 1
                self.fps_counter.update()
                
                # Process frame for AI inference
                should_process = (self.frame_count % self.frame_skip == 0)
                
                if should_process:
                    processed_frame = self.preprocess_frame(frame)
                    
                    if self.enable_async:
                        # Async processing
                        try:
                            self.processing_queue.put_nowait((
                                processed_frame, self.frame_count, time.time()
                            ))
                        except:
                            pass  # Queue full, skip frame
                    else:
                        # Sync processing
                        output = self.model_manager.process_frame(processed_frame)
                        self.last_caption = output.caption
                        self.last_objects = output.objects
                
                # Check for async results
                if self.enable_async:
                    try:
                        frame_id, result, proc_time = self.result_queue.get_nowait()
                        self.last_caption = result.caption
                        self.last_objects = result.objects
                    except Empty:
                        pass
                
                # Create output with latest results
                current_output = ModelOutput(
                    caption=self.last_caption,
                    confidence=1.0,
                    objects=self.last_objects,
                    emotions=[],
                    processing_time=0.0,
                    timestamp=time.time()
                )
                
                # Draw visualizations
                display_frame = self.draw_visualizations(frame, current_output)
                
                # Callback for external processing
                if callback:
                    callback(display_frame, current_output)
                
                # Display frame
                cv2.imshow("Real-Time AI Scene Description", display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Video processing error: {e}")
        finally:
            self.cleanup()
    
    def save_frame(self, frame: np.ndarray, output: ModelOutput):
        """Save frame with metadata."""
        if not self.config["video"]["output"]["save_video"]:
            return
        
        output_dir = Path(self.config["video"]["output"]["output_path"])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        frame_path = output_dir / f"frame_{timestamp}.jpg"
        
        cv2.imwrite(str(frame_path), frame)
        
        # Save metadata
        metadata_path = output_dir / f"frame_{timestamp}.json"
        metadata = {
            "caption": output.caption,
            "confidence": output.confidence,
            "objects": output.objects,
            "emotions": output.emotions,
            "timestamp": output.timestamp
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up video processor...")
        
        self.is_running = False
        
        # Stop processing threads
        if self.enable_async:
            for _ in self.worker_threads:
                self.processing_queue.put(None)  # Shutdown signal
            
            for thread in self.worker_threads:
                thread.join(timeout=2.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        logger.info("Video processor cleanup complete")


class FPSCounter:
    """FPS counter for performance monitoring."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        self.last_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
