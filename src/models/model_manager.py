"""
Advanced AI Model Manager with multi-modal capabilities.
Handles BLIP-2, CLIP, YOLOv8, and emotion analysis models.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
from pathlib import Path
import time
from dataclasses import dataclass
from loguru import logger

# Model imports
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    pipeline
)
import clip
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer


@dataclass
class ModelOutput:
    """Structured output from AI models."""
    caption: str
    confidence: float
    objects: List[Dict]
    emotions: List[str]
    processing_time: float
    timestamp: float


class ModelManager:
    """Advanced AI model manager with optimization and caching."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = self._get_device()
        self.models = {}
        self.processors = {}
        self.cache = {}
        
        logger.info(f"Initializing ModelManager on device: {self.device}")
        self._load_models()
    
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available() and self.config.get("device", "auto") != "cpu":
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    
    def _load_models(self):
        """Load all AI models with optimizations."""
        start_time = time.time()
        
        # Load BLIP-2 for image captioning
        self._load_blip_model()
        
        # Load CLIP for vision-language understanding
        self._load_clip_model()
        
        # Load YOLO for object detection
        self._load_yolo_model()
        
        # Load emotion analysis model
        self._load_emotion_model()
        
        load_time = time.time() - start_time
        logger.info(f"All models loaded in {load_time:.2f}s")
    
    def _load_blip_model(self):
        """Load BLIP-2 model with optimizations."""
        try:
            model_name = self.config["models"]["blip"]["model_name"]
            logger.info(f"Loading BLIP model: {model_name}")
            
            # Use BLIP-2 for better performance
            if "blip2" in model_name.lower():
                self.processors["blip"] = Blip2Processor.from_pretrained(model_name)
                self.models["blip"] = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config["models"]["blip"]["precision"] == "fp16" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    load_in_8bit=self.config["models"]["blip"]["precision"] == "int8"
                )
            else:
                self.processors["blip"] = BlipProcessor.from_pretrained(model_name)
                self.models["blip"] = BlipForConditionalGeneration.from_pretrained(model_name)
                self.models["blip"].to(self.device)
            
            if self.device.type == "cuda":
                self.models["blip"].half()
            
            logger.success("BLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            # Fallback to base model
            self._load_fallback_blip()
    
    def _load_fallback_blip(self):
        """Load fallback BLIP model."""
        try:
            model_name = "Salesforce/blip-image-captioning-base"
            logger.info(f"Loading fallback BLIP model: {model_name}")
            
            self.processors["blip"] = BlipProcessor.from_pretrained(model_name)
            self.models["blip"] = BlipForConditionalGeneration.from_pretrained(model_name)
            self.models["blip"].to(self.device)
            
            logger.success("Fallback BLIP model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load fallback BLIP model: {e}")
    
    def _load_clip_model(self):
        """Load CLIP model for vision-language understanding."""
        try:
            model_name = self.config["models"]["clip"]["model_name"]
            logger.info(f"Loading CLIP model: {model_name}")
            
            self.models["clip"], self.processors["clip"] = clip.load(
                model_name, device=self.device
            )
            
            logger.success("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
    
    def _load_yolo_model(self):
        """Load YOLO model for object detection."""
        try:
            model_name = self.config["models"]["yolo"]["model_name"]
            logger.info(f"Loading YOLO model: {model_name}")
            
            self.models["yolo"] = YOLO(model_name)
            if self.device.type == "cuda":
                self.models["yolo"].to(self.device)
            
            logger.success("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def _load_emotion_model(self):
        """Load emotion analysis model."""
        try:
            model_name = self.config["models"]["emotion"]["model_name"]
            logger.info(f"Loading emotion model: {model_name}")
            
            self.models["emotion"] = pipeline(
                "text-classification",
                model=model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.success("Emotion model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
    
    def generate_caption(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, float]:
        """Generate image caption with confidence score."""
        start_time = time.time()
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Check cache
            cache_key = hash(image.tobytes())
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Generate caption
            if "blip" in self.models:
                inputs = self.processors["blip"](
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.models["blip"].generate(
                        **inputs,
                        max_new_tokens=self.config["models"]["blip"]["max_new_tokens"],
                        temperature=self.config["models"]["blip"]["temperature"],
                        do_sample=True,
                        num_return_sequences=1
                    )
                
                caption = self.processors["blip"].decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # Calculate confidence (simplified)
                confidence = min(len(caption.split()) / 10, 1.0)
                
                # Cache result
                self.cache[cache_key] = (caption, confidence)
                
                processing_time = time.time() - start_time
                logger.debug(f"Caption generated in {processing_time:.3f}s: {caption}")
                
                return caption, confidence
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "Unable to generate caption", 0.0
        
        return "No caption model available", 0.0
    
    def detect_objects(self, image: Union[Image.Image, np.ndarray]) -> List[Dict]:
        """Detect objects in image using YOLO."""
        try:
            if "yolo" not in self.models:
                return []
            
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            results = self.models["yolo"](
                image,
                conf=self.config["models"]["yolo"]["confidence"],
                iou=self.config["models"]["yolo"]["iou_threshold"],
                max_det=self.config["models"]["yolo"]["max_detections"]
            )
            
            objects = []
            for result in results:
                for box in result.boxes:
                    objects.append({
                        "class": result.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "center": [(box.xyxy[0][0] + box.xyxy[0][2]) / 2, 
                                  (box.xyxy[0][1] + box.xyxy[0][3]) / 2]
                    })
            
            return objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def analyze_emotions(self, text: str) -> List[str]:
        """Analyze emotions from text."""
        try:
            if "emotion" not in self.models or not text:
                return []
            
            results = self.models["emotion"](text)
            emotions = [result["label"] for result in results if result["score"] > 0.3]
            
            return emotions
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray) -> ModelOutput:
        """Process a video frame with all models."""
        start_time = time.time()
        
        # Convert frame to PIL Image
        image = Image.fromarray(frame)
        
        # Generate caption
        caption, confidence = self.generate_caption(image)
        
        # Detect objects
        objects = self.detect_objects(frame)
        
        # Analyze emotions from caption
        emotions = self.analyze_emotions(caption)
        
        processing_time = time.time() - start_time
        
        return ModelOutput(
            caption=caption,
            confidence=confidence,
            objects=objects,
            emotions=emotions,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            "device": str(self.device),
            "models_loaded": list(self.models.keys()),
            "memory_usage": {}
        }
        
        if self.device.type == "cuda":
            info["memory_usage"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            }
        
        return info
    
    def clear_cache(self):
        """Clear model cache."""
        self.cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
