"""
Real-Time AI Scene Description System - Legacy Interface
This file provides backward compatibility with the original simple implementation.

For the advanced features, use the new modular system:
- Run with CLI: python -m src.main --mode cli
- Run with Web UI: python -m src.main --mode streamlit  
- Run API server: python -m src.main --mode api

This legacy file is maintained for compatibility and quick testing.
"""

import warnings
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from utils.config_loader import ConfigLoader
    from models.model_manager import ModelManager
    from processors.video_processor import VideoProcessor
    
    def main():
        """Main function using the advanced system."""
        print("🤖 Real-Time AI Scene Description System v2.0")
        print("=" * 50)
        print("Loading advanced AI models and configuration...")
        
        try:
            # Load configuration
            config_loader = ConfigLoader()
            config = config_loader.load_config()
            
            # Initialize model manager
            print("📦 Loading AI models (BLIP-2, YOLO, CLIP, Emotion Analysis)...")
            model_manager = ModelManager(config)
            
            # Initialize video processor
            print("🎥 Initializing advanced video processor...")
            video_processor = VideoProcessor(config, model_manager)
            
            print("✅ System ready! Press 'q' to quit.")
            print("🌟 Features: Smart captioning, object detection, emotion analysis")
            print("-" * 50)
            
            # Start processing
            video_processor.process_video_stream()
            
        except Exception as e:
            print(f"❌ Advanced system failed: {e}")
            print("🔄 Falling back to simple implementation...")
            run_simple_version()
    
    def run_simple_version():
        """Fallback to original simple implementation."""
        import cv2
        import torch
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        warnings.warn(
            "Using legacy simple implementation. "
            "For advanced features, ensure all dependencies are installed.",
            DeprecationWarning
        )
        
        print("Loading basic BLIP model...")
        
        # Load BLIP processor and model (original implementation)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        print(f"✅ Model loaded on: {device}")
        
        # Caption generation function (original)
        def generate_caption(frame):
            image = Image.fromarray(frame)
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        
        # Start camera (original implementation)
        cap = cv2.VideoCapture(0)
        frame_count = 0
        caption = ""
        
        print("🎥 Camera started. Press 'q' to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Resize for captioning model input (optional)
            resized_frame = cv2.resize(frame, (384, 384))
            
            # Generate caption every 30 frames (~1 sec if 30 FPS)
            if frame_count % 30 == 0:
                try:
                    caption = generate_caption(resized_frame)
                    print(f"📝 Caption: {caption}")
                except Exception as e:
                    print(f"❌ Captioning error: {e}")
                    caption = "Error generating caption"
            
            frame_count += 1
            
            # Show caption on the video
            cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Real-Time AI Scene Description (Legacy)", frame)
            
            # Quit with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("🔄 Camera released. Goodbye!")

except ImportError as e:
    print(f"⚠️ Advanced modules not available: {e}")
    print("🔄 Running in legacy mode...")
    
    def main():
        """Fallback main function."""
        run_simple_version()
    
    def run_simple_version():
        """Original implementation for backward compatibility."""
        import cv2
        import torch
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print("🤖 Real-Time AI Scene Description System (Legacy Mode)")
        print("=" * 50)
        print("📦 Loading BLIP model...")
        
        # Load BLIP processor and model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        print(f"✅ Model loaded on: {device}")
        
        # Caption generation function
        def generate_caption(frame):
            image = Image.fromarray(frame)
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        
        # Start camera
        cap = cv2.VideoCapture(0)
        frame_count = 0
        caption = ""
        
        print("🎥 Camera started. Press 'q' to quit...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for captioning model input (optional)
            resized_frame = cv2.resize(frame, (384, 384))
            
            # Generate caption every 30 frames (~1 sec if 30 FPS)
            if frame_count % 30 == 0:
                try:
                    caption = generate_caption(resized_frame)
                    print("Caption:", caption)
                except Exception as e:
                    print("Captioning error:", e)
                    caption = ""
            
            frame_count += 1
            
            # Show caption on the video
            cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Real-Time AI Scene Description", frame)
            
            # Quit with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "🌟" * 20)
    print("Real-Time AI Scene Description System")
    print("🌟" * 20)
    print("\n💡 TIP: For advanced features, try:")
    print("   python -m src.main --mode streamlit  # Web interface")
    print("   python -m src.main --mode api        # REST API")
    print("   python -m src.main --mode cli        # Advanced CLI")
    print("\n🚀 Starting application...")
    print("-" * 50)
    
    main()
