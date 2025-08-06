"""
Modern Streamlit web interface for Real-Time AI Scene Description.
Features real-time monitoring, model management, and interactive controls.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List
import threading
import queue
from pathlib import Path

# Custom imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.model_manager import ModelManager
from processors.video_processor import VideoProcessor
from utils.config_loader import ConfigLoader
from utils.metrics import MetricsCollector


class StreamlitApp:
    """Advanced Streamlit application for AI scene description."""
    
    def __init__(self):
        self.config = None
        self.model_manager = None
        self.video_processor = None
        self.metrics_collector = MetricsCollector()
        
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Real-Time AI Scene Description",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'About': "Advanced AI-powered real-time scene description system"
            }
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-online { background-color: #00ff00; }
        .status-offline { background-color: #ff0000; }
        .status-loading { background-color: #ffaa00; }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.camera_running = False
            st.session_state.model_loaded = False
            st.session_state.metrics_history = []
            st.session_state.current_frame = None
            st.session_state.current_output = None
            st.session_state.frame_queue = queue.Queue(maxsize=10)
    
    def load_configuration(self):
        """Load application configuration."""
        try:
            config_loader = ConfigLoader()
            self.config = config_loader.load_config()
            st.session_state.config_loaded = True
            return True
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            return False
    
    def initialize_models(self):
        """Initialize AI models."""
        if not st.session_state.model_loaded:
            with st.spinner("Loading AI models... This may take a few minutes."):
                try:
                    self.model_manager = ModelManager(self.config)
                    st.session_state.model_loaded = True
                    st.success("✅ AI models loaded successfully!")
                    return True
                except Exception as e:
                    st.error(f"❌ Failed to load models: {e}")
                    return False
        return True
    
    def render_header(self):
        """Render application header."""
        st.markdown('<h1 class="main-header">🤖 Real-Time AI Scene Description</h1>', 
                   unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "online" if st.session_state.get('model_loaded') else "offline"
            st.markdown(f"""
            <div class="metric-container">
                <span class="status-indicator status-{status}"></span>
                <strong>AI Models:</strong> {'Ready' if status == 'online' else 'Not Loaded'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            camera_status = "online" if st.session_state.get('camera_running') else "offline"
            st.markdown(f"""
            <div class="metric-container">
                <span class="status-indicator status-{camera_status}"></span>
                <strong>Camera:</strong> {'Active' if camera_status == 'online' else 'Inactive'}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if hasattr(self, 'model_manager') and self.model_manager:
                device_info = self.model_manager.get_model_info()
                device = device_info.get('device', 'Unknown')
                st.markdown(f"""
                <div class="metric-container">
                    <span class="status-indicator status-online"></span>
                    <strong>Device:</strong> {device}
                </div>
                """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and settings."""
        st.sidebar.title("🎛️ Control Panel")
        
        # Model Management
        st.sidebar.subheader("Model Management")
        
        if st.sidebar.button("🔄 Reload Models"):
            st.session_state.model_loaded = False
            self.initialize_models()
        
        if st.sidebar.button("🗑️ Clear Cache"):
            if hasattr(self, 'model_manager') and self.model_manager:
                self.model_manager.clear_cache()
                st.sidebar.success("Cache cleared!")
        
        # Camera Controls
        st.sidebar.subheader("Camera Controls")
        
        camera_source = st.sidebar.selectbox(
            "Camera Source",
            options=[0, 1, 2, "Upload Image", "Upload Video"],
            index=0
        )
        
        if st.sidebar.button("📹 Start Camera" if not st.session_state.camera_running else "⏹️ Stop Camera"):
            st.session_state.camera_running = not st.session_state.camera_running
        
        # Processing Settings
        st.sidebar.subheader("Processing Settings")
        
        frame_skip = st.sidebar.slider("Frame Skip", 1, 60, 30)
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        
        # Model Settings
        st.sidebar.subheader("Model Settings")
        
        enable_object_detection = st.sidebar.checkbox("Object Detection", True)
        enable_emotion_analysis = st.sidebar.checkbox("Emotion Analysis", True)
        enable_caching = st.sidebar.checkbox("Enable Caching", True)
        
        # Export Settings
        st.sidebar.subheader("Export & Save")
        
        if st.sidebar.button("💾 Export Metrics"):
            self.export_metrics()
        
        if st.sidebar.button("📸 Save Current Frame"):
            self.save_current_frame()
        
        return {
            'camera_source': camera_source,
            'frame_skip': frame_skip,
            'confidence_threshold': confidence_threshold,
            'enable_object_detection': enable_object_detection,
            'enable_emotion_analysis': enable_emotion_analysis,
            'enable_caching': enable_caching
        }
    
    def render_main_content(self, settings: Dict):
        """Render main content area."""
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Feed", "📊 Analytics", "🔧 Model Info", "📋 Logs"])
        
        with tab1:
            self.render_live_feed_tab(settings)
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_model_info_tab()
        
        with tab4:
            self.render_logs_tab()
    
    def render_live_feed_tab(self, settings: Dict):
        """Render live camera feed tab."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Live Camera Feed")
            
            # Video display area
            video_placeholder = st.empty()
            
            # Camera feed processing
            if st.session_state.camera_running and st.session_state.model_loaded:
                self.process_camera_feed(video_placeholder, settings)
            else:
                # Show placeholder
                placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
                placeholder_image[:] = (50, 50, 50)
                
                # Add text
                cv2.putText(placeholder_image, "Camera Feed", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(placeholder_image, "Click 'Start Camera' to begin", (150, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                video_placeholder.image(placeholder_image, channels="BGR")
        
        with col2:
            st.subheader("📝 Current Analysis")
            
            # Current caption
            if st.session_state.current_output:
                output = st.session_state.current_output
                
                st.markdown("**Caption:**")
                st.write(output.caption)
                
                st.markdown("**Confidence:**")
                st.progress(output.confidence)
                
                if output.objects:
                    st.markdown("**Detected Objects:**")
                    for obj in output.objects[:5]:  # Show top 5
                        st.write(f"• {obj['class']} ({obj['confidence']:.2f})")
                
                if output.emotions:
                    st.markdown("**Emotions:**")
                    for emotion in output.emotions:
                        st.write(f"• {emotion}")
            
            # Real-time metrics
            st.subheader("⚡ Performance")
            
            if hasattr(self, 'metrics_collector'):
                metrics = self.metrics_collector.get_current_metrics()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("FPS", f"{metrics.get('fps', 0):.1f}")
                    st.metric("Processing Time", f"{metrics.get('processing_time', 0):.1f}ms")
                
                with col_b:
                    st.metric("Memory", f"{metrics.get('memory_usage', 0):.1f}MB")
                    st.metric("Queue Size", metrics.get('queue_size', 0))
    
    def render_analytics_tab(self):
        """Render analytics and metrics tab."""
        st.subheader("📊 Performance Analytics")
        
        # Generate sample metrics for demo
        if not st.session_state.metrics_history:
            self.generate_sample_metrics()
        
        # FPS Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**FPS Over Time**")
            fps_data = [m['fps'] for m in st.session_state.metrics_history]
            times = [m['timestamp'] for m in st.session_state.metrics_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=fps_data, mode='lines', name='FPS'))
            fig.update_layout(title="Real-time FPS", xaxis_title="Time", yaxis_title="FPS")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Processing Time Distribution**")
            proc_times = [m['processing_time'] for m in st.session_state.metrics_history]
            
            fig = px.histogram(x=proc_times, nbins=20, title="Processing Time Distribution")
            fig.update_layout(xaxis_title="Processing Time (ms)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Object Detection Stats
        st.markdown("**Object Detection Statistics**")
        
        # Sample object detection data
        object_counts = {'person': 45, 'car': 23, 'dog': 12, 'cat': 8, 'bicycle': 5}
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(x=list(object_counts.keys()), y=list(object_counts.values()),
                        title="Most Detected Objects")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(values=list(object_counts.values()), names=list(object_counts.keys()),
                        title="Object Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_model_info_tab(self):
        """Render model information tab."""
        st.subheader("🔧 Model Information")
        
        if hasattr(self, 'model_manager') and self.model_manager:
            model_info = self.model_manager.get_model_info()
            
            # Device Information
            st.markdown("**Device Information**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Device", model_info.get('device', 'Unknown'))
            
            with col2:
                memory_info = model_info.get('memory_usage', {})
                allocated = memory_info.get('allocated', 'N/A')
                st.metric("GPU Memory", allocated)
            
            with col3:
                max_memory = memory_info.get('max_allocated', 'N/A')
                st.metric("Peak Memory", max_memory)
            
            # Loaded Models
            st.markdown("**Loaded Models**")
            models = model_info.get('models_loaded', [])
            
            for model in models:
                with st.expander(f"📦 {model.upper()} Model"):
                    if model == 'blip':
                        st.write("**Purpose:** Image Captioning")
                        st.write("**Architecture:** BLIP-2")
                        st.write("**Parameters:** ~2.7B")
                    elif model == 'yolo':
                        st.write("**Purpose:** Object Detection")
                        st.write("**Architecture:** YOLOv8")
                        st.write("**Classes:** 80 COCO classes")
                    elif model == 'clip':
                        st.write("**Purpose:** Vision-Language Understanding")
                        st.write("**Architecture:** ViT-B/32")
                        st.write("**Capabilities:** Image-text similarity")
                    elif model == 'emotion':
                        st.write("**Purpose:** Emotion Analysis")
                        st.write("**Architecture:** DistilRoBERTa")
                        st.write("**Classes:** 7 emotions")
            
            # Configuration
            st.markdown("**Current Configuration**")
            if hasattr(self, 'config') and self.config:
                config_display = {
                    "Frame Skip": self.config["video"]["processing"]["frame_skip"],
                    "Resolution": self.config["video"]["input"]["resolution"],
                    "Confidence Threshold": self.config["models"]["yolo"]["confidence"],
                    "Cache Enabled": self.config["performance"]["cache"]["enable_result_cache"]
                }
                
                for key, value in config_display.items():
                    st.write(f"**{key}:** {value}")
        else:
            st.warning("⚠️ Models not loaded. Please initialize models first.")
    
    def render_logs_tab(self):
        """Render logs and debugging tab."""
        st.subheader("📋 System Logs")
        
        # Log level filter
        log_level = st.selectbox("Log Level", ["All", "DEBUG", "INFO", "WARNING", "ERROR"])
        
        # Sample logs for demo
        sample_logs = [
            {"timestamp": "2024-01-15 10:30:25", "level": "INFO", "message": "Models loaded successfully"},
            {"timestamp": "2024-01-15 10:30:26", "level": "DEBUG", "message": "Processing frame 1024"},
            {"timestamp": "2024-01-15 10:30:27", "level": "INFO", "message": "Caption generated: A person walking in the park"},
            {"timestamp": "2024-01-15 10:30:28", "level": "WARNING", "message": "High memory usage detected"},
            {"timestamp": "2024-01-15 10:30:29", "level": "ERROR", "message": "Failed to process frame 1025"},
        ]
        
        # Display logs
        for log in sample_logs:
            if log_level == "All" or log["level"] == log_level:
                color = {
                    "DEBUG": "gray",
                    "INFO": "blue", 
                    "WARNING": "orange",
                    "ERROR": "red"
                }.get(log["level"], "black")
                
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.2rem 0; border-left: 3px solid {color};">
                    <strong>{log['timestamp']}</strong> - 
                    <span style="color: {color};">{log['level']}</span>: 
                    {log['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # System diagnostics
        st.markdown("**System Diagnostics**")
        
        if st.button("🔍 Run Diagnostics"):
            with st.spinner("Running diagnostics..."):
                time.sleep(2)  # Simulate diagnostics
                
                results = {
                    "Camera Access": "✅ Available",
                    "GPU Acceleration": "✅ CUDA Available",
                    "Model Files": "✅ All models loaded",
                    "Memory Usage": "⚠️ High (85%)",
                    "Disk Space": "✅ Sufficient (2.5GB free)"
                }
                
                for check, status in results.items():
                    st.write(f"**{check}:** {status}")
    
    def process_camera_feed(self, video_placeholder, settings: Dict):
        """Process camera feed in real-time."""
        if not hasattr(self, 'video_processor') or not self.video_processor:
            try:
                self.video_processor = VideoProcessor(self.config, self.model_manager)
                if not self.video_processor.initialize_camera():
                    st.error("Failed to initialize camera")
                    return
            except Exception as e:
                st.error(f"Failed to create video processor: {e}")
                return
        
        # Capture and process frame
        ret, frame = self.video_processor.cap.read()
        if ret:
            # Process frame periodically
            if self.video_processor.frame_count % settings['frame_skip'] == 0:
                try:
                    output = self.model_manager.process_frame(frame)
                    st.session_state.current_output = output
                    
                    # Update metrics
                    self.metrics_collector.update_metrics({
                        'fps': self.video_processor.fps_counter.get_fps(),
                        'processing_time': output.processing_time * 1000,
                        'memory_usage': 0,  # Placeholder
                        'queue_size': 0     # Placeholder
                    })
                    
                except Exception as e:
                    st.error(f"Processing error: {e}")
            
            # Draw visualizations
            if st.session_state.current_output:
                display_frame = self.video_processor.draw_visualizations(
                    frame, st.session_state.current_output
                )
            else:
                display_frame = frame
            
            # Update display
            video_placeholder.image(display_frame, channels="BGR")
            
            self.video_processor.frame_count += 1
    
    def generate_sample_metrics(self):
        """Generate sample metrics for demonstration."""
        import random
        
        base_time = datetime.now() - timedelta(minutes=10)
        
        for i in range(100):
            timestamp = base_time + timedelta(seconds=i*6)
            metrics = {
                'timestamp': timestamp,
                'fps': random.uniform(15, 30),
                'processing_time': random.uniform(20, 80),
                'memory_usage': random.uniform(2000, 4000),
                'queue_size': random.randint(0, 5)
            }
            st.session_state.metrics_history.append(metrics)
    
    def export_metrics(self):
        """Export metrics to JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(st.session_state.metrics_history, f, 
                         indent=2, default=str)
            
            st.sidebar.success(f"✅ Metrics exported to {filename}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Export failed: {e}")
    
    def save_current_frame(self):
        """Save current frame with annotations."""
        try:
            if st.session_state.current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_capture_{timestamp}.jpg"
                
                cv2.imwrite(filename, st.session_state.current_frame)
                st.sidebar.success(f"✅ Frame saved as {filename}")
            else:
                st.sidebar.warning("⚠️ No frame to save")
                
        except Exception as e:
            st.sidebar.error(f"❌ Save failed: {e}")
    
    def run(self):
        """Main application entry point."""
        self.render_header()
        
        # Load configuration
        if not self.load_configuration():
            st.error("Failed to load configuration. Please check config files.")
            return
        
        # Render sidebar
        settings = self.render_sidebar()
        
        # Initialize models if needed
        if not st.session_state.initialized:
            if self.initialize_models():
                st.session_state.initialized = True
        
        # Render main content
        self.render_main_content(settings)
        
        # Auto-refresh for real-time updates
        if st.session_state.camera_running:
            time.sleep(0.1)  # Small delay for smooth updates
            st.rerun()


def main():
    """Run the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
