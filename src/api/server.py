"""
FastAPI REST API server for Real-Time AI Scene Description.
Provides endpoints for image processing, model management, and system monitoring.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import asyncio
from pathlib import Path

# Custom imports
from ..models.model_manager import ModelManager, ModelOutput
from ..utils.metrics import MetricsCollector


class ImageProcessRequest(BaseModel):
    """Request model for image processing."""
    image_data: str  # Base64 encoded image
    include_objects: bool = True
    include_emotions: bool = True
    confidence_threshold: float = 0.5


class ImageProcessResponse(BaseModel):
    """Response model for image processing."""
    caption: str
    confidence: float
    objects: List[Dict[str, Any]]
    emotions: List[str]
    processing_time: float
    timestamp: float


class SystemStatus(BaseModel):
    """System status response."""
    status: str
    uptime: float
    models_loaded: List[str]
    device: str
    memory_usage: Dict[str, Any]
    performance_grade: str


def create_app(config: Dict[str, Any]) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Real-Time AI Scene Description API",
        description="Advanced AI-powered scene description with real-time capabilities",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    if config["api"]["enable_cors"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config["security"]["cors_origins"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize components
    model_manager = None
    metrics_collector = MetricsCollector()
    start_time = time.time()
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize models on startup."""
        nonlocal model_manager
        try:
            model_manager = ModelManager(config)
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Real-Time AI Scene Description API",
            "version": "2.0.0",
            "status": "online" if model_manager else "initializing",
            "documentation": "/docs",
            "endpoints": {
                "process_image": "/process",
                "system_status": "/status",
                "metrics": "/metrics",
                "health": "/health"
            }
        }
    
    @app.post("/process", response_model=ImageProcessResponse)
    async def process_image(request: ImageProcessRequest):
        """Process image and return AI analysis."""
        if not model_manager:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(request.image_data)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array
            frame = np.array(image)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process with AI models
            start_time = time.time()
            output = model_manager.process_frame(frame)
            
            # Update metrics
            metrics_collector.update_metrics({
                'processing_time': output.processing_time * 1000,
                'fps': 1.0 / output.processing_time if output.processing_time > 0 else 0
            })
            
            return ImageProcessResponse(
                caption=output.caption,
                confidence=output.confidence,
                objects=output.objects if request.include_objects else [],
                emotions=output.emotions if request.include_emotions else [],
                processing_time=output.processing_time,
                timestamp=output.timestamp
            )
            
        except Exception as e:
            metrics_collector.record_error("processing_error")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    @app.post("/process_file")
    async def process_uploaded_file(file: UploadFile = File(...)):
        """Process uploaded image file."""
        if not model_manager:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        try:
            # Read file
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Convert to numpy array
            frame = np.array(image)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process with AI models
            output = model_manager.process_frame(frame)
            
            return {
                "filename": file.filename,
                "caption": output.caption,
                "confidence": output.confidence,
                "objects": output.objects,
                "emotions": output.emotions,
                "processing_time": output.processing_time
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
    
    @app.get("/status", response_model=SystemStatus)
    async def get_system_status():
        """Get system status and health information."""
        uptime = time.time() - start_time
        
        if model_manager:
            model_info = model_manager.get_model_info()
            models_loaded = model_info.get('models_loaded', [])
            device = model_info.get('device', 'unknown')
            memory_usage = model_info.get('memory_usage', {})
        else:
            models_loaded = []
            device = "unknown"
            memory_usage = {}
        
        performance_grade = metrics_collector.get_performance_grade()
        
        return SystemStatus(
            status="online" if model_manager else "initializing",
            uptime=uptime,
            models_loaded=models_loaded,
            device=device,
            memory_usage=memory_usage,
            performance_grade=performance_grade
        )
    
    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics and statistics."""
        current_metrics = metrics_collector.get_current_metrics()
        statistics = metrics_collector.get_statistics(duration_seconds=300)  # Last 5 minutes
        alerts = metrics_collector.check_alerts()
        
        return {
            "current": current_metrics,
            "statistics": statistics,
            "alerts": alerts,
            "performance_grade": metrics_collector.get_performance_grade()
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers."""
        if not model_manager:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        # Perform quick health checks
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "models_loaded": bool(model_manager),
                "memory_ok": True,  # Could add actual memory check
                "gpu_available": False  # Could add GPU check
            }
        }
        
        try:
            import torch
            health_status["checks"]["gpu_available"] = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Check if any critical issues
        alerts = metrics_collector.check_alerts()
        critical_alerts = [alert for alert in alerts if "🚨" in alert]
        
        if critical_alerts:
            health_status["status"] = "degraded"
            health_status["alerts"] = critical_alerts
        
        return health_status
    
    @app.get("/models")
    async def get_model_info():
        """Get detailed model information."""
        if not model_manager:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        return model_manager.get_model_info()
    
    @app.post("/models/reload")
    async def reload_models(background_tasks: BackgroundTasks):
        """Reload all AI models."""
        def reload_task():
            nonlocal model_manager
            try:
                model_manager = ModelManager(config)
                print("✅ Models reloaded successfully")
            except Exception as e:
                print(f"❌ Model reload failed: {e}")
        
        background_tasks.add_task(reload_task)
        return {"message": "Model reload initiated"}
    
    @app.post("/cache/clear")
    async def clear_cache():
        """Clear model caches."""
        if model_manager:
            model_manager.clear_cache()
        
        metrics_collector.reset()
        return {"message": "Caches cleared successfully"}
    
    @app.get("/export/metrics")
    async def export_metrics(format: str = "json"):
        """Export metrics to file."""
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        try:
            timestamp = int(time.time())
            filename = f"metrics_export_{timestamp}.{format}"
            filepath = f"/tmp/{filename}"
            
            metrics_collector.export_metrics(filepath, format)
            
            # Read file and return as response
            with open(filepath, 'rb') as f:
                content = f.read()
            
            media_type = "application/json" if format == "json" else "text/csv"
            
            return StreamingResponse(
                io.BytesIO(content),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    
    # WebSocket endpoint for real-time updates
    @app.websocket("/ws/metrics")
    async def websocket_metrics(websocket):
        """WebSocket endpoint for real-time metrics."""
        await websocket.accept()
        
        try:
            while True:
                # Send current metrics
                metrics = {
                    "current": metrics_collector.get_current_metrics(),
                    "alerts": metrics_collector.check_alerts(),
                    "timestamp": time.time()
                }
                
                await websocket.send_json(metrics)
                await asyncio.sleep(1.0)  # Update every second
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await websocket.close()
    
    return app


def main():
    """Run the API server directly."""
    import uvicorn
    from ..utils.config_loader import ConfigLoader
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Create app
    app = create_app(config)
    
    # Run server
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        workers=config["api"]["workers"]
    )


if __name__ == "__main__":
    main()
