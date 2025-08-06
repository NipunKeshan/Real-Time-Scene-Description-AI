"""
Main application entry point for Real-Time AI Scene Description.
Supports multiple modes: CLI, Streamlit UI, and API server.
"""

import sys
import argparse
from pathlib import Path
import yaml
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.config_loader import ConfigLoader
from models.model_manager import ModelManager
from processors.video_processor import VideoProcessor


def setup_logging(config: dict):
    """Setup structured logging."""
    try:
        from loguru import logger
        import sys
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=config["logging"]["format"],
            level=config["logging"]["level"],
            colorize=True
        )
        
        # Add file handlers
        if "files" in config["logging"]:
            for file_config in config["logging"]["files"]:
                logger.add(
                    file_config["sink"],
                    format=config["logging"]["format"],
                    level=file_config.get("level", "INFO"),
                    rotation=config["logging"].get("rotation", "1 day"),
                    retention=config["logging"].get("retention", "30 days"),
                    backtrace=config["logging"].get("backtrace", True),
                    diagnose=config["logging"].get("diagnose", True),
                    filter=file_config.get("filter")
                )
        
        return logger
        
    except ImportError:
        # Fallback to standard logging
        import logging
        
        logging.basicConfig(
            level=getattr(logging, config["logging"]["level"]),
            format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
        
        return logging.getLogger(__name__)


def run_cli_mode(config: dict, args):
    """Run in command-line interface mode."""
    logger = setup_logging(config)
    logger.info("Starting CLI mode...")
    
    try:
        # Initialize models
        logger.info("Loading AI models...")
        model_manager = ModelManager(config)
        
        # Initialize video processor
        logger.info("Initializing video processor...")
        video_processor = VideoProcessor(config, model_manager)
        
        # Start processing
        logger.info("Starting video processing...")
        video_processor.process_video_stream()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def run_streamlit_mode(config: dict, args):
    """Run Streamlit web interface."""
    logger = setup_logging(config)
    logger.info("Starting Streamlit mode...")
    
    try:
        import subprocess
        import os
        
        # Set environment variables for Streamlit
        env = os.environ.copy()
        env['RTSD_CONFIG_PATH'] = str(Path(__file__).parent.parent / "config" / "config.yaml")
        
        # Streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).parent / "ui" / "streamlit_app.py"),
            "--server.port", str(config["ui"]["streamlit"]["port"]),
            "--server.address", config["ui"]["streamlit"]["host"],
            "--theme.base", config["ui"]["streamlit"]["theme"]
        ]
        
        logger.info(f"Starting Streamlit on {config['ui']['streamlit']['host']}:{config['ui']['streamlit']['port']}")
        subprocess.run(cmd, env=env)
        
    except Exception as e:
        logger.error(f"Streamlit startup error: {e}")
        sys.exit(1)


def run_api_mode(config: dict, args):
    """Run FastAPI REST server."""
    logger = setup_logging(config)
    logger.info("Starting API mode...")
    
    try:
        from api.server import create_app
        import uvicorn
        
        # Create FastAPI app
        app = create_app(config)
        
        # Run server
        uvicorn.run(
            app,
            host=config["api"]["host"],
            port=config["api"]["port"],
            workers=config["api"]["workers"]
        )
        
    except Exception as e:
        logger.error(f"API server error: {e}")
        sys.exit(1)


def run_benchmark_mode(config: dict, args):
    """Run benchmark tests."""
    logger = setup_logging(config)
    logger.info("Starting benchmark mode...")
    
    try:
        from utils.benchmark import BenchmarkRunner
        
        benchmark = BenchmarkRunner(config)
        results = benchmark.run_all_benchmarks()
        
        logger.info("Benchmark Results:")
        for test_name, result in results.items():
            logger.info(f"  {test_name}: {result}")
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        sys.exit(1)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time AI Scene Description System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode cli                    # Run with OpenCV display
  python -m src.main --mode streamlit              # Run Streamlit web interface
  python -m src.main --mode api                    # Run REST API server
  python -m src.main --mode benchmark              # Run performance benchmarks
  
Environment Variables:
  RTSD_DEBUG=true                                  # Enable debug mode
  RTSD_DEVICE=cuda                                 # Force GPU usage
  RTSD_CAMERA_ID=1                                 # Use camera 1
  RTSD_LOG_LEVEL=DEBUG                            # Set log level
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "streamlit", "api", "benchmark"],
        default="cli",
        help="Application mode (default: cli)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Force specific device"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera ID to use"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI display (CLI mode)"
    )
    
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save processed frames and metadata"
    )
    
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run quick benchmark and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_loader = ConfigLoader(args.config)
        config = config_loader.load_config()
        
        # Apply command-line overrides
        if args.device:
            config["models"]["blip"]["device"] = args.device
            config["models"]["clip"]["device"] = args.device
            config["models"]["yolo"]["device"] = args.device
        
        if args.camera is not None:
            config["video"]["input"]["camera_id"] = args.camera
        
        if args.debug:
            config["app"]["debug"] = True
            config["app"]["log_level"] = "DEBUG"
        
        if args.save_output:
            config["video"]["output"]["save_video"] = True
        
        # Quick benchmark mode
        if args.benchmark_only:
            run_benchmark_mode(config, args)
            return
        
        # Run application in specified mode
        if args.mode == "cli":
            run_cli_mode(config, args)
        elif args.mode == "streamlit":
            run_streamlit_mode(config, args)
        elif args.mode == "api":
            run_api_mode(config, args)
        elif args.mode == "benchmark":
            run_benchmark_mode(config, args)
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
