"""
Performance metrics collection and monitoring.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import psutil
import numpy as np
from datetime import datetime


@dataclass
class MetricSnapshot:
    """Single metric snapshot."""
    timestamp: float
    fps: float
    processing_time: float
    memory_usage: float
    cpu_usage: float
    gpu_memory: Optional[float] = None
    queue_size: int = 0
    errors: int = 0


class MetricsCollector:
    """Advanced metrics collection and analysis."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = {}
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        self.error_count = 0
        
        # System monitoring
        self.process = psutil.Process()
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update current metrics."""
        with self.lock:
            self.current_metrics.update(metrics)
            self.current_metrics['timestamp'] = time.time()
            
            # Calculate additional metrics
            self._calculate_derived_metrics()
            
            # Add to history
            snapshot = self._create_snapshot()
            self.metrics_history.append(snapshot)
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from raw data."""
        current_time = time.time()
        
        # Update frame timing
        if len(self.frame_times) > 0:
            frame_interval = current_time - self.frame_times[-1]
            if frame_interval > 0:
                self.current_metrics['instantaneous_fps'] = 1.0 / frame_interval
        
        self.frame_times.append(current_time)
        
        # System metrics
        self.current_metrics['cpu_usage'] = self.process.cpu_percent()
        self.current_metrics['memory_usage'] = self.process.memory_info().rss / 1024 / 1024  # MB
        self.current_metrics['uptime'] = current_time - self.start_time
        
        # GPU metrics (if available)
        try:
            import torch
            if torch.cuda.is_available():
                self.current_metrics['gpu_memory'] = torch.cuda.memory_allocated() / 1024 / 1024
                self.current_metrics['gpu_utilization'] = torch.cuda.utilization()
        except ImportError:
            pass
    
    def _create_snapshot(self) -> MetricSnapshot:
        """Create a metric snapshot from current metrics."""
        return MetricSnapshot(
            timestamp=self.current_metrics.get('timestamp', time.time()),
            fps=self.current_metrics.get('fps', 0.0),
            processing_time=self.current_metrics.get('processing_time', 0.0),
            memory_usage=self.current_metrics.get('memory_usage', 0.0),
            cpu_usage=self.current_metrics.get('cpu_usage', 0.0),
            gpu_memory=self.current_metrics.get('gpu_memory'),
            queue_size=self.current_metrics.get('queue_size', 0),
            errors=self.error_count
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_history(self, duration_seconds: Optional[float] = None) -> List[MetricSnapshot]:
        """Get metrics history, optionally filtered by duration."""
        with self.lock:
            if duration_seconds is None:
                return list(self.metrics_history)
            
            cutoff_time = time.time() - duration_seconds
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_statistics(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get statistical analysis of metrics."""
        history = self.get_history(duration_seconds)
        
        if not history:
            return {}
        
        # Convert to numpy arrays for easy calculation
        fps_values = np.array([m.fps for m in history])
        processing_times = np.array([m.processing_time for m in history])
        memory_values = np.array([m.memory_usage for m in history])
        cpu_values = np.array([m.cpu_usage for m in history])
        
        statistics = {
            'fps': {
                'mean': float(np.mean(fps_values)),
                'std': float(np.std(fps_values)),
                'min': float(np.min(fps_values)),
                'max': float(np.max(fps_values)),
                'p50': float(np.percentile(fps_values, 50)),
                'p95': float(np.percentile(fps_values, 95)),
                'p99': float(np.percentile(fps_values, 99))
            },
            'processing_time': {
                'mean': float(np.mean(processing_times)),
                'std': float(np.std(processing_times)),
                'min': float(np.min(processing_times)),
                'max': float(np.max(processing_times)),
                'p50': float(np.percentile(processing_times, 50)),
                'p95': float(np.percentile(processing_times, 95)),
                'p99': float(np.percentile(processing_times, 99))
            },
            'memory': {
                'mean': float(np.mean(memory_values)),
                'max': float(np.max(memory_values)),
                'current': float(memory_values[-1]) if len(memory_values) > 0 else 0
            },
            'cpu': {
                'mean': float(np.mean(cpu_values)),
                'max': float(np.max(cpu_values)),
                'current': float(cpu_values[-1]) if len(cpu_values) > 0 else 0
            },
            'duration': len(history),
            'error_rate': self.error_count / len(history) if len(history) > 0 else 0
        }
        
        return statistics
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error occurrence."""
        self.error_count += 1
        with self.lock:
            self.current_metrics['last_error'] = error_type
            self.current_metrics['last_error_time'] = time.time()
    
    def get_performance_grade(self) -> str:
        """Get overall performance grade based on metrics."""
        stats = self.get_statistics(duration_seconds=60)  # Last minute
        
        if not stats:
            return "Unknown"
        
        # Scoring criteria
        fps_score = min(stats['fps']['mean'] / 30, 1.0)  # Target 30 FPS
        processing_score = max(0, 1 - (stats['processing_time']['mean'] / 100))  # Target <100ms
        memory_score = max(0, 1 - (stats['memory']['current'] / 4000))  # Target <4GB
        error_score = max(0, 1 - stats['error_rate'])
        
        overall_score = (fps_score + processing_score + memory_score + error_score) / 4
        
        if overall_score >= 0.9:
            return "Excellent"
        elif overall_score >= 0.8:
            return "Good"
        elif overall_score >= 0.6:
            return "Fair"
        elif overall_score >= 0.4:
            return "Poor"
        else:
            return "Critical"
    
    def check_alerts(self) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        current = self.get_current_metrics()
        
        # FPS alerts
        if current.get('fps', 0) < 10:
            alerts.append("⚠️ Low FPS detected")
        
        # Processing time alerts
        if current.get('processing_time', 0) > 200:  # >200ms
            alerts.append("⚠️ High processing latency")
        
        # Memory alerts
        if current.get('memory_usage', 0) > 8000:  # >8GB
            alerts.append("⚠️ High memory usage")
        
        # Error rate alerts
        stats = self.get_statistics(duration_seconds=300)  # Last 5 minutes
        if stats and stats['error_rate'] > 0.1:  # >10% error rate
            alerts.append("🚨 High error rate")
        
        return alerts
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        history = self.get_history()
        
        if format.lower() == "json":
            import json
            data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_samples': len(history),
                    'duration_seconds': time.time() - self.start_time
                },
                'statistics': self.get_statistics(),
                'metrics': [asdict(snapshot) for snapshot in history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(filepath, 'w', newline='') as f:
                if history:
                    writer = csv.DictWriter(f, fieldnames=asdict(history[0]).keys())
                    writer.writeheader()
                    for snapshot in history:
                        writer.writerow(asdict(snapshot))
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset(self):
        """Reset all metrics and history."""
        with self.lock:
            self.metrics_history.clear()
            self.current_metrics.clear()
            self.frame_times.clear()
            self.processing_times.clear()
            self.error_count = 0
            self.start_time = time.time()


class PerformanceMonitor:
    """High-level performance monitoring interface."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring_enabled = True
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background monitoring thread."""
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    # Check for alerts
                    alerts = self.metrics.check_alerts()
                    if alerts:
                        for callback in self.alert_callbacks:
                            try:
                                callback(alerts)
                            except Exception as e:
                                print(f"Alert callback error: {e}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_enabled = False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        current = self.metrics.get_current_metrics()
        stats = self.metrics.get_statistics(duration_seconds=300)  # Last 5 minutes
        grade = self.metrics.get_performance_grade()
        alerts = self.metrics.check_alerts()
        
        return {
            'current': current,
            'statistics': stats,
            'grade': grade,
            'alerts': alerts,
            'uptime': current.get('uptime', 0)
        }
