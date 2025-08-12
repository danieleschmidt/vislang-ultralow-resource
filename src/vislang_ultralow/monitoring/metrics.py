"""Advanced metrics collection and performance monitoring."""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
try:
    import psutil
except ImportError:
    # Fallback psutil implementation
    class psutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 15.0  # Mock CPU usage
        
        @staticmethod
        def virtual_memory():
            class Memory:
                total = 8 * 1024 * 1024 * 1024  # 8GB
                available = 4 * 1024 * 1024 * 1024  # 4GB available
                percent = 50.0
                used = total - available
            return Memory()
        
        @staticmethod
        def disk_usage(path):
            class Disk:
                total = 100 * 1024 * 1024 * 1024  # 100GB
                used = 50 * 1024 * 1024 * 1024  # 50GB used
                free = total - used
                percent = (used / total) * 100
            return Disk()
import gc

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMING = "timing"
    RATE = "rate"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MetricSeries:
    """Time series of metric points."""
    name: str
    metric_type: MetricType
    description: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: Union[int, float], labels: Dict[str, str] = None, 
                 metadata: Dict[str, Any] = None):
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        self.points.append(point)


class MetricsCollector:
    """Advanced metrics collection system with real-time analytics."""
    
    def __init__(self, max_series: int = 10000, retention_hours: int = 24):
        self.metrics = {}
        self.max_series = max_series
        self.retention_seconds = retention_hours * 3600
        self.collectors = []
        self.is_collecting = False
        self.collection_thread = None
        self.lock = threading.RLock()
        
        # Performance tracking
        self.performance_stats = {
            'collection_time': deque(maxlen=100),
            'metrics_count': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
        
        logger.info(f"Metrics collector initialized (max_series={max_series}, retention={retention_hours}h)")
    
    def create_counter(self, name: str, description: str = "", labels: Dict[str, str] = None) -> str:
        """Create a counter metric."""
        return self._create_metric(name, MetricType.COUNTER, description, labels)
    
    def create_gauge(self, name: str, description: str = "", labels: Dict[str, str] = None) -> str:
        """Create a gauge metric."""
        return self._create_metric(name, MetricType.GAUGE, description, labels)
    
    def create_histogram(self, name: str, description: str = "", labels: Dict[str, str] = None) -> str:
        """Create a histogram metric."""
        return self._create_metric(name, MetricType.HISTOGRAM, description, labels)
    
    def create_timing(self, name: str, description: str = "", labels: Dict[str, str] = None) -> str:
        """Create a timing metric."""
        return self._create_metric(name, MetricType.TIMING, description, labels)
    
    def _create_metric(self, name: str, metric_type: MetricType, description: str, 
                      labels: Dict[str, str] = None) -> str:
        """Create a new metric series."""
        with self.lock:
            full_name = self._build_metric_name(name, labels or {})
            
            if full_name in self.metrics:
                logger.warning(f"Metric {full_name} already exists")
                return full_name
            
            if len(self.metrics) >= self.max_series:
                logger.warning(f"Maximum metrics limit reached: {self.max_series}")
                return full_name
            
            series = MetricSeries(
                name=name,
                metric_type=metric_type,
                description=description,
                labels=labels or {}
            )
            
            self.metrics[full_name] = series
            logger.debug(f"Created metric: {full_name} ({metric_type.value})")
            
            return full_name
    
    def _build_metric_name(self, name: str, labels: Dict[str, str]) -> str:
        """Build full metric name with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def increment(self, name: str, value: Union[int, float] = 1, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        full_name = self._build_metric_name(name, labels or {})
        
        with self.lock:
            if full_name not in self.metrics:
                self.create_counter(name, labels=labels)
            
            series = self.metrics[full_name]
            if series.metric_type != MetricType.COUNTER:
                logger.error(f"Metric {name} is not a counter")
                return
            
            # Get current value and add increment
            current_value = 0
            if series.points:
                current_value = series.points[-1].value
            
            new_value = current_value + value
            series.add_point(new_value, labels)
    
    def set_gauge(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        full_name = self._build_metric_name(name, labels or {})
        
        with self.lock:
            if full_name not in self.metrics:
                self.create_gauge(name, labels=labels)
            
            series = self.metrics[full_name]
            if series.metric_type != MetricType.GAUGE:
                logger.error(f"Metric {name} is not a gauge")
                return
            
            series.add_point(value, labels)
    
    def record_timing(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timing measurement."""
        full_name = self._build_metric_name(name, labels or {})
        
        with self.lock:
            if full_name not in self.metrics:
                self.create_timing(name, labels=labels)
            
            series = self.metrics[full_name]
            if series.metric_type != MetricType.TIMING:
                logger.error(f"Metric {name} is not a timing metric")
                return
            
            series.add_point(duration, labels)
    
    def record_histogram(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Record a histogram value."""
        full_name = self._build_metric_name(name, labels or {})
        
        with self.lock:
            if full_name not in self.metrics:
                self.create_histogram(name, labels=labels)
            
            series = self.metrics[full_name]
            if series.metric_type != MetricType.HISTOGRAM:
                logger.error(f"Metric {name} is not a histogram")
                return
            
            series.add_point(value, labels)
    
    def time_function(self, name: str, labels: Dict[str, str] = None):
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_timing(name, duration, labels)
            return wrapper
        return decorator
    
    def get_metric_stats(self, name: str, labels: Dict[str, str] = None, 
                        time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Get statistics for a metric over a time range."""
        full_name = self._build_metric_name(name, labels or {})
        
        with self.lock:
            if full_name not in self.metrics:
                return {}
            
            series = self.metrics[full_name]
            cutoff_time = time.time() - time_range_seconds
            
            # Filter points within time range
            recent_points = [p for p in series.points if p.timestamp >= cutoff_time]
            
            if not recent_points:
                return {'count': 0}
            
            values = [p.value for p in recent_points]
            
            stats = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'first_timestamp': recent_points[0].timestamp,
                'last_timestamp': recent_points[-1].timestamp
            }
            
            if len(values) > 1:
                stats['std_dev'] = statistics.stdev(values)
                
                # Calculate percentiles for histograms and timings
                if series.metric_type in [MetricType.HISTOGRAM, MetricType.TIMING]:
                    sorted_values = sorted(values)
                    stats['p50'] = self._percentile(sorted_values, 50)
                    stats['p90'] = self._percentile(sorted_values, 90)
                    stats['p95'] = self._percentile(sorted_values, 95)
                    stats['p99'] = self._percentile(sorted_values, 99)
                
                # Calculate rate for counters
                if series.metric_type == MetricType.COUNTER and len(recent_points) > 1:
                    time_span = recent_points[-1].timestamp - recent_points[0].timestamp
                    value_diff = recent_points[-1].value - recent_points[0].value
                    stats['rate'] = value_diff / time_span if time_span > 0 else 0
            
            return stats
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile of sorted values."""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def add_collector(self, collector_func: Callable):
        """Add a custom metrics collector function."""
        self.collectors.append(collector_func)
        logger.info(f"Added custom collector: {collector_func.__name__}")
    
    def start_collection(self, interval_seconds: int = 60):
        """Start automatic metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, 
            args=(interval_seconds,), 
            daemon=True
        )
        self.collection_thread.start()
        logger.info(f"Started metrics collection (interval={interval_seconds}s)")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self, interval_seconds: int):
        """Main collection loop."""
        while self.is_collecting:
            start_time = time.time()
            
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Run custom collectors
                for collector in self.collectors:
                    try:
                        collector(self)
                    except Exception as e:
                        logger.error(f"Custom collector failed: {e}")
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Track collection performance
                collection_time = time.time() - start_time
                self.performance_stats['collection_time'].append(collection_time)
                self.performance_stats['metrics_count'].append(len(self.metrics))
                self.performance_stats['memory_usage'].append(self._get_memory_usage())
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            
            # Wait for next collection
            time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.set_gauge('system_cpu_percent', cpu_percent)
            
            # Memory metrics  
            memory = psutil.virtual_memory()
            self.set_gauge('system_memory_percent', memory.percent)
            self.set_gauge('system_memory_used', memory.used)
            self.set_gauge('system_memory_available', memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.set_gauge('system_disk_percent', disk_percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self.set_gauge('system_network_bytes_sent', net_io.bytes_sent)
                self.set_gauge('system_network_bytes_recv', net_io.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            self.set_gauge('process_memory_percent', process.memory_percent())
            self.set_gauge('process_cpu_percent', process.cpu_percent())
            self.set_gauge('process_num_threads', process.num_threads())
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _cleanup_old_data(self):
        """Remove old metric data points."""
        cutoff_time = time.time() - self.retention_seconds
        
        with self.lock:
            for series in self.metrics.values():
                # Filter out old points
                series.points = deque(
                    (p for p in series.points if p.timestamp >= cutoff_time),
                    maxlen=series.points.maxlen
                )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self.lock:
            for full_name, series in self.metrics.items():
                # Add help and type comments
                lines.append(f"# HELP {series.name} {series.description}")
                lines.append(f"# TYPE {series.name} {self._prometheus_metric_type(series.metric_type)}")
                
                # Get latest value
                if series.points:
                    latest_point = series.points[-1]
                    
                    # Build label string
                    if latest_point.labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in latest_point.labels.items())
                        metric_line = f"{series.name}{{{label_str}}} {latest_point.value}"
                    else:
                        metric_line = f"{series.name} {latest_point.value}"
                    
                    lines.append(metric_line)
                
                lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def _prometheus_metric_type(self, metric_type: MetricType) -> str:
        """Convert metric type to Prometheus type."""
        mapping = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMING: "histogram",
            MetricType.RATE: "gauge"
        }
        return mapping.get(metric_type, "gauge")
    
    def _export_json_format(self) -> str:
        """Export metrics in JSON format."""
        export_data = {}
        
        with self.lock:
            for full_name, series in self.metrics.items():
                export_data[full_name] = {
                    'name': series.name,
                    'type': series.metric_type.value,
                    'description': series.description,
                    'labels': series.labels,
                    'points': [
                        {
                            'timestamp': p.timestamp,
                            'value': p.value,
                            'labels': p.labels,
                            'metadata': p.metadata
                        }
                        for p in list(series.points)
                    ]
                }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection performance statistics."""
        stats = {}
        
        for metric_name, values in self.performance_stats.items():
            if values:
                stats[metric_name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        stats['total_metrics'] = len(self.metrics)
        stats['is_collecting'] = self.is_collecting
        
        return stats


class PerformanceMonitor:
    """High-level performance monitoring with alerting."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.performance_thresholds = {
            'response_time': {'warning': 1.0, 'critical': 5.0},
            'error_rate': {'warning': 0.05, 'critical': 0.20},
            'throughput': {'warning': 100, 'critical': 50},
            'memory_usage': {'warning': 0.80, 'critical': 0.95},
            'cpu_usage': {'warning': 80, 'critical': 95}
        }
        self.alerts = []
        
    def track_request(self, endpoint: str, duration: float, status_code: int):
        """Track HTTP request metrics."""
        # Record timing
        self.metrics.record_timing('http_request_duration', duration, 
                                 {'endpoint': endpoint, 'status': str(status_code)})
        
        # Increment request counter
        self.metrics.increment('http_requests_total', 1, 
                             {'endpoint': endpoint, 'status': str(status_code)})
        
        # Track error rate
        if status_code >= 400:
            self.metrics.increment('http_errors_total', 1, {'endpoint': endpoint})
    
    def track_ocr_performance(self, engine: str, confidence: float, processing_time: float):
        """Track OCR engine performance."""
        self.metrics.record_timing('ocr_processing_time', processing_time, {'engine': engine})
        self.metrics.set_gauge('ocr_confidence', confidence, {'engine': engine})
        self.metrics.increment('ocr_requests_total', 1, {'engine': engine})
    
    def track_model_inference(self, model_name: str, inference_time: float, 
                            batch_size: int, success: bool):
        """Track model inference performance."""
        self.metrics.record_timing('model_inference_time', inference_time, 
                                 {'model': model_name, 'batch_size': str(batch_size)})
        
        self.metrics.increment('model_inferences_total', 1, 
                             {'model': model_name, 'success': str(success)})
        
        if not success:
            self.metrics.increment('model_inference_errors', 1, {'model': model_name})
    
    def track_dataset_processing(self, stage: str, items_processed: int, 
                               processing_time: float):
        """Track dataset processing metrics."""
        self.metrics.record_timing('dataset_processing_time', processing_time, {'stage': stage})
        self.metrics.set_gauge('dataset_items_processed', items_processed, {'stage': stage})
        self.metrics.increment('dataset_operations_total', 1, {'stage': stage})
    
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts based on recent metrics."""
        alerts = []
        
        # Check response time
        response_stats = self.metrics.get_metric_stats('http_request_duration', time_range_seconds=300)
        if response_stats and response_stats.get('p95', 0) > self.performance_thresholds['response_time']['critical']:
            alerts.append({
                'type': 'response_time',
                'severity': 'critical',
                'message': f"P95 response time: {response_stats['p95']:.2f}s",
                'value': response_stats['p95']
            })
        elif response_stats and response_stats.get('p95', 0) > self.performance_thresholds['response_time']['warning']:
            alerts.append({
                'type': 'response_time',
                'severity': 'warning', 
                'message': f"P95 response time: {response_stats['p95']:.2f}s",
                'value': response_stats['p95']
            })
        
        # Check error rate
        error_stats = self.metrics.get_metric_stats('http_errors_total', time_range_seconds=300)
        request_stats = self.metrics.get_metric_stats('http_requests_total', time_range_seconds=300)
        
        if error_stats and request_stats:
            error_rate = error_stats.get('rate', 0) / max(request_stats.get('rate', 1), 1)
            if error_rate > self.performance_thresholds['error_rate']['critical']:
                alerts.append({
                    'type': 'error_rate',
                    'severity': 'critical',
                    'message': f"Error rate: {error_rate:.1%}",
                    'value': error_rate
                })
            elif error_rate > self.performance_thresholds['error_rate']['warning']:
                alerts.append({
                    'type': 'error_rate', 
                    'severity': 'warning',
                    'message': f"Error rate: {error_rate:.1%}",
                    'value': error_rate
                })
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': time.time(),
            'metrics': {},
            'alerts': self.check_performance_alerts()
        }
        
        # Key performance metrics
        key_metrics = [
            'http_request_duration',
            'http_requests_total', 
            'http_errors_total',
            'ocr_processing_time',
            'model_inference_time',
            'system_cpu_percent',
            'system_memory_percent'
        ]
        
        for metric in key_metrics:
            stats = self.metrics.get_metric_stats(metric, time_range_seconds=3600)
            if stats:
                summary['metrics'][metric] = stats
        
        return summary