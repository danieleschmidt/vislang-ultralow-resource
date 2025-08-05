"""Performance optimization and scaling utilities."""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional, Tuple
import logging
import time
import multiprocessing as mp
from functools import wraps, lru_cache
import threading
from queue import Queue, Empty
import gc
import psutil
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    processing_rate: float
    queue_size: int
    active_workers: int
    timestamp: float


class PerformanceMonitor:
    """Monitor system performance and auto-scale resources."""
    
    def __init__(self, check_interval: float = 5.0):
        """Initialize performance monitor.
        
        Args:
            check_interval: Seconds between performance checks
        """
        self.check_interval = check_interval
        self.metrics_history = []
        self.max_history = 100
        self.thresholds = {
            'cpu_high': 80.0,
            'cpu_low': 30.0,
            'memory_high': 85.0,
            'memory_low': 40.0,
            'queue_high': 1000,
            'queue_low': 10
        }
        self._stop_monitoring = threading.Event()
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.wait(self.check_interval):
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._check_thresholds(metrics)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_metrics = {
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        }
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_metrics = {
            'bytes_sent': net_io.bytes_sent if net_io else 0,
            'bytes_recv': net_io.bytes_recv if net_io else 0
        }
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=disk_metrics,
            network_io=net_metrics,
            processing_rate=0.0,  # Will be updated by processors
            queue_size=0,  # Will be updated by queue managers
            active_workers=0,  # Will be updated by worker pools
            timestamp=time.time()
        )
    
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger scaling."""
        # CPU threshold checks
        if metrics.cpu_usage > self.thresholds['cpu_high']:
            logger.warning(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            self._trigger_scale_up('cpu')
        elif metrics.cpu_usage < self.thresholds['cpu_low']:
            self._trigger_scale_down('cpu')
        
        # Memory threshold checks  
        if metrics.memory_usage > self.thresholds['memory_high']:
            logger.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
            self._trigger_memory_cleanup()
        
        # Queue size checks
        if metrics.queue_size > self.thresholds['queue_high']:
            logger.warning(f"High queue size: {metrics.queue_size}")
            self._trigger_scale_up('queue')
        elif metrics.queue_size < self.thresholds['queue_low']:
            self._trigger_scale_down('queue')
    
    def _trigger_scale_up(self, reason: str):
        """Trigger scale up event."""
        logger.info(f"Triggering scale up due to: {reason}")
        # This would typically interface with a worker pool manager
    
    def _trigger_scale_down(self, reason: str):
        """Trigger scale down event."""
        logger.debug(f"Triggering scale down due to: {reason}")
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup."""
        logger.info("Triggering memory cleanup")
        gc.collect()
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent performance metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, window: int = 10) -> Optional[PerformanceMetrics]:
        """Get average metrics over time window."""
        if not self.metrics_history:
            return None
        
        recent = self.metrics_history[-window:]
        
        return PerformanceMetrics(
            cpu_usage=np.mean([m.cpu_usage for m in recent]),
            memory_usage=np.mean([m.memory_usage for m in recent]),
            disk_io={'read_bytes': 0, 'write_bytes': 0},  # Simplified
            network_io={'bytes_sent': 0, 'bytes_recv': 0},  # Simplified
            processing_rate=np.mean([m.processing_rate for m in recent]),
            queue_size=int(np.mean([m.queue_size for m in recent])),
            active_workers=int(np.mean([m.active_workers for m in recent])),
            timestamp=time.time()
        )


class AdaptiveWorkerPool:
    """Dynamically scaling worker pool for concurrent processing."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = None,
        scale_factor: float = 1.5,
        idle_timeout: float = 60.0
    ):
        """Initialize adaptive worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (defaults to CPU count * 2)
            scale_factor: Factor to scale workers by
            idle_timeout: Seconds before idle workers are terminated
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or (mp.cpu_count() * 2)
        self.scale_factor = scale_factor
        self.idle_timeout = idle_timeout
        
        self.executor = None
        self.current_workers = 0
        self.task_queue = Queue()
        self.result_futures = []
        self.worker_last_used = {}
        
        self.performance_monitor = PerformanceMonitor()
        self.performance_monitor.start_monitoring()
        
        logger.info(f"Initialized adaptive worker pool: {min_workers}-{max_workers} workers")
    
    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.shutdown()
    
    def start(self):
        """Start the worker pool."""
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.min_workers,
                thread_name_prefix="AdaptiveWorker"
            )
            self.current_workers = self.min_workers
            logger.info(f"Started worker pool with {self.current_workers} workers")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        if self.executor:
            self.executor.shutdown(wait=wait)
            self.executor = None
            self.current_workers = 0
        
        self.performance_monitor.stop_monitoring()
        logger.info("Worker pool shutdown complete")
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to worker pool with adaptive scaling.
        
        Args:
            fn: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object for the task
        """
        if not self.executor:
            self.start()
        
        # Check if scaling is needed
        self._maybe_scale()
        
        # Submit task
        future = self.executor.submit(fn, *args, **kwargs)
        self.result_futures.append(weakref.ref(future))
        
        return future
    
    def map(self, fn: Callable, iterable, chunksize: int = 1) -> List[Any]:
        """Map function over iterable with adaptive scaling."""
        if not self.executor:
            self.start()
        
        # Check if scaling is needed for batch processing
        self._maybe_scale()
        
        # Use executor map
        results = list(self.executor.map(fn, iterable, chunksize=chunksize))
        return results
    
    def _maybe_scale(self):
        """Check if scaling is needed and adjust worker count."""
        metrics = self.performance_monitor.get_current_metrics()
        if not metrics:
            return
        
        # Calculate pending tasks
        pending_tasks = sum(1 for f_ref in self.result_futures 
                           if f_ref() and not f_ref().done())
        
        # Scale up conditions
        if (metrics.cpu_usage < 70 and  # CPU not overloaded
            pending_tasks > self.current_workers * 2 and  # Queue is building
            self.current_workers < self.max_workers):
            
            new_workers = min(
                int(self.current_workers * self.scale_factor),
                self.max_workers
            )
            self._scale_to(new_workers)
        
        # Scale down conditions
        elif (metrics.cpu_usage < 30 and  # Low CPU usage
              pending_tasks < self.current_workers // 2 and  # Low queue
              self.current_workers > self.min_workers):
            
            new_workers = max(
                int(self.current_workers / self.scale_factor),
                self.min_workers
            )
            self._scale_to(new_workers)
    
    def _scale_to(self, target_workers: int):
        """Scale worker pool to target number of workers."""
        if target_workers == self.current_workers:
            return
        
        logger.info(f"Scaling worker pool: {self.current_workers} -> {target_workers}")
        
        # Create new executor with target workers
        old_executor = self.executor
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=target_workers,
            thread_name_prefix="AdaptiveWorker"
        )
        self.current_workers = target_workers
        
        # Shutdown old executor (after brief delay to finish current tasks)
        if old_executor:
            threading.Timer(1.0, lambda: old_executor.shutdown(wait=False)).start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        active_futures = sum(1 for f_ref in self.result_futures 
                           if f_ref() and not f_ref().done())
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'active_tasks': active_futures,
            'total_submitted': len(self.result_futures)
        }


class BatchProcessor:
    """Efficient batch processing with memory management."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_memory_mb: int = 1024,
        prefetch_batches: int = 2
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            max_memory_mb: Maximum memory usage in MB
            prefetch_batches: Number of batches to prefetch
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.prefetch_batches = prefetch_batches
        
        self.processed_batches = 0
        self.total_items = 0
        self.start_time = None
        
    def process_batches(
        self, 
        items: List[Any], 
        process_fn: Callable[[List[Any]], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items in batches with memory management.
        
        Args:
            items: Items to process
            process_fn: Function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            List of processed results
        """
        self.start_time = time.time()
        self.total_items = len(items)
        
        results = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Memory check
            if self._check_memory():
                logger.warning("High memory usage, triggering garbage collection")
                gc.collect()
            
            # Process batch
            try:
                batch_results = process_fn(batch)
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
                
                self.processed_batches += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + len(batch), self.total_items)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size}: {e}")
                continue
        
        self._log_performance()
        return results
    
    def _check_memory(self) -> bool:
        """Check if memory usage is high."""
        memory_usage = psutil.virtual_memory().percent
        return memory_usage > 80
    
    def _log_performance(self):
        """Log processing performance."""
        if not self.start_time:
            return
        
        duration = time.time() - self.start_time
        items_per_second = self.total_items / duration if duration > 0 else 0
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  Items processed: {self.total_items}")
        logger.info(f"  Batches processed: {self.processed_batches}")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Items/second: {items_per_second:.1f}")


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for async retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        break
            
            raise last_exception
        return wrapper
    return decorator


@contextmanager
def memory_limit(max_memory_mb: int):
    """Context manager to monitor memory usage."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        if memory_used > max_memory_mb:
            logger.warning(f"Memory usage exceeded limit: {memory_used:.1f}MB > {max_memory_mb}MB")


class ResourcePool:
    """Generic resource pool with auto-scaling."""
    
    def __init__(self, factory: Callable, min_size: int = 2, max_size: int = 10):
        """Initialize resource pool.
        
        Args:
            factory: Function to create new resources
            min_size: Minimum pool size
            max_size: Maximum pool size
        """
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        
        self.pool = Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = threading.Lock()
        
        # Pre-populate with minimum resources
        for _ in range(min_size):
            resource = self.factory()
            self.pool.put(resource)
            self.created_count += 1
    
    @contextmanager
    def get_resource(self, timeout: float = 30.0):
        """Get resource from pool with timeout."""
        resource = None
        try:
            # Try to get existing resource
            try:
                resource = self.pool.get(timeout=timeout)
            except Empty:
                # Create new resource if under limit
                with self.lock:
                    if self.created_count < self.max_size:
                        resource = self.factory()
                        self.created_count += 1
                    else:
                        raise TimeoutError("Resource pool exhausted")
            
            yield resource
            
        finally:
            # Return resource to pool
            if resource and not self.pool.full():
                self.pool.put(resource)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            'available': self.pool.qsize(),
            'created': self.created_count,
            'max_size': self.max_size
        }