"""Advanced performance optimization system with adaptive algorithms."""

import time
import threading
import logging
import asyncio
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import gc
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools
import weakref

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    MEMORY_OPTIMIZED = "memory_optimized"
    CPU_OPTIMIZED = "cpu_optimized"
    IO_OPTIMIZED = "io_optimized"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    operation_name: str
    execution_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    success_count: int = 0
    error_count: int = 0
    last_optimized: float = 0
    optimization_applied: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_execution_time(self) -> float:
        """Average execution time."""
        return statistics.mean(self.execution_times) if self.execution_times else 0.0
    
    @property
    def p95_execution_time(self) -> float:
        """95th percentile execution time."""
        if not self.execution_times:
            return 0.0
        sorted_times = sorted(self.execution_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]
    
    @property
    def success_rate(self) -> float:
        """Success rate."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0


class AdaptiveBatchProcessor:
    """Adaptive batch processor that optimizes batch sizes dynamically."""
    
    def __init__(self, initial_batch_size: int = 8, max_batch_size: int = 128):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = deque(maxlen=50)
        self.adaptation_threshold = 0.1  # 10% improvement needed
        
    def process_batch(self, items: List[Any], process_func: Callable) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        results = []
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(items), self.current_batch_size):
            batch = items[i:i + self.current_batch_size]
            batch_start = time.time()
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                self.performance_history.append({
                    'batch_size': len(batch),
                    'processing_time': batch_time,
                    'throughput': len(batch) / batch_time,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Fall back to individual processing
                for item in batch:
                    try:
                        result = process_func([item])
                        results.extend(result)
                    except Exception as item_e:
                        logger.error(f"Individual item processing error: {item_e}")
        
        total_time = time.time() - start_time
        
        # Adapt batch size based on performance
        self._adapt_batch_size(total_time, len(items))
        
        return results
    
    def _adapt_batch_size(self, total_time: float, total_items: int):
        """Adapt batch size based on performance metrics."""
        if len(self.performance_history) < 10:  # Need enough data
            return
        
        recent_performance = list(self.performance_history)[-10:]
        avg_throughput = statistics.mean(p['throughput'] for p in recent_performance)
        
        # Analyze different batch sizes
        batch_size_performance = defaultdict(list)
        for perf in recent_performance:
            batch_size_performance[perf['batch_size']].append(perf['throughput'])
        
        # Find optimal batch size
        best_batch_size = self.current_batch_size
        best_throughput = avg_throughput
        
        for batch_size, throughputs in batch_size_performance.items():
            if len(throughputs) >= 3:  # Need enough samples
                avg_tp = statistics.mean(throughputs)
                if avg_tp > best_throughput * (1 + self.adaptation_threshold):
                    best_batch_size = batch_size
                    best_throughput = avg_tp
        
        # Adjust batch size
        if best_batch_size != self.current_batch_size:
            old_size = self.current_batch_size
            self.current_batch_size = min(best_batch_size, self.max_batch_size)
            logger.info(f"Adapted batch size: {old_size} -> {self.current_batch_size}")
        
        # Experiment with new batch sizes occasionally
        if len(recent_performance) % 20 == 0:  # Every 20 batches
            if self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
                logger.debug(f"Experimenting with larger batch size: {self.current_batch_size}")


class MemoryPool:
    """Memory pool for efficient object reuse."""
    
    def __init__(self, object_factory: Callable, max_size: int = 100):
        self.object_factory = object_factory
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()
        
    def acquire(self) -> Any:
        """Acquire object from pool."""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
            else:
                obj = self.object_factory()
            
            self.in_use.add(id(obj))
            return obj
    
    def release(self, obj: Any):
        """Release object back to pool."""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.in_use:
                self.in_use.remove(obj_id)
                if len(self.pool) < self.max_size:
                    # Reset object state if needed
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    self.pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        with self.lock:
            self.pool.clear()
            self.in_use.clear()


class PerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.profiles = {}  # operation_name -> PerformanceProfile
        self.memory_pools = {}  # pool_name -> MemoryPool
        self.batch_processors = {}  # operation_name -> AdaptiveBatchProcessor
        self.optimization_rules = []
        self.is_monitoring = False
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Resource limits
        self.resource_limits = self._get_default_resource_limits()
        
        logger.info(f"Performance optimizer initialized with strategy: {strategy.value}")
    
    def _get_default_resource_limits(self) -> Dict[str, Any]:
        """Get default resource limits based on system."""
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        limits = {
            'max_threads': min(cpu_count * 2, 32),
            'max_processes': cpu_count,
            'max_memory_mb': int(memory_gb * 0.8 * 1024),  # 80% of available memory
            'batch_size_limit': 128,
            'concurrent_operations': cpu_count * 4
        }
        
        # Adjust based on strategy
        if self.strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            limits['max_threads'] = cpu_count
            limits['max_memory_mb'] = int(memory_gb * 0.6 * 1024)
        elif self.strategy == OptimizationStrategy.CPU_OPTIMIZED:
            limits['max_threads'] = cpu_count * 4
            limits['concurrent_operations'] = cpu_count * 8
        elif self.strategy == OptimizationStrategy.AGGRESSIVE:
            limits['max_threads'] = min(cpu_count * 4, 64)
            limits['max_processes'] = min(cpu_count * 2, 16)
            limits['concurrent_operations'] = cpu_count * 8
        
        return limits
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operation performance."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_profiling(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def execute_with_profiling(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with performance profiling."""
        # Initialize profile if not exists
        if operation_name not in self.profiles:
            self.profiles[operation_name] = PerformanceProfile(operation_name)
        
        profile = self.profiles[operation_name]
        
        # Monitor resources before execution
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu_time = process.cpu_times().user
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            profile.success_count += 1
            return result
            
        except Exception as e:
            profile.error_count += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
            
        finally:
            # Record performance metrics
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss
            final_cpu_time = process.cpu_times().user
            
            memory_delta = final_memory - initial_memory
            cpu_delta = final_cpu_time - initial_cpu_time
            
            profile.execution_times.append(execution_time)
            profile.memory_usage.append(memory_delta)
            profile.cpu_usage.append(cpu_delta)
            
            # Check if optimization is needed
            self._check_optimization_needed(operation_name, profile)
    
    def _check_optimization_needed(self, operation_name: str, profile: PerformanceProfile):
        """Check if operation needs optimization."""
        # Optimization criteria
        needs_optimization = (
            profile.p95_execution_time > 5.0 or  # Slow operations
            profile.avg_execution_time > 1.0 or   # Average too high
            profile.success_rate < 0.95 or        # High error rate
            len(profile.execution_times) % 100 == 0  # Periodic optimization
        )
        
        if needs_optimization and time.time() - profile.last_optimized > 300:  # 5 minutes cooldown
            self._apply_optimization(operation_name, profile)
    
    def _apply_optimization(self, operation_name: str, profile: PerformanceProfile):
        """Apply optimization strategies to operation."""
        logger.info(f"Applying optimization to operation: {operation_name}")
        
        optimizations = []
        
        # Memory optimization
        if statistics.mean(profile.memory_usage) > 100 * 1024 * 1024:  # > 100MB
            optimizations.append("memory_pooling")
            self._create_memory_pool(operation_name)
        
        # Batch processing optimization
        if len(profile.execution_times) > 10:
            avg_time = statistics.mean(list(profile.execution_times)[-10:])
            if avg_time > 0.5:  # Slow operations benefit from batching
                optimizations.append("batch_processing")
                self._create_batch_processor(operation_name)
        
        # Caching optimization
        if profile.success_rate > 0.98:  # Stable operations
            optimizations.append("result_caching")
        
        # Parallel processing optimization
        if profile.avg_execution_time > 2.0:
            optimizations.append("parallel_processing")
        
        profile.optimization_applied = {
            'timestamp': time.time(),
            'optimizations': optimizations,
            'avg_time_before': profile.avg_execution_time
        }
        profile.last_optimized = time.time()
        
        logger.info(f"Applied optimizations for {operation_name}: {optimizations}")
    
    def _create_memory_pool(self, operation_name: str):
        """Create memory pool for operation."""
        def factory():
            # Generic object factory - would be customized per operation
            return {}
        
        pool_name = f"{operation_name}_pool"
        self.memory_pools[pool_name] = MemoryPool(factory, max_size=50)
        logger.debug(f"Created memory pool: {pool_name}")
    
    def _create_batch_processor(self, operation_name: str):
        """Create adaptive batch processor for operation."""
        self.batch_processors[operation_name] = AdaptiveBatchProcessor(
            initial_batch_size=8,
            max_batch_size=self.resource_limits['batch_size_limit']
        )
        logger.debug(f"Created batch processor: {operation_name}")
    
    def optimize_parallel_execution(self, func: Callable, items: List[Any], 
                                  max_workers: Optional[int] = None) -> List[Any]:
        """Execute function in parallel with optimization."""
        if not items:
            return []
        
        # Determine optimal worker count
        if max_workers is None:
            max_workers = min(
                len(items),
                self.resource_limits['max_threads'],
                multiprocessing.cpu_count() * 2
            )
        
        # Choose execution method based on strategy and workload
        if len(items) < 10 or max_workers == 1:
            # Sequential execution for small workloads
            return [func(item) for item in items]
        
        # Determine if CPU-bound or I/O-bound
        is_cpu_bound = self._estimate_cpu_bound(func, items[:min(3, len(items))])
        
        if is_cpu_bound and len(items) > 50:
            # Use process pool for CPU-bound tasks
            return self._execute_with_processes(func, items, max_workers)
        else:
            # Use thread pool for I/O-bound tasks
            return self._execute_with_threads(func, items, max_workers)
    
    def _estimate_cpu_bound(self, func: Callable, sample_items: List[Any]) -> bool:
        """Estimate if function is CPU-bound by sampling."""
        try:
            process = psutil.Process()
            initial_cpu = process.cpu_times().user
            
            # Run sample
            for item in sample_items:
                func(item)
            
            cpu_delta = process.cpu_times().user - initial_cpu
            # If significant CPU time used relative to wall time, likely CPU-bound
            return cpu_delta > 0.01 * len(sample_items)  # Heuristic
            
        except Exception:
            return False  # Default to I/O-bound
    
    def _execute_with_threads(self, func: Callable, items: List[Any], max_workers: int) -> List[Any]:
        """Execute with thread pool."""
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Thread execution error at index {index}: {e}")
                    results[index] = None
        
        return [r for r in results if r is not None]
    
    def _execute_with_processes(self, func: Callable, items: List[Any], max_workers: int) -> List[Any]:
        """Execute with process pool."""
        max_processes = min(max_workers, self.resource_limits['max_processes'])
        results = []
        
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all tasks
            futures = [executor.submit(func, item) for item in items]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Process execution error: {e}")
        
        return results
    
    def get_memory_pool(self, pool_name: str) -> Optional[MemoryPool]:
        """Get memory pool by name."""
        return self.memory_pools.get(pool_name)
    
    def get_batch_processor(self, operation_name: str) -> Optional[AdaptiveBatchProcessor]:
        """Get batch processor for operation."""
        return self.batch_processors.get(operation_name)
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Performance monitoring loop."""
        while self.is_monitoring:
            try:
                # Monitor system resources
                self._monitor_system_resources()
                
                # Check for optimization opportunities
                self._check_global_optimizations()
                
                # Garbage collection if needed
                if self.strategy != OptimizationStrategy.CONSERVATIVE:
                    self._optimize_memory()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _monitor_system_resources(self):
        """Monitor system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.counters['cpu_high_usage'] += 1 if cpu_percent > 80 else 0
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                self.counters['memory_high_usage'] += 1
                logger.warning(f"High memory usage: {memory.percent}%")
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.timers['disk_read_time'].append(disk_io.read_time)
                self.timers['disk_write_time'].append(disk_io.write_time)
            
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
    
    def _check_global_optimizations(self):
        """Check for global optimization opportunities."""
        # Memory pool cleanup
        for pool_name, pool in self.memory_pools.items():
            if len(pool.pool) > pool.max_size // 2:
                # Pool is underutilized, reduce size
                pool.pool = pool.pool[:pool.max_size // 2]
                logger.debug(f"Reduced memory pool size: {pool_name}")
        
        # Batch processor optimization
        for op_name, processor in self.batch_processors.items():
            if len(processor.performance_history) > 30:
                # Analyze recent performance
                recent_perf = list(processor.performance_history)[-10:]
                avg_throughput = statistics.mean(p['throughput'] for p in recent_perf)
                
                if avg_throughput < 10:  # Low throughput
                    processor.current_batch_size = max(processor.current_batch_size // 2, 1)
                    logger.debug(f"Reduced batch size for {op_name}: {processor.current_batch_size}")
    
    def _optimize_memory(self):
        """Optimize memory usage."""
        # Force garbage collection periodically
        if self.counters.get('memory_high_usage', 0) > 5:
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")
            self.counters['memory_high_usage'] = 0
        
        # Clear old performance data
        for profile in self.profiles.values():
            if len(profile.execution_times) > 100:
                # Keep only recent data
                profile.execution_times = deque(
                    list(profile.execution_times)[-100:], 
                    maxlen=1000
                )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'strategy': self.strategy.value,
            'resource_limits': self.resource_limits,
            'profiles': {},
            'memory_pools': {},
            'batch_processors': {},
            'system_counters': dict(self.counters)
        }
        
        # Profile summaries
        for op_name, profile in self.profiles.items():
            report['profiles'][op_name] = {
                'avg_execution_time': profile.avg_execution_time,
                'p95_execution_time': profile.p95_execution_time,
                'success_rate': profile.success_rate,
                'total_executions': profile.success_count + profile.error_count,
                'last_optimized': profile.last_optimized,
                'optimizations_applied': profile.optimization_applied
            }
        
        # Memory pool stats
        for pool_name, pool in self.memory_pools.items():
            with pool.lock:
                report['memory_pools'][pool_name] = {
                    'pool_size': len(pool.pool),
                    'max_size': pool.max_size,
                    'in_use_count': len(pool.in_use)
                }
        
        # Batch processor stats
        for op_name, processor in self.batch_processors.items():
            recent_performance = list(processor.performance_history)[-10:] if processor.performance_history else []
            report['batch_processors'][op_name] = {
                'current_batch_size': processor.current_batch_size,
                'max_batch_size': processor.max_batch_size,
                'avg_throughput': statistics.mean(p['throughput'] for p in recent_performance) if recent_performance else 0
            }
        
        return report
    
    def clear_profiles(self):
        """Clear all performance profiles."""
        with self.lock:
            self.profiles.clear()
            self.counters.clear()
            self.timers.clear()
            logger.info("Performance profiles cleared")
    
    def export_profiles(self, output_file: str):
        """Export performance profiles to file."""
        import json
        
        report = self.get_performance_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance profiles exported to {output_file}")


class CacheOptimizer:
    """Cache optimization strategies."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'evictions': 0})
        
    def optimize_cache_size(self, cache_name: str, current_size: int, 
                          hit_rate: float, memory_usage: int) -> int:
        """Optimize cache size based on performance metrics."""
        
        # If hit rate is high and memory allows, increase cache size
        if hit_rate > 0.8 and memory_usage < self.max_memory_bytes * 0.7:
            return min(current_size * 2, current_size + 1000)
        
        # If hit rate is low or memory pressure, decrease cache size
        elif hit_rate < 0.5 or memory_usage > self.max_memory_bytes * 0.9:
            return max(current_size // 2, 100)
        
        # Otherwise keep current size
        return current_size
    
    def get_optimal_eviction_strategy(self, access_pattern: List[str]) -> str:
        """Determine optimal cache eviction strategy based on access patterns."""
        
        if not access_pattern:
            return "lru"
        
        # Analyze access pattern
        unique_keys = set(access_pattern)
        recency_importance = self._analyze_recency_importance(access_pattern)
        frequency_importance = self._analyze_frequency_importance(access_pattern)
        
        # Choose strategy based on analysis
        if recency_importance > frequency_importance:
            return "lru"  # Least Recently Used
        elif frequency_importance > recency_importance * 1.5:
            return "lfu"  # Least Frequently Used
        else:
            return "adaptive"  # Adaptive Replacement Cache
    
    def _analyze_recency_importance(self, access_pattern: List[str]) -> float:
        """Analyze importance of recency in access pattern."""
        if len(access_pattern) < 10:
            return 0.5
        
        # Calculate how often recent items are accessed again
        recent_reaccess = 0
        total_recent = 0
        window_size = min(20, len(access_pattern) // 4)
        
        for i in range(len(access_pattern) - window_size):
            recent_items = set(access_pattern[i:i + window_size])
            future_accesses = access_pattern[i + window_size:]
            
            for item in recent_items:
                total_recent += 1
                if item in future_accesses[:window_size]:
                    recent_reaccess += 1
        
        return recent_reaccess / total_recent if total_recent > 0 else 0.5
    
    def _analyze_frequency_importance(self, access_pattern: List[str]) -> float:
        """Analyze importance of frequency in access pattern."""
        from collections import Counter
        
        if len(access_pattern) < 10:
            return 0.5
        
        # Calculate correlation between frequency and future access
        item_counts = Counter(access_pattern[:len(access_pattern) // 2])
        future_accesses = Counter(access_pattern[len(access_pattern) // 2:])
        
        correlation = 0
        total_items = 0
        
        for item, past_count in item_counts.items():
            future_count = future_accesses.get(item, 0)
            correlation += min(past_count, future_count)
            total_items += max(past_count, future_count)
        
        return correlation / total_items if total_items > 0 else 0.5