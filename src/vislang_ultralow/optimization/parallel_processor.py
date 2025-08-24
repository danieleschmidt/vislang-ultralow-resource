"""Advanced parallel processing and task scheduling system."""

import asyncio
import threading
import multiprocessing
import logging
import time
import queue
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Iterator
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from collections import deque, defaultdict
import psutil
import numpy as np
import functools
import pickle
import uuid

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """Compare tasks by priority for priority queue."""
        return self.priority.value > other.priority.value


@dataclass
class WorkerStats:
    """Worker performance statistics."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_task_time: float = 0.0
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))


class AdaptiveWorkerPool:
    """Adaptive worker pool that scales based on workload."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = None, 
                 worker_type: str = "thread", scale_factor: float = 1.5):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.worker_type = worker_type
        self.scale_factor = scale_factor
        
        self.current_workers = min_workers
        self.pending_tasks = 0
        self.active_tasks = 0
        self.last_scale_time = 0
        self.scale_cooldown = 30  # 30 seconds
        
        # Performance tracking
        self.throughput_history = deque(maxlen=60)  # Last 60 measurements
        self.utilization_history = deque(maxlen=60)
        
        # Initialize executor
        self.executor = self._create_executor()
        
    def _create_executor(self):
        """Create appropriate executor based on worker type."""
        if self.worker_type == "thread":
            return ThreadPoolExecutor(max_workers=self.current_workers)
        elif self.worker_type == "process":
            return ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            raise ValueError(f"Unknown worker type: {self.worker_type}")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Future:
        """Submit task to worker pool."""
        self.pending_tasks += 1
        
        # Check if scaling is needed
        self._check_scaling_needed()
        
        # Submit to executor
        future = self.executor.submit(func, *args, **kwargs)
        
        # Track task completion
        def task_completed(fut):
            self.pending_tasks = max(0, self.pending_tasks - 1)
            self.active_tasks = max(0, self.active_tasks - 1)
            
        future.add_done_callback(task_completed)
        
        return future
    
    def _check_scaling_needed(self):
        """Check if worker pool scaling is needed."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        # Calculate utilization
        total_tasks = self.pending_tasks + self.active_tasks
        utilization = total_tasks / self.current_workers if self.current_workers > 0 else 0
        
        # Scale up if high utilization and pending tasks
        if (utilization > 0.8 and self.pending_tasks > 5 and 
            self.current_workers < self.max_workers):
            self._scale_up()
        
        # Scale down if low utilization
        elif (utilization < 0.3 and self.pending_tasks == 0 and 
              self.current_workers > self.min_workers):
            self._scale_down()
    
    def _scale_up(self):
        """Scale up worker pool."""
        new_workers = min(
            int(self.current_workers * self.scale_factor),
            self.max_workers
        )
        
        if new_workers > self.current_workers:
            logger.info(f"Scaling up workers: {self.current_workers} -> {new_workers}")
            self._resize_pool(new_workers)
    
    def _scale_down(self):
        """Scale down worker pool."""
        new_workers = max(
            int(self.current_workers / self.scale_factor),
            self.min_workers
        )
        
        if new_workers < self.current_workers:
            logger.info(f"Scaling down workers: {self.current_workers} -> {new_workers}")
            self._resize_pool(new_workers)
    
    def _resize_pool(self, new_size: int):
        """Resize the worker pool."""
        old_executor = self.executor
        self.current_workers = new_size
        
        # Create new executor with new size
        self.executor = self._create_executor()
        self.last_scale_time = time.time()
        
        # Shutdown old executor gracefully
        old_executor.shutdown(wait=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'pending_tasks': self.pending_tasks,
            'active_tasks': self.active_tasks,
            'worker_type': self.worker_type,
            'scale_factor': self.scale_factor
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        if self.executor:
            self.executor.shutdown(wait=wait)


class TaskScheduler:
    """Advanced task scheduler with dependency management and priority queuing."""
    
    def __init__(self, max_concurrent_tasks: int = None):
        self.max_concurrent_tasks = max_concurrent_tasks or multiprocessing.cpu_count() * 2
        
        # Task storage
        self.tasks = {}  # task_id -> Task
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}  # task_id -> Future
        self.completed_tasks = {}  # task_id -> Task
        
        # Dependency graph
        self.dependency_graph = defaultdict(set)  # task_id -> set of dependent tasks
        self.dependency_count = defaultdict(int)   # task_id -> number of dependencies
        
        # Worker pools
        self.thread_pool = AdaptiveWorkerPool(
            min_workers=2, 
            max_workers=multiprocessing.cpu_count() * 2,
            worker_type="thread"
        )
        self.process_pool = AdaptiveWorkerPool(
            min_workers=1,
            max_workers=multiprocessing.cpu_count(),
            worker_type="process"
        )
        
        # Statistics
        self.stats = defaultdict(int)
        self.execution_times = deque(maxlen=1000)
        
        # Control
        self.is_running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        logger.info("Task scheduler initialized")
    
    def submit_task(self, func: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None, max_retries: int = 3,
                   dependencies: List[str] = None, callback: Optional[Callable] = None,
                   use_process: bool = False, **kwargs) -> str:
        """Submit a task to the scheduler."""
        
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            dependencies=dependencies or [],
            callback=callback
        )
        
        with self.lock:
            self.tasks[task_id] = task
            
            # Set up dependencies
            dependency_count = 0
            for dep_id in task.dependencies:
                if dep_id in self.tasks and self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    self.dependency_graph[dep_id].add(task_id)
                    dependency_count += 1
            
            self.dependency_count[task_id] = dependency_count
            
            # Add to queue if no dependencies
            if dependency_count == 0:
                # Mark pool preference
                task.kwargs['_use_process'] = use_process
                self.task_queue.put(task)
                logger.debug(f"Task {task_id} added to queue")
            else:
                logger.debug(f"Task {task_id} waiting for {dependency_count} dependencies")
        
        self.stats['tasks_submitted'] += 1
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task_id} cancelled")
                return True
            elif task_id in self.running_tasks:
                future = self.running_tasks[task_id]
                if future.cancel():
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Running task {task_id} cancelled")
                    return True
            
            return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        with self.lock:
            task = self.tasks.get(task_id)
            return task.status if task else None
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result, blocking if necessary."""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise task.error
            elif task.status == TaskStatus.CANCELLED:
                raise RuntimeError(f"Task {task_id} was cancelled")
        
        # Wait for completion
        start_time = time.time()
        while True:
            with self.lock:
                task = self.tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise task.error
                elif task.status == TaskStatus.CANCELLED:
                    raise RuntimeError(f"Task {task_id} was cancelled")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            
            time.sleep(0.1)
    
    def start(self):
        """Start the task scheduler."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")
    
    def stop(self, wait_for_completion: bool = True):
        """Stop the task scheduler."""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        # Shutdown worker pools
        self.thread_pool.shutdown(wait=wait_for_completion)
        self.process_pool.shutdown(wait=wait_for_completion)
        
        logger.info("Task scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check for completed tasks
                self._check_completed_tasks()
                
                # Schedule new tasks
                self._schedule_pending_tasks()
                
                # Clean up old tasks
                self._cleanup_completed_tasks()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1)
    
    def _check_completed_tasks(self):
        """Check for completed running tasks."""
        completed_task_ids = []
        
        with self.lock:
            for task_id, future in list(self.running_tasks.items()):
                if future.done():
                    completed_task_ids.append(task_id)
        
        # Process completed tasks
        for task_id in completed_task_ids:
            self._handle_task_completion(task_id)
    
    def _handle_task_completion(self, task_id: str):
        """Handle completion of a task."""
        with self.lock:
            task = self.tasks[task_id]
            future = self.running_tasks.pop(task_id)
            
            try:
                result = future.result()
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                
                # Calculate execution time
                if task.started_at:
                    execution_time = task.completed_at - task.started_at
                    self.execution_times.append(execution_time)
                
                # Execute callback if provided
                if task.callback:
                    try:
                        task.callback(result)
                    except Exception as e:
                        logger.error(f"Task callback failed: {e}")
                
                # Update dependent tasks
                self._update_dependent_tasks(task_id)
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                
                self.stats['tasks_completed'] += 1
                logger.debug(f"Task {task_id} completed successfully")
                
            except Exception as e:
                task.error = e
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                
                # Retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    
                    # Re-add to queue
                    self.task_queue.put(task)
                    logger.info(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                else:
                    self.stats['tasks_failed'] += 1
                    logger.error(f"Task {task_id} failed permanently: {e}")
    
    def _update_dependent_tasks(self, completed_task_id: str):
        """Update tasks that depend on the completed task."""
        dependent_tasks = self.dependency_graph.get(completed_task_id, set())
        
        for dep_task_id in dependent_tasks:
            self.dependency_count[dep_task_id] -= 1
            
            # If all dependencies are satisfied, add to queue
            if self.dependency_count[dep_task_id] == 0:
                task = self.tasks[dep_task_id]
                if task.status == TaskStatus.PENDING:
                    self.task_queue.put(task)
                    logger.debug(f"Task {dep_task_id} dependencies satisfied, added to queue")
        
        # Clean up dependency graph
        if completed_task_id in self.dependency_graph:
            del self.dependency_graph[completed_task_id]
    
    def _schedule_pending_tasks(self):
        """Schedule pending tasks to available workers."""
        current_running = len(self.running_tasks)
        
        while (current_running < self.max_concurrent_tasks and 
               not self.task_queue.empty()):
            
            try:
                task = self.task_queue.get_nowait()
                
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Choose worker pool
                use_process = task.kwargs.pop('_use_process', False)
                worker_pool = self.process_pool if use_process else self.thread_pool
                
                # Submit task
                future = worker_pool.submit_task(task.func, *task.args, **task.kwargs)
                
                # Update task status
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                # Track running task
                self.running_tasks[task.id] = future
                current_running += 1
                
                logger.debug(f"Task {task.id} started on {'process' if use_process else 'thread'} pool")
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error scheduling task: {e}")
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks to manage memory."""
        if len(self.completed_tasks) > 1000:  # Keep last 1000
            # Remove oldest completed tasks
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].completed_at or 0
            )
            
            to_remove = sorted_tasks[:-1000]  # Keep last 1000
            
            for task_id, _ in to_remove:
                del self.completed_tasks[task_id]
                if task_id in self.tasks:
                    del self.tasks[task_id]
            
            logger.debug(f"Cleaned up {len(to_remove)} old completed tasks")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.lock:
            stats = dict(self.stats)
            stats.update({
                'total_tasks': len(self.tasks),
                'pending_tasks': self.task_queue.qsize(),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'thread_pool_stats': self.thread_pool.get_stats(),
                'process_pool_stats': self.process_pool.get_stats()
            })
            
            # Execution time statistics
            if self.execution_times:
                import statistics as stat_module
                stats['avg_execution_time'] = stat_module.mean(self.execution_times)
                stats['p95_execution_time'] = np.percentile(self.execution_times, 95)
                stats['p99_execution_time'] = np.percentile(self.execution_times, 99)
        
        return stats
    
    def get_task_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get summary of all tasks."""
        with self.lock:
            summary = {
                'pending': [],
                'running': [],
                'completed': [],
                'failed': []
            }
            
            for task in self.tasks.values():
                task_info = {
                    'id': task.id,
                    'priority': task.priority.value,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'retry_count': task.retry_count
                }
                
                if task.status == TaskStatus.PENDING:
                    summary['pending'].append(task_info)
                elif task.status == TaskStatus.RUNNING:
                    summary['running'].append(task_info)
                elif task.status == TaskStatus.COMPLETED:
                    summary['completed'].append(task_info)
                elif task.status == TaskStatus.FAILED:
                    summary['failed'].append(task_info)
        
        return summary


class ParallelProcessor:
    """High-level parallel processing interface."""
    
    def __init__(self, scheduler: TaskScheduler = None, max_workers: int = None):
        self.scheduler = scheduler or TaskScheduler()
        self.batch_processors = {}
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        if not self.scheduler.is_running:
            self.scheduler.start()
    
    def process_batch(self, items: List[Any], func: Callable, batch_size: int = None) -> List[Any]:
        """Process items in batches for compatibility."""
        return self.batch_process(func, items, batch_size or 32)
    
    def map(self, func: Callable, items: List[Any], 
            use_processes: bool = False, chunk_size: int = None) -> List[Any]:
        """Parallel map function."""
        if not items:
            return []
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (multiprocessing.cpu_count() * 2))
        
        # Submit tasks
        task_ids = []
        results = [None] * len(items)
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_indices = list(range(i, min(i + chunk_size, len(items))))
            
            def process_chunk(items_chunk, indices):
                return [(idx, func(item)) for idx, item in zip(indices, items_chunk)]
            
            task_id = self.scheduler.submit_task(
                process_chunk, chunk, chunk_indices,
                use_process=use_processes
            )
            task_ids.append(task_id)
        
        # Collect results
        for task_id in task_ids:
            chunk_results = self.scheduler.get_task_result(task_id)
            for idx, result in chunk_results:
                results[idx] = result
        
        return results
    
    def batch_process(self, func: Callable, items: List[Any], 
                     batch_size: int = 32, use_processes: bool = False) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        # Submit batch tasks
        task_ids = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            task_id = self.scheduler.submit_task(
                func, batch,
                use_process=use_processes
            )
            task_ids.append(task_id)
        
        # Collect results
        all_results = []
        for task_id in task_ids:
            batch_results = self.scheduler.get_task_result(task_id)
            all_results.extend(batch_results)
        
        return all_results
    
    def pipeline(self, stages: List[Callable], items: List[Any]) -> List[Any]:
        """Process items through a pipeline of stages."""
        current_items = items
        
        for i, stage_func in enumerate(stages):
            # Submit pipeline stage
            task_ids = []
            
            for item in current_items:
                task_id = self.scheduler.submit_task(stage_func, item)
                task_ids.append(task_id)
            
            # Collect results for next stage
            current_items = []
            for task_id in task_ids:
                result = self.scheduler.get_task_result(task_id)
                current_items.append(result)
        
        return current_items
    
    def shutdown(self):
        """Shutdown parallel processor."""
        if self.scheduler:
            self.scheduler.stop()