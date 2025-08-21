"""Distributed processing for scalable vision-language model operations."""

import logging
import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue
import multiprocessing as mp
from pathlib import Path
import hashlib
import uuid
from datetime import datetime, timedelta
import pickle
import socket
import subprocess
from collections import defaultdict, deque

try:
    import redis
    _redis_available = True
except ImportError:
    _redis_available = False
    logging.warning("Redis not available, using local queue")

try:
    import ray
    _ray_available = True
except ImportError:
    _ray_available = False
    logging.warning("Ray not available, using local distributed processing")

try:
    import dask
    from dask.distributed import Client, as_completed as dask_as_completed
    _dask_available = True
except ImportError:
    _dask_available = False
    logging.warning("Dask not available, using alternative distributed processing")

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class DistributedTask:
    """Distributed task definition."""
    task_id: str
    function_name: str
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[int] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value > other.priority.value  # Higher priority first


@dataclass
class TaskResult:
    """Result of distributed task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time,
            'worker_id': self.worker_id,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count
        }


@dataclass
class WorkerNode:
    """Distributed worker node information."""
    worker_id: str
    hostname: str
    cpu_count: int
    memory_gb: float
    gpu_count: int = 0
    capabilities: List[str] = field(default_factory=list)
    load_factor: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_tasks: List[str] = field(default_factory=list)
    
    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """Check if worker is healthy."""
        return (datetime.now() - self.last_heartbeat).seconds < timeout_seconds


class DistributedTaskManager:
    """Manages distributed task execution across multiple workers."""
    
    def __init__(self, 
                 cluster_config: Optional[Dict[str, Any]] = None,
                 use_ray: bool = False,
                 use_dask: bool = False):
        self.cluster_config = cluster_config or {}
        self.use_ray = use_ray and _ray_available
        self.use_dask = use_dask and _dask_available
        
        # Task management
        self.task_queue = PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.task_registry = {}
        self.task_dependencies = defaultdict(set)
        
        # Worker management
        self.workers = {}
        self.worker_stats = defaultdict(lambda: {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_load': 0.0
        })
        
        # Execution tracking
        self.execution_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'throughput_tasks_per_second': 0.0
        }
        
        # Initialize distributed backend
        self._initialize_distributed_backend()
        
        # Start background processes
        self._start_background_processes()
        
        logger.info(f"Distributed task manager initialized", extra={
            'use_ray': self.use_ray,
            'use_dask': self.use_dask,
            'cluster_config': bool(self.cluster_config)
        })
    
    def _initialize_distributed_backend(self):
        """Initialize distributed computing backend."""
        if self.use_ray:
            try:
                if not ray.is_initialized():
                    ray.init(**self.cluster_config.get('ray', {}))
                self.ray_client = ray
                logger.info("Ray distributed backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                self.use_ray = False
        
        if self.use_dask:
            try:
                scheduler_address = self.cluster_config.get('dask', {}).get('scheduler_address')
                if scheduler_address:
                    self.dask_client = Client(scheduler_address)
                else:
                    self.dask_client = Client()  # Local cluster
                logger.info("Dask distributed backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Dask: {e}")
                self.use_dask = False
        
        # Fallback to multiprocessing
        if not (self.use_ray or self.use_dask):
            max_workers = self.cluster_config.get('max_workers', mp.cpu_count())
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers * 2)
            logger.info(f"Using multiprocessing backend with {max_workers} workers")
    
    def _start_background_processes(self):
        """Start background processes for task management."""
        self.running = True
        
        # Task scheduler thread
        self.scheduler_thread = threading.Thread(target=self._task_scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        # Health monitor thread
        self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_monitor_thread.start()
        
        # Metrics collector thread
        self.metrics_thread = threading.Thread(target=self._metrics_collector_loop, daemon=True)
        self.metrics_thread.start()
    
    def register_worker(self, worker: WorkerNode) -> bool:
        """Register a new worker node."""
        try:
            self.workers[worker.worker_id] = worker
            logger.info(f"Registered worker {worker.worker_id} on {worker.hostname}")
            return True
        except Exception as e:
            logger.error(f"Failed to register worker {worker.worker_id}: {e}")
            return False
    
    def submit_task(self, 
                   function: Callable,
                   *args,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[int] = None,
                   max_retries: int = 3,
                   dependencies: List[str] = None,
                   **kwargs) -> str:
        """Submit a task for distributed execution."""
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            function_name=function.__name__ if hasattr(function, '__name__') else str(function),
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            dependencies=dependencies or []
        )
        
        # Register function for execution
        self.task_registry[function.__name__] = function
        
        # Add to queue
        self.task_queue.put(task)
        
        # Track dependencies
        if dependencies:
            self.task_dependencies[task_id] = set(dependencies)
        
        self.execution_metrics['total_tasks'] += 1
        
        logger.debug(f"Submitted task {task_id} with priority {priority.name}")
        return task_id
    
    def submit_batch(self, 
                    tasks: List[Tuple[Callable, Tuple, Dict[str, Any]]],
                    priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """Submit a batch of tasks for execution."""
        task_ids = []
        
        for task_spec in tasks:
            if len(task_spec) == 3:
                function, args, kwargs = task_spec
            elif len(task_spec) == 2:
                function, args = task_spec
                kwargs = {}
            else:
                function = task_spec[0]
                args = ()
                kwargs = {}
            
            task_id = self.submit_task(function, *args, priority=priority, **kwargs)
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(tasks)} tasks")
        return task_ids
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result of a completed task."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            if task_id in self.running_tasks:
                time.sleep(0.1)  # Task is still running
                continue
            
            # Task not found
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="Task not found"
            )
        
        # Timeout
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error="Timeout waiting for result"
        )
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple tasks to complete."""
        results = []
        start_time = time.time()
        
        for task_id in task_ids:
            remaining_time = None
            if timeout is not None:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    break
            
            result = self.get_task_result(task_id, remaining_time)
            results.append(result)
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        try:
            # Remove from queue if pending
            # Note: PriorityQueue doesn't support efficient removal, 
            # so we'll mark as cancelled in running_tasks
            
            if task_id in self.running_tasks:
                self.running_tasks[task_id].status = TaskStatus.CANCELLED
                return True
            
            # Add to completed as cancelled
            self.completed_tasks[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                completed_at=datetime.now()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def _task_scheduler_loop(self):
        """Main task scheduling loop."""
        while self.running:
            try:
                # Get next task from queue (blocks for 1 second)
                if not self.task_queue.empty():
                    task = self.task_queue.get(timeout=1)
                    
                    # Check dependencies
                    if self._check_task_dependencies(task):
                        self._execute_task(task)
                    else:
                        # Put back in queue
                        self.task_queue.put(task)
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                if self.running:  # Ignore errors during shutdown
                    logger.error(f"Task scheduler error: {e}")
                time.sleep(1)
    
    def _check_task_dependencies(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
            
            dep_result = self.completed_tasks[dep_task_id]
            if dep_result.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _execute_task(self, task: DistributedTask):
        """Execute a task using available backend."""
        self.running_tasks[task.task_id] = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            if self.use_ray:
                future = self._execute_task_ray(task)
            elif self.use_dask:
                future = self._execute_task_dask(task)
            else:
                future = self._execute_task_local(task)
            
            # Handle result asynchronously
            self._handle_task_completion(task, future, start_time)
            
        except Exception as e:
            self._handle_task_failure(task, str(e), start_time)
    
    def _execute_task_ray(self, task: DistributedTask):
        """Execute task using Ray."""
        @ray.remote
        def remote_function(*args, **kwargs):
            func = self.task_registry[task.function_name]
            return func(*args, **kwargs)
        
        return remote_function.remote(*task.args, **task.kwargs)
    
    def _execute_task_dask(self, task: DistributedTask):
        """Execute task using Dask."""
        func = self.task_registry[task.function_name]
        return self.dask_client.submit(func, *task.args, **task.kwargs)
    
    def _execute_task_local(self, task: DistributedTask):
        """Execute task using local multiprocessing."""
        func = self.task_registry[task.function_name]
        
        # Choose executor based on function characteristics
        if hasattr(func, '_cpu_intensive'):
            executor = self.process_pool
        else:
            executor = self.thread_pool
        
        return executor.submit(func, *task.args, **task.kwargs)
    
    def _handle_task_completion(self, task: DistributedTask, future, start_time: float):
        """Handle task completion in background thread."""
        def completion_handler():
            try:
                if self.use_ray:
                    result = ray.get(future)
                elif self.use_dask:
                    result = future.result()
                else:
                    result = future.result(timeout=task.timeout)
                
                execution_time = time.time() - start_time
                
                task_result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    execution_time=execution_time,
                    completed_at=datetime.now()
                )
                
                self.completed_tasks[task.task_id] = task_result
                del self.running_tasks[task.task_id]
                
                self.execution_metrics['completed_tasks'] += 1
                self._update_execution_metrics(execution_time)
                
                logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s")
                
            except Exception as e:
                self._handle_task_failure(task, str(e), start_time)
        
        # Run completion handler in background
        completion_thread = threading.Thread(target=completion_handler, daemon=True)
        completion_thread.start()
    
    def _handle_task_failure(self, task: DistributedTask, error: str, start_time: float):
        """Handle task failure and retry logic."""
        execution_time = time.time() - start_time
        
        current_result = self.running_tasks.get(task.task_id)
        retry_count = current_result.retry_count if current_result else 0
        
        if retry_count < task.max_retries:
            # Retry task
            retry_count += 1
            logger.warning(f"Task {task.task_id} failed, retrying ({retry_count}/{task.max_retries}): {error}")
            
            # Add delay before retry
            time.sleep(task.retry_delay * (2 ** (retry_count - 1)))  # Exponential backoff
            
            # Update task for retry
            self.running_tasks[task.task_id] = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.RETRYING,
                retry_count=retry_count
            )
            
            # Re-queue task
            self.task_queue.put(task)
        else:
            # Task failed permanently
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=error,
                execution_time=execution_time,
                completed_at=datetime.now(),
                retry_count=retry_count
            )
            
            self.completed_tasks[task.task_id] = task_result
            del self.running_tasks[task.task_id]
            
            self.execution_metrics['failed_tasks'] += 1
            
            logger.error(f"Task {task.task_id} failed permanently: {error}")
    
    def _health_monitor_loop(self):
        """Monitor worker health and system status."""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check worker health
                unhealthy_workers = []
                for worker_id, worker in self.workers.items():
                    if not worker.is_healthy():
                        unhealthy_workers.append(worker_id)
                        logger.warning(f"Worker {worker_id} appears unhealthy")
                
                # Remove unhealthy workers
                for worker_id in unhealthy_workers:
                    del self.workers[worker_id]
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                if self.running:
                    logger.error(f"Health monitor error: {e}")
                time.sleep(30)
    
    def _metrics_collector_loop(self):
        """Collect and update execution metrics."""
        last_update = time.time()
        last_completed_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                current_completed = self.execution_metrics['completed_tasks']
                
                # Calculate throughput
                time_delta = current_time - last_update
                task_delta = current_completed - last_completed_count
                
                if time_delta > 0:
                    self.execution_metrics['throughput_tasks_per_second'] = task_delta / time_delta
                
                last_update = current_time
                last_completed_count = current_completed
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                if self.running:
                    logger.error(f"Metrics collector error: {e}")
                time.sleep(10)
    
    def _update_execution_metrics(self, execution_time: float):
        """Update execution time metrics."""
        total_time = self.execution_metrics.get('total_execution_time', 0.0)
        completed_count = self.execution_metrics['completed_tasks']
        
        total_time += execution_time
        self.execution_metrics['total_execution_time'] = total_time
        self.execution_metrics['average_execution_time'] = total_time / completed_count
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        return {
            'workers': {
                'total': len(self.workers),
                'healthy': sum(1 for w in self.workers.values() if w.is_healthy()),
                'details': {wid: {
                    'hostname': w.hostname,
                    'cpu_count': w.cpu_count,
                    'memory_gb': w.memory_gb,
                    'load_factor': w.load_factor,
                    'active_tasks': len(w.active_tasks)
                } for wid, w in self.workers.items()}
            },
            'tasks': {
                'pending': self.task_queue.qsize(),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'total': self.execution_metrics['total_tasks']
            },
            'performance': self.execution_metrics,
            'backend': {
                'ray_enabled': self.use_ray,
                'dask_enabled': self.use_dask,
                'local_pools': not (self.use_ray or self.use_dask)
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the distributed task manager."""
        logger.info("Shutting down distributed task manager")
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'scheduler_thread'):
            self.scheduler_thread.join(timeout=5)
        if hasattr(self, 'health_monitor_thread'):
            self.health_monitor_thread.join(timeout=5)
        if hasattr(self, 'metrics_thread'):
            self.metrics_thread.join(timeout=5)
        
        # Shutdown distributed backends
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
        
        if self.use_dask and hasattr(self, 'dask_client'):
            self.dask_client.close()
        
        # Shutdown local executors
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        logger.info("Distributed task manager shutdown complete")


class ClusterResourceManager:
    """Manages resource allocation across cluster nodes."""
    
    def __init__(self):
        self.nodes = {}
        self.resource_allocations = defaultdict(dict)
        self.allocation_history = deque(maxlen=1000)
        
    def register_node(self, node_id: str, resources: Dict[str, float]):
        """Register a compute node with available resources."""
        self.nodes[node_id] = {
            'total_resources': resources.copy(),
            'available_resources': resources.copy(),
            'allocated_resources': defaultdict(float),
            'utilization': defaultdict(float),
            'last_update': datetime.now()
        }
        logger.info(f"Registered node {node_id} with resources: {resources}")
    
    def allocate_resources(self, 
                          task_id: str, 
                          resource_requirements: Dict[str, float]) -> Optional[str]:
        """Allocate resources for a task, returning the selected node."""
        best_node = None
        best_score = float('-inf')
        
        for node_id, node_info in self.nodes.items():
            # Check if node can satisfy requirements
            can_satisfy = all(
                node_info['available_resources'].get(resource, 0) >= amount
                for resource, amount in resource_requirements.items()
            )
            
            if can_satisfy:
                # Calculate allocation score (prefer less loaded nodes)
                score = self._calculate_allocation_score(node_info, resource_requirements)
                if score > best_score:
                    best_score = score
                    best_node = node_id
        
        if best_node:
            # Allocate resources
            node_info = self.nodes[best_node]
            for resource, amount in resource_requirements.items():
                node_info['available_resources'][resource] -= amount
                node_info['allocated_resources'][resource] += amount
            
            self.resource_allocations[task_id] = {
                'node_id': best_node,
                'resources': resource_requirements.copy(),
                'allocated_at': datetime.now()
            }
            
            self.allocation_history.append({
                'task_id': task_id,
                'node_id': best_node,
                'resources': resource_requirements.copy(),
                'timestamp': datetime.now()
            })
            
            logger.debug(f"Allocated resources for task {task_id} on node {best_node}")
            return best_node
        
        logger.warning(f"Could not allocate resources for task {task_id}: {resource_requirements}")
        return None
    
    def deallocate_resources(self, task_id: str):
        """Deallocate resources when task completes."""
        if task_id not in self.resource_allocations:
            return
        
        allocation = self.resource_allocations[task_id]
        node_id = allocation['node_id']
        resources = allocation['resources']
        
        if node_id in self.nodes:
            node_info = self.nodes[node_id]
            for resource, amount in resources.items():
                node_info['available_resources'][resource] += amount
                node_info['allocated_resources'][resource] -= amount
        
        del self.resource_allocations[task_id]
        logger.debug(f"Deallocated resources for task {task_id}")
    
    def _calculate_allocation_score(self, 
                                  node_info: Dict[str, Any], 
                                  requirements: Dict[str, float]) -> float:
        """Calculate score for resource allocation on a node."""
        # Score based on available capacity and current load
        score = 0.0
        
        for resource, amount in requirements.items():
            total = node_info['total_resources'].get(resource, 0)
            available = node_info['available_resources'].get(resource, 0)
            
            if total > 0:
                utilization = (total - available) / total
                # Prefer nodes with lower utilization
                score += (1.0 - utilization) * amount
        
        return score
    
    def get_cluster_utilization(self) -> Dict[str, Any]:
        """Get cluster-wide resource utilization."""
        total_resources = defaultdict(float)
        allocated_resources = defaultdict(float)
        
        for node_info in self.nodes.values():
            for resource, amount in node_info['total_resources'].items():
                total_resources[resource] += amount
            for resource, amount in node_info['allocated_resources'].items():
                allocated_resources[resource] += amount
        
        utilization = {}
        for resource in total_resources:
            if total_resources[resource] > 0:
                utilization[resource] = allocated_resources[resource] / total_resources[resource]
            else:
                utilization[resource] = 0.0
        
        return {
            'total_resources': dict(total_resources),
            'allocated_resources': dict(allocated_resources),
            'utilization': utilization,
            'node_count': len(self.nodes),
            'active_allocations': len(self.resource_allocations)
        }


class FaultTolerantProcessor:
    """Fault-tolerant processing with automatic recovery."""
    
    def __init__(self, 
                 max_failures: int = 3,
                 recovery_strategies: List[str] = None):
        self.max_failures = max_failures
        self.recovery_strategies = recovery_strategies or ['retry', 'redistribute', 'degrade']
        self.failure_history = defaultdict(list)
        self.recovery_actions = defaultdict(int)
        
    def execute_with_fault_tolerance(self, 
                                   task: DistributedTask,
                                   executor: Callable) -> TaskResult:
        """Execute task with fault tolerance."""
        attempts = 0
        last_error = None
        
        while attempts < self.max_failures:
            try:
                result = executor(task)
                
                # Reset failure count on success
                if task.task_id in self.failure_history:
                    del self.failure_history[task.task_id]
                
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    completed_at=datetime.now()
                )
                
            except Exception as e:
                attempts += 1
                last_error = str(e)
                
                # Record failure
                self.failure_history[task.task_id].append({
                    'error': last_error,
                    'timestamp': datetime.now(),
                    'attempt': attempts
                })
                
                logger.warning(f"Task {task.task_id} failed (attempt {attempts}): {last_error}")
                
                # Apply recovery strategy
                if attempts < self.max_failures:
                    self._apply_recovery_strategy(task, last_error, attempts)
        
        # All attempts failed
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Failed after {self.max_failures} attempts: {last_error}",
            completed_at=datetime.now()
        )
    
    def _apply_recovery_strategy(self, 
                               task: DistributedTask, 
                               error: str, 
                               attempt: int):
        """Apply recovery strategy based on failure type."""
        strategy = self._select_recovery_strategy(error, attempt)
        
        if strategy == 'retry':
            # Simple retry with backoff
            delay = min(60, 2 ** attempt)  # Exponential backoff, max 60s
            time.sleep(delay)
            
        elif strategy == 'redistribute':
            # Try to redistribute to different worker
            # This would be implemented with worker selection logic
            pass
            
        elif strategy == 'degrade':
            # Reduce resource requirements or quality
            if 'resource_requirements' in task.metadata:
                # Reduce resource requirements by 50%
                for resource in task.resource_requirements:
                    task.resource_requirements[resource] *= 0.5
        
        self.recovery_actions[strategy] += 1
    
    def _select_recovery_strategy(self, error: str, attempt: int) -> str:
        """Select appropriate recovery strategy based on error and attempt."""
        if 'memory' in error.lower() or 'resource' in error.lower():
            return 'degrade'
        elif 'network' in error.lower() or 'connection' in error.lower():
            return 'redistribute'
        else:
            return 'retry'
    
    def get_fault_tolerance_metrics(self) -> Dict[str, Any]:
        """Get fault tolerance metrics."""
        total_failures = sum(len(failures) for failures in self.failure_history.values())
        
        return {
            'total_failures': total_failures,
            'tasks_with_failures': len(self.failure_history),
            'recovery_actions': dict(self.recovery_actions),
            'failure_rate': total_failures / max(1, len(self.failure_history)),
            'most_common_errors': self._get_most_common_errors()
        }
    
    def _get_most_common_errors(self) -> Dict[str, int]:
        """Get most common error types."""
        error_counts = defaultdict(int)
        
        for failures in self.failure_history.values():
            for failure in failures:
                error_type = failure['error'].split(':')[0]  # Get error type
                error_counts[error_type] += 1
        
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10])