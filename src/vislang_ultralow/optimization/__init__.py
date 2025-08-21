"""Performance optimization modules for VisLang UltraLow Resource."""

from .performance_optimizer import PerformanceOptimizer, OptimizationStrategy
from .parallel_processor import ParallelProcessor, TaskScheduler

# Enhanced Generation 3 optimization
try:
    from .quantum_optimizer import (
        QuantumInspiredOptimizer,
        AdaptiveResourceAllocator,
        ResourcePerformancePredictor
    )
    _quantum_optimizer_available = True
except ImportError:
    _quantum_optimizer_available = False

# Enhanced Generation 3+ optimization modules
try:
    from .distributed_processor import (
        DistributedTaskManager,
        ClusterResourceManager,
        FaultTolerantProcessor
    )
    _distributed_available = True
except ImportError:
    _distributed_available = False

try:
    from .adaptive_scaling import (
        AutoScaler,
        PredictiveScaler,
        ResourceUsageOptimizer,
        LoadBalancer
    )
    _adaptive_scaling_available = True
except ImportError:
    _adaptive_scaling_available = False

try:
    from .memory_optimizer import (
        MemoryManager,
        CacheOptimizer,
        GarbageCollectionTuner
    )
    _memory_optimizer_available = True
except ImportError:
    _memory_optimizer_available = False

__all__ = [
    "PerformanceOptimizer",
    "OptimizationStrategy",
    "ParallelProcessor", 
    "TaskScheduler"
]

if _quantum_optimizer_available:
    __all__.extend([
        "QuantumInspiredOptimizer",
        "AdaptiveResourceAllocator",
        "ResourcePerformancePredictor"
    ])

if _distributed_available:
    __all__.extend([
        "DistributedTaskManager",
        "ClusterResourceManager",
        "FaultTolerantProcessor"
    ])

if _adaptive_scaling_available:
    __all__.extend([
        "AutoScaler",
        "PredictiveScaler", 
        "ResourceUsageOptimizer",
        "LoadBalancer"
    ])

if _memory_optimizer_available:
    __all__.extend([
        "MemoryManager",
        "CacheOptimizer",
        "GarbageCollectionTuner"
    ])