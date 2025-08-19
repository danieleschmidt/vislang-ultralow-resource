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