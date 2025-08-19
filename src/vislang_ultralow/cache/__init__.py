"""Caching layer for VisLang-UltraLow-Resource."""

from .cache_manager import CacheManager, get_cache_manager
from .decorators import cached, cache_key, invalidate_cache

# Enhanced Generation 3 caching
try:
    from .performance import PerformanceOptimizedCache, CacheMetrics
    _perf_cache_available = True
except ImportError:
    _perf_cache_available = False

try:
    from .intelligent_cache import (
        IntelligentCacheManager,
        AccessPatternPredictor,
        CacheValueEstimator,
        CompressionManager,
        CacheStatistics
    )
    _intelligent_cache_available = True
except ImportError:
    _intelligent_cache_available = False

__all__ = [
    "CacheManager",
    "get_cache_manager",
    "cached", 
    "cache_key",
    "invalidate_cache",
]

if _perf_cache_available:
    __all__.extend(["PerformanceOptimizedCache", "CacheMetrics"])

if _intelligent_cache_available:
    __all__.extend([
        "IntelligentCacheManager",
        "AccessPatternPredictor", 
        "CacheValueEstimator",
        "CompressionManager",
        "CacheStatistics"
    ])