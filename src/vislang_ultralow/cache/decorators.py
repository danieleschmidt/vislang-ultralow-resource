"""Cache decorators for easy caching integration."""

import functools
import hashlib
import inspect
import logging
from typing import Any, Callable, Optional, Union, Dict, List
from .cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Convert all arguments to strings for hashing
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and relevant attributes
            key_parts.append(f"{arg.__class__.__name__}:{str(arg)}")
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments (sorted for consistency)
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{str(v)}")
    
    # Create hash of the combined key
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    exclude_args: Optional[List[str]] = None,
    condition: Optional[Callable] = None,
    invalidate_on: Optional[List[str]] = None
):
    """Decorator to cache function results.
    
    Args:
        ttl: Cache time-to-live in seconds
        key_prefix: Prefix for cache key
        exclude_args: Arguments to exclude from cache key
        condition: Function to determine if result should be cached
        invalidate_on: List of function names that should invalidate this cache
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        exclude_args = exclude_args or []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if caching is disabled or cache manager unavailable
            if not cache_manager or not cache_manager.redis:
                return func(*args, **kwargs)
            
            # Filter arguments for cache key generation
            filtered_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in exclude_args
            }
            
            # Get function signature to handle positional args
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **filtered_kwargs)
            bound_args.apply_defaults()
            
            # Generate cache key
            func_name = f"{func.__module__}.{func.__qualname__}"
            arg_key = cache_key(*bound_args.args, **bound_args.kwargs)
            cache_key_str = f"{key_prefix}{func_name}:{arg_key}"
            
            # Try to get from cache
            try:
                cached_result = cache_manager.get(cache_key_str)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func_name}")
                    return cached_result
            except Exception as e:
                logger.error(f"Cache get error for {func_name}: {e}")
            
            # Execute function
            logger.debug(f"Cache miss for {func_name}")
            result = func(*args, **kwargs)
            
            # Check condition before caching
            should_cache = True
            if condition:
                try:
                    should_cache = condition(result)
                except Exception as e:
                    logger.error(f"Cache condition error for {func_name}: {e}")
                    should_cache = False
            
            # Cache the result
            if should_cache and result is not None:
                try:
                    cache_manager.set(cache_key_str, result, ttl=ttl)
                    logger.debug(f"Cached result for {func_name}")
                except Exception as e:
                    logger.error(f"Cache set error for {func_name}: {e}")
            
            return result
        
        # Store metadata for cache invalidation
        wrapper._cache_prefix = key_prefix
        wrapper._cache_func_name = f"{func.__module__}.{func.__qualname__}"
        wrapper._invalidate_on = invalidate_on or []
        
        return wrapper
    
    return decorator


def invalidate_cache(
    func: Optional[Callable] = None,
    pattern: Optional[str] = None,
    keys: Optional[List[str]] = None
):
    """Decorator to invalidate cache entries when function is called.
    
    Args:
        func: Function to invalidate cache for
        pattern: Cache key pattern to invalidate
        keys: Specific cache keys to invalidate
    """
    def decorator(target_func: Callable) -> Callable:
        cache_manager = get_cache_manager()
        
        @functools.wraps(target_func)
        def wrapper(*args, **kwargs):
            # Execute the function first
            result = target_func(*args, **kwargs)
            
            # Invalidate cache entries
            if cache_manager and cache_manager.redis:
                try:
                    if func:
                        # Invalidate cache for specific function
                        func_pattern = f"*{func._cache_func_name}:*"
                        deleted = cache_manager.delete_pattern(func_pattern)
                        logger.debug(f"Invalidated {deleted} cache entries for {func._cache_func_name}")
                    
                    if pattern:
                        # Invalidate by pattern
                        deleted = cache_manager.delete_pattern(pattern)
                        logger.debug(f"Invalidated {deleted} cache entries matching pattern: {pattern}")
                    
                    if keys:
                        # Invalidate specific keys
                        for key in keys:
                            cache_manager.delete(key)
                        logger.debug(f"Invalidated {len(keys)} specific cache keys")
                        
                except Exception as e:
                    logger.error(f"Cache invalidation error: {e}")
            
            return result
        
        return wrapper
    
    return decorator


def cache_result_if(condition: Callable[[Any], bool]):
    """Cache result only if condition is met."""
    return lambda func: cached(condition=condition)(func)


def cache_with_timeout(seconds: int):
    """Cache with specific timeout."""
    return lambda func: cached(ttl=seconds)(func)


def cache_expensive_operation(ttl: int = 3600):
    """Decorator for expensive operations with longer TTL."""
    return cached(ttl=ttl, key_prefix="expensive:")


def cache_ocr_result(ttl: int = 86400):  # 24 hours
    """Decorator specifically for OCR results."""
    return cached(
        ttl=ttl,
        key_prefix="ocr:",
        condition=lambda result: result and result.get('text', '').strip()
    )


def cache_model_prediction(ttl: int = 3600):
    """Decorator for model predictions."""
    return cached(
        ttl=ttl,
        key_prefix="prediction:",
        exclude_args=['model'],  # Exclude model object from cache key
        condition=lambda result: result is not None
    )


def cache_dataset_stats(ttl: int = 1800):  # 30 minutes
    """Decorator for dataset statistics."""
    return cached(ttl=ttl, key_prefix="stats:")


def cache_document_content(ttl: int = 86400):  # 24 hours
    """Decorator for document content caching."""
    return cached(
        ttl=ttl,
        key_prefix="document:",
        condition=lambda result: result and len(result.get('content', '')) > 100
    )


class CacheNamespace:
    """Context manager for cache namespacing."""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.cache_manager = get_cache_manager()
        self.original_prefix = None
    
    def __enter__(self):
        if self.cache_manager:
            self.original_prefix = self.cache_manager.key_prefix
            self.cache_manager.key_prefix = f"{self.original_prefix}{self.namespace}:"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache_manager and self.original_prefix is not None:
            self.cache_manager.key_prefix = self.original_prefix
    
    def invalidate_all(self) -> int:
        """Invalidate all cache entries in this namespace."""
        if self.cache_manager:
            pattern = f"{self.namespace}:*"
            return self.cache_manager.delete_pattern(pattern)
        return 0


# Utility functions for manual cache operations
def get_cached(key: str, default: Any = None) -> Any:
    """Get value from cache manually."""
    cache_manager = get_cache_manager()
    return cache_manager.get(key, default) if cache_manager else default


def set_cached(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in cache manually."""
    cache_manager = get_cache_manager()
    return cache_manager.set(key, value, ttl) if cache_manager else False


def delete_cached(key: str) -> bool:
    """Delete value from cache manually."""
    cache_manager = get_cache_manager()
    return cache_manager.delete(key) if cache_manager else False


def invalidate_function_cache(func: Callable) -> int:
    """Invalidate all cache entries for a specific function."""
    if hasattr(func, '_cache_func_name'):
        cache_manager = get_cache_manager()
        if cache_manager:
            pattern = f"*{func._cache_func_name}:*"
            return cache_manager.delete_pattern(pattern)
    return 0