"""Cache management with Redis backend."""

import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Union, Dict, List, Callable
from datetime import datetime, timedelta
import redis
from redis import Redis
import os

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager with advanced features."""
    
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "vislang:",
        compression: bool = True,
        serialization: str = "json"
    ):
        """Initialize cache manager.
        
        Args:
            redis_client: Existing Redis client
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            key_prefix: Prefix for all cache keys
            compression: Enable compression for large values
            serialization: Serialization method ('json' or 'pickle')
        """
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.compression = compression
        self.serialization = serialization
        
        # Initialize Redis client
        if redis_client:
            self.redis = redis_client
        else:
            redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=False,  # Keep as bytes for pickle support
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                self.redis.ping()
                logger.info("Cache manager initialized with Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis = None
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    def _generate_key(self, key: str) -> str:
        """Generate prefixed cache key."""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.serialization == "json":
                data = json.dumps(value, ensure_ascii=False).encode('utf-8')
            else:  # pickle
                data = pickle.dumps(value)
            
            if self.compression and len(data) > 1024:  # Compress if > 1KB
                import gzip
                data = gzip.compress(data)
                # Add compression flag
                data = b"GZIP:" + data
            
            return data
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Check for compression
            if data.startswith(b"GZIP:"):
                import gzip
                data = gzip.decompress(data[5:])
            
            if self.serialization == "json":
                return json.loads(data.decode('utf-8'))
            else:  # pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not self.redis:
            return default
        
        try:
            cache_key = self._generate_key(key)
            data = self.redis.get(cache_key)
            
            if data is None:
                self._stats['misses'] += 1
                return default
            
            self._stats['hits'] += 1
            return self._deserialize(data)
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self._stats['errors'] += 1
            return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
        """
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_key(key)
            data = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            result = self.redis.set(cache_key, data, ex=ttl, nx=nx, xx=xx)
            if result:
                self._stats['sets'] += 1
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_key(key)
            result = self.redis.delete(cache_key)
            if result:
                self._stats['deletes'] += 1
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self._stats['errors'] += 1
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_key(key)
            return bool(self.redis.exists(cache_key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key."""
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_key(key)
            return bool(self.redis.expire(cache_key, ttl))
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get time-to-live for key."""
        if not self.redis:
            return -1
        
        try:
            cache_key = self._generate_key(key)
            return self.redis.ttl(cache_key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -1
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value."""
        if not self.redis:
            return None
        
        try:
            cache_key = self._generate_key(key)
            return self.redis.incrby(cache_key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement numeric value."""
        return self.increment(key, -amount)
    
    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.redis or not keys:
            return {}
        
        try:
            cache_keys = [self._generate_key(key) for key in keys]
            values = self.redis.mget(cache_keys)
            
            result = {}
            for i, (original_key, data) in enumerate(zip(keys, values)):
                if data is not None:
                    try:
                        result[original_key] = self._deserialize(data)
                        self._stats['hits'] += 1
                    except Exception as e:
                        logger.error(f"Error deserializing key {original_key}: {e}")
                        self._stats['errors'] += 1
                else:
                    self._stats['misses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_multiple error: {e}")
            self._stats['errors'] += 1
            return {}
    
    def set_multiple(
        self, 
        mapping: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        if not self.redis or not mapping:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            cache_mapping = {}
            
            for key, value in mapping.items():
                cache_key = self._generate_key(key)
                cache_mapping[cache_key] = self._serialize(value)
            
            # Use pipeline for better performance
            pipe = self.redis.pipeline()
            pipe.mset(cache_mapping)
            
            # Set expiration for each key
            for cache_key in cache_mapping.keys():
                pipe.expire(cache_key, ttl)
            
            pipe.execute()
            self._stats['sets'] += len(mapping)
            return True
            
        except Exception as e:
            logger.error(f"Cache set_multiple error: {e}")
            self._stats['errors'] += 1
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        if not self.redis:
            return 0
        
        try:
            pattern_key = self._generate_key(pattern)
            keys = self.redis.keys(pattern_key)
            
            if keys:
                deleted = self.redis.delete(*keys)
                self._stats['deletes'] += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache delete_pattern error: {e}")
            self._stats['errors'] += 1
            return 0
    
    def flush_all(self) -> bool:
        """Flush all cache entries with prefix."""
        if not self.redis:
            return False
        
        try:
            pattern = f"{self.key_prefix}*"
            return self.delete_pattern("*") > 0
        except Exception as e:
            logger.error(f"Cache flush_all error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.copy()
        
        total_ops = stats['hits'] + stats['misses']
        if total_ops > 0:
            stats['hit_rate'] = stats['hits'] / total_ops
        else:
            stats['hit_rate'] = 0.0
        
        if self.redis:
            try:
                info = self.redis.info()
                stats['redis_info'] = {
                    'used_memory': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except Exception as e:
                logger.error(f"Error getting Redis info: {e}")
                stats['redis_info'] = {}
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        health = {
            'available': False,
            'latency_ms': None,
            'memory_usage': None,
            'error': None
        }
        
        if not self.redis:
            health['error'] = "Redis client not initialized"
            return health
        
        try:
            # Measure latency
            import time
            start = time.time()
            self.redis.ping()
            latency = (time.time() - start) * 1000
            
            health['available'] = True
            health['latency_ms'] = round(latency, 2)
            
            # Get memory usage
            info = self.redis.info()
            health['memory_usage'] = info.get('used_memory_human', 'N/A')
            
        except Exception as e:
            health['error'] = str(e)
        
        return health


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def invalidate_cache_pattern(pattern: str) -> int:
    """Invalidate cache entries matching pattern."""
    cache = get_cache_manager()
    return cache.delete_pattern(pattern)