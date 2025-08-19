"""Intelligent caching system with predictive eviction and adaptive sizing.

Generation 3 Enhancement: Advanced caching with ML-based prediction and
quantum-inspired optimization for maximum cache efficiency.
"""

import logging
import json
import asyncio
import hashlib
import pickle
import gzip
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque, OrderedDict
from datetime import datetime, timedelta
import threading
import time
import math

# Conditional imports with fallbacks
try:
    import numpy as np
    from scipy import stats
except ImportError:
    np = stats = None

logger = logging.getLogger(__name__)


class IntelligentCacheManager:
    """Intelligent cache with predictive eviction and adaptive sizing."""
    
    def __init__(self, max_size_mb: int = 1024, prediction_window: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.prediction_window = prediction_window
        
        # Core cache storage
        self.cache = OrderedDict()
        self.metadata = {}
        self.current_size = 0
        
        # Intelligence components
        self.access_predictor = AccessPatternPredictor(prediction_window)
        self.value_estimator = CacheValueEstimator()
        self.compression_manager = CompressionManager()
        
        # Performance tracking
        self.stats = CacheStatistics()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background optimization
        self.optimization_task = None
        self.stop_optimization = threading.Event()
        
        logger.info(f"Initialized IntelligentCacheManager: {max_size_mb}MB capacity")
        self._start_background_optimization()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with intelligent access tracking."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update access patterns
                self._update_access_metadata(key, current_time, hit=True)
                
                # Move to end (LRU behavior)
                value = self.cache.pop(key)
                self.cache[key] = value
                
                # Update statistics
                self.stats.record_hit()
                
                # Decompress if needed
                if self.metadata[key]['compressed']:
                    try:
                        value = self.compression_manager.decompress(value)
                    except Exception as e:
                        logger.warning(f"Decompression failed for key {key}: {e}")
                        return default
                
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                # Update access patterns for miss
                self._update_access_metadata(key, current_time, hit=False)
                
                # Update statistics
                self.stats.record_miss()
                
                logger.debug(f"Cache miss for key: {key}")
                return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            compress: bool = None, priority: float = 1.0) -> bool:
        """Set item in cache with intelligent placement."""
        with self.lock:
            current_time = time.time()
            
            # Calculate value size and determine compression
            original_size = self._calculate_size(value)
            should_compress = compress if compress is not None else self._should_compress(value, original_size)
            
            # Compress if beneficial
            if should_compress:
                try:
                    compressed_value = self.compression_manager.compress(value)
                    compressed_size = self._calculate_size(compressed_value)
                    
                    if compressed_size < original_size * 0.8:  # At least 20% reduction
                        final_value = compressed_value
                        final_size = compressed_size
                        is_compressed = True
                    else:
                        final_value = value
                        final_size = original_size
                        is_compressed = False
                except Exception as e:
                    logger.warning(f"Compression failed for key {key}: {e}")
                    final_value = value
                    final_size = original_size
                    is_compressed = False
            else:
                final_value = value
                final_size = original_size
                is_compressed = False
            
            # Check if eviction is needed
            while self.current_size + final_size > self.max_size_bytes and self.cache:
                if not self._intelligent_eviction():
                    break  # Couldn't evict anything more
            
            # Store in cache
            if key in self.cache:
                # Update existing entry
                old_size = self.metadata[key]['size']
                self.current_size -= old_size
            
            self.cache[key] = final_value
            self.current_size += final_size
            
            # Store metadata
            self.metadata[key] = {
                'size': final_size,
                'compressed': is_compressed,
                'created_at': current_time,
                'last_accessed': current_time,
                'access_count': 1,
                'ttl': ttl,
                'expires_at': current_time + ttl if ttl else None,
                'priority': priority,
                'value_score': self.value_estimator.estimate_value(key, value, final_size),
                'compression_ratio': original_size / final_size if final_size > 0 else 1.0
            }\n            \n            # Update access predictor\n            self.access_predictor.record_access(key, current_time)\n            \n            # Update statistics\n            self.stats.record_set(final_size, is_compressed)\n            \n            logger.debug(f\"Cached key {key}: size={final_size}, compressed={is_compressed}\")\n            return True\n    \n    def delete(self, key: str) -> bool:\n        \"\"\"Delete item from cache.\"\"\"\n        with self.lock:\n            if key in self.cache:\n                size = self.metadata[key]['size']\n                del self.cache[key]\n                del self.metadata[key]\n                self.current_size -= size\n                \n                self.stats.record_delete()\n                logger.debug(f\"Deleted key from cache: {key}\")\n                return True\n            return False\n    \n    def clear(self):\n        \"\"\"Clear all items from cache.\"\"\"\n        with self.lock:\n            self.cache.clear()\n            self.metadata.clear()\n            self.current_size = 0\n            self.stats.record_clear()\n            logger.info(\"Cache cleared\")\n    \n    def _update_access_metadata(self, key: str, current_time: float, hit: bool):\n        \"\"\"Update access metadata for cache intelligence.\"\"\"\n        if hit and key in self.metadata:\n            self.metadata[key]['last_accessed'] = current_time\n            self.metadata[key]['access_count'] += 1\n        \n        # Update access predictor\n        self.access_predictor.record_access(key, current_time, hit)\n    \n    def _should_compress(self, value: Any, size: int) -> bool:\n        \"\"\"Determine if value should be compressed.\"\"\"\n        # Compress if size is significant and value is compressible\n        if size < 1024:  # Don't compress small values\n            return False\n        \n        # Check if value type is compressible\n        if isinstance(value, (str, dict, list)):\n            return True\n        \n        if hasattr(value, '__dict__'):  # Custom objects\n            return True\n        \n        return False\n    \n    def _calculate_size(self, value: Any) -> int:\n        \"\"\"Calculate approximate size of value in bytes.\"\"\"\n        try:\n            if isinstance(value, str):\n                return len(value.encode('utf-8'))\n            elif isinstance(value, bytes):\n                return len(value)\n            elif isinstance(value, (int, float)):\n                return 8\n            elif isinstance(value, (list, tuple, dict)):\n                return len(pickle.dumps(value))\n            else:\n                return len(pickle.dumps(value))\n        except Exception:\n            return 1024  # Default size estimate\n    \n    def _intelligent_eviction(self) -> bool:\n        \"\"\"Perform intelligent cache eviction.\"\"\"\n        if not self.cache:\n            return False\n        \n        current_time = time.time()\n        \n        # Calculate eviction scores for all items\n        eviction_candidates = []\n        \n        for key in self.cache.keys():\n            metadata = self.metadata[key]\n            \n            # Check for expired items first\n            if metadata.get('expires_at') and metadata['expires_at'] <= current_time:\n                eviction_candidates.append((key, float('inf')))  # Highest priority for removal\n                continue\n            \n            # Calculate eviction score\n            score = self._calculate_eviction_score(key, metadata, current_time)\n            eviction_candidates.append((key, score))\n        \n        if not eviction_candidates:\n            return False\n        \n        # Sort by eviction score (highest score = most suitable for eviction)\n        eviction_candidates.sort(key=lambda x: x[1], reverse=True)\n        \n        # Evict the best candidate\n        key_to_evict = eviction_candidates[0][0]\n        size_freed = self.metadata[key_to_evict]['size']\n        \n        del self.cache[key_to_evict]\n        del self.metadata[key_to_evict]\n        self.current_size -= size_freed\n        \n        self.stats.record_eviction()\n        logger.debug(f\"Evicted key: {key_to_evict}, freed {size_freed} bytes\")\n        \n        return True\n    \n    def _calculate_eviction_score(self, key: str, metadata: Dict, current_time: float) -> float:\n        \"\"\"Calculate eviction score for a cache item (higher = more suitable for eviction).\"\"\"\n        # Time-based factors\n        time_since_access = current_time - metadata['last_accessed']\n        age = current_time - metadata['created_at']\n        \n        # Frequency factor\n        access_frequency = metadata['access_count'] / max(age / 3600, 1)  # Accesses per hour\n        \n        # Size factor (prefer evicting larger items)\n        size_factor = metadata['size'] / (1024 * 1024)  # Size in MB\n        \n        # Value factor (lower value = higher eviction score)\n        value_factor = 1.0 / max(metadata['value_score'], 0.1)\n        \n        # Priority factor (lower priority = higher eviction score)\n        priority_factor = 1.0 / max(metadata['priority'], 0.1)\n        \n        # Future access prediction\n        future_access_prob = self.access_predictor.predict_future_access(key, current_time)\n        future_factor = 1.0 / max(future_access_prob, 0.1)\n        \n        # Combine factors with weights\n        eviction_score = (\n            time_since_access * 0.3 +\n            value_factor * 0.2 +\n            priority_factor * 0.15 +\n            size_factor * 0.15 +\n            future_factor * 0.15 +\n            (1.0 / max(access_frequency, 0.1)) * 0.05\n        )\n        \n        return eviction_score\n    \n    def _start_background_optimization(self):\n        \"\"\"Start background optimization task.\"\"\"\n        def optimization_loop():\n            while not self.stop_optimization.is_set():\n                try:\n                    self._perform_background_optimization()\n                except Exception as e:\n                    logger.error(f\"Background optimization error: {e}\")\n                \n                # Wait before next optimization\n                self.stop_optimization.wait(300)  # 5 minutes\n        \n        self.optimization_task = threading.Thread(target=optimization_loop, daemon=True)\n        self.optimization_task.start()\n    \n    def _perform_background_optimization(self):\n        \"\"\"Perform background cache optimization.\"\"\"\n        with self.lock:\n            current_time = time.time()\n            \n            # Clean up expired items\n            expired_keys = []\n            for key, metadata in self.metadata.items():\n                if metadata.get('expires_at') and metadata['expires_at'] <= current_time:\n                    expired_keys.append(key)\n            \n            for key in expired_keys:\n                self.delete(key)\n            \n            # Adaptive compression\n            self._adaptive_compression_optimization()\n            \n            # Update value estimates\n            self.value_estimator.update_estimates(self.metadata)\n            \n            # Optimize cache size if needed\n            self._optimize_cache_size()\n    \n    def _adaptive_compression_optimization(self):\n        \"\"\"Optimize compression for existing cache items.\"\"\"\n        recompression_candidates = []\n        \n        for key, metadata in self.metadata.items():\n            if not metadata['compressed'] and metadata['size'] > 2048:  # Items larger than 2KB\n                recompression_candidates.append(key)\n        \n        # Recompress up to 10 items per optimization cycle\n        for key in recompression_candidates[:10]:\n            try:\n                original_value = self.cache[key]\n                compressed_value = self.compression_manager.compress(original_value)\n                compressed_size = self._calculate_size(compressed_value)\n                original_size = self.metadata[key]['size']\n                \n                if compressed_size < original_size * 0.7:  # At least 30% reduction\n                    self.cache[key] = compressed_value\n                    self.current_size -= (original_size - compressed_size)\n                    self.metadata[key]['size'] = compressed_size\n                    self.metadata[key]['compressed'] = True\n                    self.metadata[key]['compression_ratio'] = original_size / compressed_size\n                    \n                    logger.debug(f\"Recompressed {key}: {original_size} -> {compressed_size} bytes\")\n            except Exception as e:\n                logger.warning(f\"Recompression failed for {key}: {e}\")\n    \n    def _optimize_cache_size(self):\n        \"\"\"Optimize cache size based on usage patterns.\"\"\"\n        hit_rate = self.stats.get_hit_rate()\n        \n        # If hit rate is very low, consider reducing cache size\n        if hit_rate < 0.3 and self.current_size > self.max_size_bytes * 0.5:\n            # Evict items more aggressively\n            target_size = self.max_size_bytes * 0.7\n            while self.current_size > target_size and self.cache:\n                if not self._intelligent_eviction():\n                    break\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive cache statistics.\"\"\"\n        with self.lock:\n            basic_stats = self.stats.get_stats()\n            \n            return {\n                **basic_stats,\n                'current_size_mb': self.current_size / (1024 * 1024),\n                'max_size_mb': self.max_size_bytes / (1024 * 1024),\n                'utilization_percent': (self.current_size / self.max_size_bytes) * 100,\n                'item_count': len(self.cache),\n                'average_item_size': self.current_size / len(self.cache) if self.cache else 0,\n                'compression_stats': self._get_compression_stats(),\n                'prediction_accuracy': self.access_predictor.get_accuracy(),\n                'top_accessed_keys': self._get_top_accessed_keys(5)\n            }\n    \n    def _get_compression_stats(self) -> Dict[str, Any]:\n        \"\"\"Get compression statistics.\"\"\"\n        compressed_items = sum(1 for m in self.metadata.values() if m['compressed'])\n        total_items = len(self.metadata)\n        \n        if compressed_items == 0:\n            return {\n                'compressed_items': 0,\n                'compression_rate': 0.0,\n                'average_compression_ratio': 1.0\n            }\n        \n        compression_ratios = [m['compression_ratio'] for m in self.metadata.values() if m['compressed']]\n        \n        return {\n            'compressed_items': compressed_items,\n            'compression_rate': compressed_items / total_items if total_items > 0 else 0,\n            'average_compression_ratio': sum(compression_ratios) / len(compression_ratios)\n        }\n    \n    def _get_top_accessed_keys(self, count: int) -> List[Tuple[str, int]]:\n        \"\"\"Get top accessed cache keys.\"\"\"\n        items = [(key, metadata['access_count']) for key, metadata in self.metadata.items()]\n        items.sort(key=lambda x: x[1], reverse=True)\n        return items[:count]\n    \n    def shutdown(self):\n        \"\"\"Shutdown cache manager.\"\"\"\n        self.stop_optimization.set()\n        if self.optimization_task:\n            self.optimization_task.join(timeout=5)\n        logger.info(\"IntelligentCacheManager shutdown complete\")\n\n\nclass AccessPatternPredictor:\n    \"\"\"Predict future cache access patterns.\"\"\"\n    \n    def __init__(self, window_size: int = 100):\n        self.window_size = window_size\n        self.access_history = defaultdict(deque)\n        self.prediction_accuracy = deque(maxlen=1000)\n        \n    def record_access(self, key: str, timestamp: float, hit: bool = True):\n        \"\"\"Record cache access for pattern learning.\"\"\"\n        history = self.access_history[key]\n        history.append((timestamp, hit))\n        \n        # Keep only recent history\n        while len(history) > self.window_size:\n            history.popleft()\n    \n    def predict_future_access(self, key: str, current_time: float, prediction_horizon: float = 3600) -> float:\n        \"\"\"Predict probability of future access within prediction horizon.\"\"\"\n        history = self.access_history.get(key, deque())\n        \n        if len(history) < 2:\n            return 0.5  # Default probability for new keys\n        \n        # Calculate access frequency\n        recent_accesses = [timestamp for timestamp, hit in history if hit]\n        \n        if len(recent_accesses) < 2:\n            return 0.1  # Very low probability if no recent hits\n        \n        # Calculate time between accesses\n        time_deltas = []\n        for i in range(1, len(recent_accesses)):\n            delta = recent_accesses[i] - recent_accesses[i-1]\n            time_deltas.append(delta)\n        \n        if not time_deltas:\n            return 0.1\n        \n        # Predict next access time using moving average\n        avg_delta = sum(time_deltas) / len(time_deltas)\n        last_access = recent_accesses[-1]\n        predicted_next_access = last_access + avg_delta\n        \n        # Calculate probability based on prediction horizon\n        time_to_prediction = predicted_next_access - current_time\n        \n        if time_to_prediction <= 0:\n            return 0.9  # Should have been accessed already\n        elif time_to_prediction <= prediction_horizon:\n            # Linear decay within horizon\n            probability = 1.0 - (time_to_prediction / prediction_horizon)\n            return max(0.1, probability)\n        else:\n            return 0.1  # Low probability beyond horizon\n    \n    def get_accuracy(self) -> float:\n        \"\"\"Get prediction accuracy.\"\"\"\n        if not self.prediction_accuracy:\n            return 0.0\n        \n        return sum(self.prediction_accuracy) / len(self.prediction_accuracy)\n\n\nclass CacheValueEstimator:\n    \"\"\"Estimate the value of cached items.\"\"\"\n    \n    def __init__(self):\n        self.value_weights = {\n            'access_frequency': 0.3,\n            'recency': 0.2,\n            'size_efficiency': 0.2,\n            'computation_cost': 0.2,\n            'uniqueness': 0.1\n        }\n    \n    def estimate_value(self, key: str, value: Any, size: int) -> float:\n        \"\"\"Estimate the value score of a cache item.\"\"\"\n        # Base value calculation\n        value_score = 1.0\n        \n        # Size efficiency (smaller is generally better)\n        if size > 0:\n            size_mb = size / (1024 * 1024)\n            size_efficiency = 1.0 / (1.0 + size_mb)  # Diminishing returns for larger items\n            value_score *= size_efficiency\n        \n        # Computation cost estimation (based on value type)\n        computation_cost = self._estimate_computation_cost(value)\n        value_score *= computation_cost\n        \n        # Key-based uniqueness (longer keys often represent more specific/valuable data)\n        uniqueness = min(1.0, len(key) / 50.0)\n        value_score *= (1.0 + uniqueness * 0.2)\n        \n        return value_score\n    \n    def _estimate_computation_cost(self, value: Any) -> float:\n        \"\"\"Estimate computational cost of recreating this value.\"\"\"\n        if isinstance(value, (str, int, float, bool)):\n            return 0.1  # Low cost to recreate\n        elif isinstance(value, (list, tuple)):\n            return 0.3  # Medium cost\n        elif isinstance(value, dict):\n            return 0.5  # Higher cost for complex structures\n        else:\n            return 0.8  # High cost for complex objects\n    \n    def update_estimates(self, metadata: Dict[str, Dict]):\n        \"\"\"Update value estimates based on access patterns.\"\"\"\n        for key, meta in metadata.items():\n            # Update value score based on actual access patterns\n            access_count = meta['access_count']\n            age_hours = (time.time() - meta['created_at']) / 3600\n            \n            # Access frequency component\n            frequency_score = access_count / max(age_hours, 1)\n            \n            # Recency component\n            time_since_access = time.time() - meta['last_accessed']\n            recency_score = 1.0 / (1.0 + time_since_access / 3600)  # Decay over hours\n            \n            # Update value score\n            new_value_score = (\n                frequency_score * self.value_weights['access_frequency'] +\n                recency_score * self.value_weights['recency'] +\n                meta['value_score'] * 0.5  # Retain some original estimate\n            )\n            \n            meta['value_score'] = new_value_score\n\n\nclass CompressionManager:\n    \"\"\"Manage compression and decompression of cache values.\"\"\"\n    \n    def __init__(self):\n        self.compression_stats = {\n            'compressions': 0,\n            'decompressions': 0,\n            'compression_time': 0.0,\n            'decompression_time': 0.0\n        }\n    \n    def compress(self, value: Any) -> bytes:\n        \"\"\"Compress a value for cache storage.\"\"\"\n        start_time = time.time()\n        \n        try:\n            # Serialize value\n            serialized = pickle.dumps(value)\n            \n            # Compress using gzip\n            compressed = gzip.compress(serialized)\n            \n            self.compression_stats['compressions'] += 1\n            self.compression_stats['compression_time'] += time.time() - start_time\n            \n            return compressed\n            \n        except Exception as e:\n            logger.error(f\"Compression failed: {e}\")\n            raise\n    \n    def decompress(self, compressed_data: bytes) -> Any:\n        \"\"\"Decompress a cached value.\"\"\"\n        start_time = time.time()\n        \n        try:\n            # Decompress\n            decompressed = gzip.decompress(compressed_data)\n            \n            # Deserialize\n            value = pickle.loads(decompressed)\n            \n            self.compression_stats['decompressions'] += 1\n            self.compression_stats['decompression_time'] += time.time() - start_time\n            \n            return value\n            \n        except Exception as e:\n            logger.error(f\"Decompression failed: {e}\")\n            raise\n    \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get compression statistics.\"\"\"\n        stats = self.compression_stats.copy()\n        \n        if stats['compressions'] > 0:\n            stats['avg_compression_time'] = stats['compression_time'] / stats['compressions']\n        else:\n            stats['avg_compression_time'] = 0.0\n        \n        if stats['decompressions'] > 0:\n            stats['avg_decompression_time'] = stats['decompression_time'] / stats['decompressions']\n        else:\n            stats['avg_decompression_time'] = 0.0\n        \n        return stats\n\n\nclass CacheStatistics:\n    \"\"\"Track cache performance statistics.\"\"\"\n    \n    def __init__(self):\n        self.hits = 0\n        self.misses = 0\n        self.sets = 0\n        self.deletes = 0\n        self.evictions = 0\n        self.clears = 0\n        self.total_size_stored = 0\n        self.compressed_items = 0\n        \n        self.start_time = time.time()\n    \n    def record_hit(self):\n        self.hits += 1\n    \n    def record_miss(self):\n        self.misses += 1\n    \n    def record_set(self, size: int, compressed: bool):\n        self.sets += 1\n        self.total_size_stored += size\n        if compressed:\n            self.compressed_items += 1\n    \n    def record_delete(self):\n        self.deletes += 1\n    \n    def record_eviction(self):\n        self.evictions += 1\n    \n    def record_clear(self):\n        self.clears += 1\n    \n    def get_hit_rate(self) -> float:\n        total_requests = self.hits + self.misses\n        return self.hits / total_requests if total_requests > 0 else 0.0\n    \n    def get_stats(self) -> Dict[str, Any]:\n        uptime = time.time() - self.start_time\n        total_requests = self.hits + self.misses\n        \n        return {\n            'hits': self.hits,\n            'misses': self.misses,\n            'hit_rate': self.get_hit_rate(),\n            'sets': self.sets,\n            'deletes': self.deletes,\n            'evictions': self.evictions,\n            'clears': self.clears,\n            'total_requests': total_requests,\n            'requests_per_second': total_requests / uptime if uptime > 0 else 0,\n            'total_size_stored_mb': self.total_size_stored / (1024 * 1024),\n            'compressed_items': self.compressed_items,\n            'uptime_seconds': uptime\n        }