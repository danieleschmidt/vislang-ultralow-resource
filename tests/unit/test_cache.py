"""Tests for caching functionality."""

import pytest
import json
import time
from unittest.mock import Mock, patch, call

from vislang_ultralow.cache.cache_manager import CacheManager
from vislang_ultralow.cache.decorators import cached, cache_key, invalidate_cache


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def test_init_with_redis_client(self, mock_redis):
        """Test CacheManager initialization with Redis client."""
        cache = CacheManager(redis_client=mock_redis)
        
        assert cache.redis == mock_redis
        assert cache.default_ttl == 3600
        assert cache.key_prefix == "vislang:"
    
    def test_init_with_custom_params(self, mock_redis):
        """Test CacheManager initialization with custom parameters."""
        cache = CacheManager(
            redis_client=mock_redis,
            default_ttl=7200,
            key_prefix="test:",
            compression=False,
            serialization="pickle"
        )
        
        assert cache.default_ttl == 7200
        assert cache.key_prefix == "test:"
        assert cache.compression is False
        assert cache.serialization == "pickle"
    
    def test_generate_key(self, mock_redis):
        """Test cache key generation."""
        cache = CacheManager(redis_client=mock_redis, key_prefix="test:")
        
        key = cache._generate_key("mykey")
        assert key == "test:mykey"
    
    def test_serialize_json(self, mock_redis):
        """Test JSON serialization."""
        cache = CacheManager(redis_client=mock_redis, serialization="json")
        
        data = {"key": "value", "number": 42}
        serialized = cache._serialize(data)
        
        assert isinstance(serialized, bytes)
        # Should be able to deserialize back
        deserialized = cache._deserialize(serialized)
        assert deserialized == data
    
    def test_serialize_pickle(self, mock_redis):
        """Test pickle serialization."""
        cache = CacheManager(redis_client=mock_redis, serialization="pickle")
        
        # Test with complex object
        data = {"list": [1, 2, 3], "set": {1, 2, 3}}
        serialized = cache._serialize(data)
        deserialized = cache._deserialize(serialized)
        
        assert deserialized["list"] == [1, 2, 3]
        assert deserialized["set"] == {1, 2, 3}
    
    def test_compression(self, mock_redis):
        """Test data compression for large values."""
        cache = CacheManager(redis_client=mock_redis, compression=True)
        
        # Large data that should be compressed
        large_data = "x" * 2000
        serialized = cache._serialize(large_data)
        
        # Should have compression flag
        assert serialized.startswith(b"GZIP:")
        
        # Should decompress correctly
        deserialized = cache._deserialize(serialized)
        assert deserialized == large_data
    
    def test_get_existing_key(self, mock_redis):
        """Test getting existing key from cache."""
        cache = CacheManager(redis_client=mock_redis)
        
        # Mock Redis to return serialized data
        test_data = {"test": "value"}
        serialized_data = cache._serialize(test_data)
        mock_redis.get.return_value = serialized_data
        
        result = cache.get("test_key")
        
        assert result == test_data
        mock_redis.get.assert_called_once_with("vislang:test_key")
        assert cache._stats["hits"] == 1
    
    def test_get_nonexistent_key(self, mock_redis):
        """Test getting non-existent key from cache."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.get.return_value = None
        
        result = cache.get("nonexistent_key", "default_value")
        
        assert result == "default_value"
        assert cache._stats["misses"] == 1
    
    def test_set_key(self, mock_redis):
        """Test setting key in cache."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.set.return_value = True
        
        result = cache.set("test_key", {"data": "value"}, ttl=300)
        
        assert result is True
        mock_redis.set.assert_called_once()
        assert cache._stats["sets"] == 1
    
    def test_set_with_nx_option(self, mock_redis):
        """Test setting key with NX (only if not exists) option."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.set.return_value = True
        
        cache.set("test_key", "value", nx=True)
        
        # Should call with nx=True
        call_args = mock_redis.set.call_args
        assert call_args.kwargs["nx"] is True
    
    def test_delete_key(self, mock_redis):
        """Test deleting key from cache."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.delete.return_value = 1
        
        result = cache.delete("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("vislang:test_key")
        assert cache._stats["deletes"] == 1
    
    def test_exists_key(self, mock_redis):
        """Test checking if key exists."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.exists.return_value = 1
        
        result = cache.exists("test_key")
        
        assert result is True
        mock_redis.exists.assert_called_once_with("vislang:test_key")
    
    def test_expire_key(self, mock_redis):
        """Test setting expiration for key."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.expire.return_value = True
        
        result = cache.expire("test_key", 600)
        
        assert result is True
        mock_redis.expire.assert_called_once_with("vislang:test_key", 600)
    
    def test_ttl_key(self, mock_redis):
        """Test getting TTL for key."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.ttl.return_value = 300
        
        result = cache.ttl("test_key")
        
        assert result == 300
        mock_redis.ttl.assert_called_once_with("vislang:test_key")
    
    def test_increment_key(self, mock_redis):
        """Test incrementing numeric value."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.incrby.return_value = 5
        
        result = cache.increment("counter", 2)
        
        assert result == 5
        mock_redis.incrby.assert_called_once_with("vislang:counter", 2)
    
    def test_get_multiple_keys(self, mock_redis):
        """Test getting multiple keys at once."""
        cache = CacheManager(redis_client=mock_redis)
        
        # Mock serialized data for multiple keys
        data1 = cache._serialize({"key1": "value1"})
        data2 = cache._serialize({"key2": "value2"})
        mock_redis.mget.return_value = [data1, data2, None]
        
        result = cache.get_multiple(["key1", "key2", "key3"])
        
        assert len(result) == 2  # Only 2 keys found
        assert result["key1"] == {"key1": "value1"}
        assert result["key2"] == {"key2": "value2"}
        assert "key3" not in result
    
    def test_set_multiple_keys(self, mock_redis):
        """Test setting multiple keys at once."""
        cache = CacheManager(redis_client=mock_redis)
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_redis.pipeline.return_value = mock_pipeline
        
        mapping = {"key1": "value1", "key2": "value2"}
        result = cache.set_multiple(mapping, ttl=300)
        
        assert result is True
        mock_pipeline.mset.assert_called_once()
        mock_pipeline.execute.assert_called_once()
    
    def test_delete_pattern(self, mock_redis):
        """Test deleting keys matching pattern."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.keys.return_value = ["vislang:key1", "vislang:key2"]
        mock_redis.delete.return_value = 2
        
        result = cache.delete_pattern("key*")
        
        assert result == 2
        mock_redis.keys.assert_called_once_with("vislang:key*")
        mock_redis.delete.assert_called_once_with("vislang:key1", "vislang:key2")
    
    def test_get_stats(self, mock_redis):
        """Test getting cache statistics."""
        cache = CacheManager(redis_client=mock_redis)
        
        # Simulate some operations
        cache._stats["hits"] = 10
        cache._stats["misses"] = 5
        
        mock_redis.info.return_value = {
            "used_memory_human": "1M",
            "connected_clients": 2
        }
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 10 / 15  # 10 hits out of 15 total
        assert "redis_info" in stats
    
    def test_health_check(self, mock_redis):
        """Test cache health check."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {"used_memory_human": "1M"}
        
        health = cache.health_check()
        
        assert health["available"] is True
        assert health["latency_ms"] is not None
        assert health["memory_usage"] == "1M"
        assert health["error"] is None
    
    def test_health_check_failure(self, mock_redis):
        """Test cache health check when Redis is down."""
        cache = CacheManager(redis_client=mock_redis)
        
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        health = cache.health_check()
        
        assert health["available"] is False
        assert health["error"] == "Connection failed"
    
    def test_no_redis_client(self):
        """Test behavior when Redis client is not available."""
        cache = CacheManager(redis_client=None)
        
        # All operations should return default values gracefully
        assert cache.get("key") is None
        assert cache.set("key", "value") is False
        assert cache.delete("key") is False
        assert cache.exists("key") is False


class TestCacheDecorators:
    """Test caching decorators."""
    
    def test_cache_key_generation(self):
        """Test cache key generation from arguments."""
        key1 = cache_key("arg1", "arg2", param1="value1", param2="value2")
        key2 = cache_key("arg1", "arg2", param2="value2", param1="value1")  # Different order
        
        # Should generate consistent keys regardless of kwarg order
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
    
    @patch('vislang_ultralow.cache.decorators.get_cache_manager')
    def test_cached_decorator_hit(self, mock_get_cache):
        """Test cached decorator with cache hit."""
        mock_cache = Mock()
        mock_cache.redis = True
        mock_cache.get.return_value = "cached_result"
        mock_get_cache.return_value = mock_cache
        
        @cached(ttl=300)
        def test_function(arg1, arg2=None):
            return f"result_{arg1}_{arg2}"
        
        result = test_function("test", arg2="value")
        
        assert result == "cached_result"
        mock_cache.get.assert_called_once()
        # Function should not be called since we got cache hit
    
    @patch('vislang_ultralow.cache.decorators.get_cache_manager')
    def test_cached_decorator_miss(self, mock_get_cache):
        """Test cached decorator with cache miss."""
        mock_cache = Mock()
        mock_cache.redis = True
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        @cached(ttl=300)
        def test_function(arg1, arg2=None):
            return f"result_{arg1}_{arg2}"
        
        result = test_function("test", arg2="value")
        
        assert result == "result_test_value"
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
    
    @patch('vislang_ultralow.cache.decorators.get_cache_manager')
    def test_cached_with_condition(self, mock_get_cache):
        """Test cached decorator with condition function."""
        mock_cache = Mock()
        mock_cache.redis = True
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        @cached(condition=lambda result: result is not None)
        def test_function(return_none=False):
            return None if return_none else "valid_result"
        
        # Should cache this result
        result1 = test_function(False)
        assert result1 == "valid_result"
        mock_cache.set.assert_called_once()
        
        # Should not cache this result
        mock_cache.reset_mock()
        result2 = test_function(True)
        assert result2 is None
        mock_cache.set.assert_not_called()
    
    @patch('vislang_ultralow.cache.decorators.get_cache_manager')
    def test_cached_exclude_args(self, mock_get_cache):
        """Test cached decorator excluding specific arguments."""
        mock_cache = Mock()
        mock_cache.redis = True
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        
        @cached(exclude_args=["sensitive_param"])
        def test_function(data, sensitive_param=None):
            return f"result_{data}"
        
        # Call with different sensitive_param values
        test_function("data1", sensitive_param="secret1")
        test_function("data1", sensitive_param="secret2")
        
        # Should generate same cache key (sensitive_param excluded)
        assert mock_cache.get.call_count == 2
        call1_key = mock_cache.get.call_args_list[0][0][0]
        call2_key = mock_cache.get.call_args_list[1][0][0]
        assert call1_key == call2_key
    
    @patch('vislang_ultralow.cache.decorators.get_cache_manager')
    def test_invalidate_cache_decorator(self, mock_get_cache):
        """Test cache invalidation decorator."""
        mock_cache = Mock()
        mock_cache.redis = True
        mock_cache.delete_pattern.return_value = 2
        mock_get_cache.return_value = mock_cache
        
        @cached()
        def cached_function():
            return "cached_result"
        
        @invalidate_cache(pattern="test:*")
        def invalidating_function():
            return "invalidated"
        
        result = invalidating_function()
        
        assert result == "invalidated"
        mock_cache.delete_pattern.assert_called_once_with("test:*")
    
    def test_specialized_decorators(self):
        """Test specialized caching decorators."""
        from vislang_ultralow.cache.decorators import (
            cache_ocr_result, cache_model_prediction, cache_dataset_stats
        )
        
        # Test that decorators are callable
        @cache_ocr_result()
        def mock_ocr():
            return {"text": "extracted", "confidence": 0.9}
        
        @cache_model_prediction()
        def mock_prediction():
            return {"prediction": "result"}
        
        @cache_dataset_stats()
        def mock_stats():
            return {"count": 100}
        
        # These should not raise errors
        assert callable(mock_ocr)
        assert callable(mock_prediction)
        assert callable(mock_stats)
    
    @patch('vislang_ultralow.cache.decorators.get_cache_manager')
    def test_cache_namespace_context(self, mock_get_cache):
        """Test cache namespace context manager."""
        from vislang_ultralow.cache.decorators import CacheNamespace
        
        mock_cache = Mock()
        mock_cache.key_prefix = "vislang:"
        mock_cache.delete_pattern.return_value = 5
        mock_get_cache.return_value = mock_cache
        
        with CacheNamespace("test_namespace") as ns:
            assert mock_cache.key_prefix == "vislang:test_namespace:"
            
            # Test invalidation within namespace
            count = ns.invalidate_all()
            assert count == 5
        
        # Should restore original prefix
        assert mock_cache.key_prefix == "vislang:"


@pytest.mark.integration
def test_cache_integration():
    """Integration test for caching functionality."""
    # This would test with a real Redis instance if available
    # For now, just test that components work together
    from vislang_ultralow.cache import get_cache_manager, cached
    
    @cached(ttl=60)
    def expensive_operation(param):
        time.sleep(0.01)  # Simulate expensive operation
        return f"result_for_{param}"
    
    # Even without Redis, should not crash
    result1 = expensive_operation("test")
    result2 = expensive_operation("test")
    
    assert result1 == "result_for_test"
    assert result2 == "result_for_test"


@pytest.mark.slow
def test_cache_performance():
    """Test cache performance with larger datasets."""
    from vislang_ultralow.cache.cache_manager import CacheManager
    
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    cache = CacheManager(redis_client=mock_redis)
    
    # Test serialization performance with larger data
    large_data = {"data": "x" * 10000, "list": list(range(1000))}
    
    start_time = time.time()
    serialized = cache._serialize(large_data)
    deserialized = cache._deserialize(serialized)
    end_time = time.time()
    
    assert deserialized == large_data
    assert end_time - start_time < 1.0  # Should be fast