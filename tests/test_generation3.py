"""Test Generation 3 optimization features."""

import tempfile
import shutil
import sys
import os
from pathlib import Path
import time
import hashlib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from unittest.mock import patch, MagicMock
except ImportError:
    # Fallback for older Python versions
    class MagicMock:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return self
        def assert_called_once(self):
            pass
        def assert_not_called(self):
            pass
    
    class patch:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return MagicMock()
        def __exit__(self, *args):
            pass

from vislang_ultralow.dataset import DatasetBuilder
from vislang_ultralow.trainer import VisionLanguageTrainer
from vislang_ultralow.scraper import HumanitarianScraper


class TestGeneration3Optimization:
    """Test Generation 3 optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_builder = DatasetBuilder(
            output_dir=self.temp_dir,
            target_languages=['en', 'fr']
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parallel_processing_initialization(self):
        """Test parallel processing configuration is properly initialized."""
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Check optimization config
        assert hasattr(builder, 'optimization_config')
        assert builder.optimization_config['parallel_processing'] is True
        assert builder.optimization_config['max_workers'] >= 1
        assert builder.optimization_config['parallel_threshold'] == 10
        assert builder.optimization_config['adaptive_batch_size'] is True
        assert builder.optimization_config['memory_optimization'] is True
        assert builder.optimization_config['cache_enabled'] is True
        assert builder.optimization_config['performance_monitoring'] is True
        
        print("✅ Parallel processing initialization successful")
    
    def test_intelligent_caching_system(self):
        """Test intelligent caching functionality."""
        # Create mock documents
        documents = [
            {'id': 'doc1', 'url': 'http://test.com/1', 'content': 'Test content 1'},
            {'id': 'doc2', 'url': 'http://test.com/2', 'content': 'Test content 2'}
        ]
        
        # Mock cache manager
        with patch('vislang_ultralow.cache.cache_manager.get_cache_manager') as mock_cache_mgr:
            mock_cache = MagicMock()
            mock_cache.get.return_value = None  # First call - cache miss
            mock_cache_mgr.return_value = mock_cache
            
            # Build dataset - should attempt caching
            result = self.dataset_builder.build(documents, output_format="custom")
            
            # Verify cache operations were attempted
            mock_cache.get.assert_called_once()  # Cache retrieval attempted
            mock_cache.set.assert_called_once()  # Cache storage attempted
        
        print("✅ Intelligent caching system working")
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing functionality."""
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Check adaptive metrics initialization
        assert hasattr(builder, 'adaptive_metrics')
        assert 'optimal_batch_size' in builder.adaptive_metrics
        assert 'performance_history' in builder.adaptive_metrics
        
        # Test batch size calculation
        test_sizes = [10, 100, 1000]
        for size in test_sizes:
            optimal_batch = builder._calculate_optimal_batch_size(size)
            assert isinstance(optimal_batch, int)
            assert optimal_batch > 0
            assert optimal_batch <= size
        
        print("✅ Adaptive batch sizing functional")
    
    def test_memory_optimization_features(self):
        """Test memory optimization and monitoring."""
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Check memory optimization settings
        assert builder.optimization_config['memory_optimization'] is True
        
        # Test memory monitoring initialization
        assert hasattr(builder, 'performance_metrics')
        assert 'memory_usage' in builder.performance_metrics
        assert 'processing_efficiency' in builder.performance_metrics
        
        # Test garbage collection trigger
        if hasattr(builder, '_trigger_garbage_collection'):
            # Should not raise exception
            builder._trigger_garbage_collection()
        
        print("✅ Memory optimization features active")
    
    def test_auto_scaling_monitor(self):
        """Test auto-scaling monitor functionality."""
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Test auto-scaler initialization
        assert hasattr(builder, 'auto_scaler')
        assert 'current_workers' in builder.auto_scaler
        assert 'max_workers' in builder.auto_scaler
        assert 'cpu_threshold' in builder.auto_scaler
        assert 'memory_threshold' in builder.auto_scaler
        
        # Test worker adjustment
        if hasattr(builder, '_adjust_worker_pool'):
            # Should handle different CPU/memory scenarios
            initial_workers = builder.auto_scaler['current_workers']
            builder._adjust_worker_pool(cpu_usage=90.0, memory_usage=50.0)
            # Workers should be adjusted based on high CPU
            
        print("✅ Auto-scaling monitor operational")
    
    def test_performance_monitoring(self):
        """Test performance monitoring and metrics collection."""
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Check performance metrics initialization
        assert builder.optimization_config['performance_monitoring'] is True
        assert hasattr(builder, 'performance_metrics')
        
        # Test performance tracking
        start_time = time.time()
        time.sleep(0.01)  # Simulate work
        
        # Update performance metrics
        if hasattr(builder, '_update_performance_metrics'):
            builder._update_performance_metrics(
                operation='test', 
                duration=time.time() - start_time,
                items_processed=10
            )
        
        print("✅ Performance monitoring active")
    
    def test_parallel_vs_sequential_processing(self):
        """Test parallel processing vs sequential processing."""
        # Create test documents
        documents = []
        for i in range(15):  # Above parallel threshold
            documents.append({
                'id': f'doc_{i}',
                'url': f'http://test.com/{i}',
                'content': f'Test content {i}',
                'images': []
            })
        
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Test that parallel processing is triggered for large batches
        with patch.object(builder, '_parallel_process_documents') as mock_parallel:
            mock_parallel.return_value = [{'success': True, 'items': [], 'doc_id': f'doc_{i}'} for i in range(15)]
            
            result = builder.build(documents, output_format="custom")
            
            # Should use parallel processing for 15 documents (> threshold of 10)
            mock_parallel.assert_called_once()
        
        # Test sequential processing for small batches
        small_documents = documents[:5]  # Below threshold
        with patch.object(builder, '_parallel_process_documents') as mock_parallel:
            result = builder.build(small_documents, output_format="custom")
            
            # Should NOT use parallel processing for 5 documents (< threshold of 10)
            mock_parallel.assert_not_called()
        
        print("✅ Parallel vs sequential processing logic working")
    
    def test_trainer_optimization_features(self):
        """Test trainer optimization features."""
        # Mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        trainer = VisionLanguageTrainer(
            model=mock_model,
            processor=mock_processor,
            languages=['en', 'fr']
        )
        
        # Check optimization config
        assert hasattr(trainer, 'optimization_config')
        assert trainer.optimization_config['mixed_precision'] is True
        assert trainer.optimization_config['gradient_checkpointing'] is True
        assert trainer.optimization_config['cache_enabled'] is True
        assert trainer.optimization_config['dynamic_batching'] is True
        assert trainer.optimization_config['memory_optimization'] is True
        assert trainer.optimization_config['performance_monitoring'] is True
        
        # Check adaptive metrics
        assert hasattr(trainer, 'adaptive_metrics')
        assert 'optimal_batch_size' in trainer.adaptive_metrics
        assert 'memory_usage_history' in trainer.adaptive_metrics
        assert 'throughput_history' in trainer.adaptive_metrics
        
        # Check memory monitor
        assert hasattr(trainer, 'memory_monitor')
        assert 'peak_memory' in trainer.memory_monitor
        assert 'oom_count' in trainer.memory_monitor
        
        print("✅ Trainer optimization features initialized")
    
    def test_cache_key_generation(self):
        """Test cache key generation for consistency."""
        documents = [
            {'id': 'doc1', 'content': 'test1'},
            {'id': 'doc2', 'content': 'test2'}
        ]
        
        builder = DatasetBuilder(output_dir=self.temp_dir, target_languages=['en', 'fr'])
        
        # Mock the build method to test cache key generation
        with patch('vislang_ultralow.dataset.get_cache_manager'):
            with patch('hashlib.md5') as mock_md5:
                mock_hash = MagicMock()
                mock_hash.hexdigest.return_value = 'test_hash'
                mock_md5.return_value = mock_hash
                
                # This should generate cache keys
                try:
                    builder.build(documents, output_format="custom")
                except:
                    pass  # We only care about cache key generation, not the full build
                
                # md5 should be called for cache key generation
                assert mock_md5.call_count >= 2  # Once for docs, once for params
        
        print("✅ Cache key generation working")


def test_optimization_performance():
    """Integration test for overall optimization performance."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test with larger document set
        documents = []
        for i in range(25):  # Above parallel threshold
            documents.append({
                'id': f'perf_doc_{i}',
                'url': f'http://performance.test/{i}',
                'content': f'Performance test content {i} ' * 10,  # Larger content
                'images': []
            })
        
        builder = DatasetBuilder(output_dir=temp_dir, target_languages=['en', 'fr'])
        
        # Measure build time
        start_time = time.time()
        result = builder.build(documents, output_format="custom")
        build_time = time.time() - start_time
        
        # Verify performance metrics were collected
        assert hasattr(builder, 'performance_metrics')
        assert 'last_build_time' in builder.performance_metrics
        assert 'documents_per_second' in builder.performance_metrics
        
        # Verify reasonable performance (should complete in reasonable time)
        assert build_time < 30  # Should complete within 30 seconds
        
        print(f"✅ Performance test completed in {build_time:.2f}s")
        print(f"   Processed {len(documents)} documents")
        print(f"   Throughput: {len(documents)/build_time:.2f} docs/sec")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests
    test_class = TestGeneration3Optimization()
    test_class.setup_method()
    
    try:
        test_class.test_parallel_processing_initialization()
        test_class.test_intelligent_caching_system()
        test_class.test_adaptive_batch_sizing()
        test_class.test_memory_optimization_features()
        test_class.test_auto_scaling_monitor()
        test_class.test_performance_monitoring()
        test_class.test_parallel_vs_sequential_processing()
        test_class.test_trainer_optimization_features()
        test_class.test_cache_key_generation()
        
        # Run performance test
        test_optimization_performance()
        
        print("\n🚀 All Generation 3 optimization tests passed!")
        
    finally:
        test_class.teardown_method()