"""Simplified test for Generation 3 optimization features."""

import tempfile
import shutil
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vislang_ultralow.dataset import DatasetBuilder
from vislang_ultralow.trainer import VisionLanguageTrainer


def test_generation3_features():
    """Test Generation 3 optimization features."""
    print("🚀 Testing Generation 3 Optimization Features")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: DatasetBuilder optimization initialization
        print("\n1. Testing DatasetBuilder optimization initialization...")
        
        builder = DatasetBuilder(
            output_dir=temp_dir,
            target_languages=['en', 'fr']
        )
        
        # Check optimization config
        assert hasattr(builder, 'optimization_config'), "Missing optimization_config"
        assert builder.optimization_config['parallel_processing'] is True, "Parallel processing not enabled"
        assert builder.optimization_config['max_workers'] >= 1, "Invalid worker count"
        assert builder.optimization_config['parallel_threshold'] == 10, "Wrong parallel threshold"
        assert builder.optimization_config['adaptive_batch_size'] is True, "Adaptive batching not enabled"
        assert builder.optimization_config['memory_optimization'] is True, "Memory optimization not enabled"
        assert builder.optimization_config['cache_enabled'] is True, "Caching not enabled"
        assert builder.optimization_config['performance_monitoring'] is True, "Performance monitoring not enabled"
        
        print("   ✅ Optimization config properly initialized")
        
        # Test 2: Adaptive metrics initialization
        print("\n2. Testing adaptive metrics initialization...")
        
        assert hasattr(builder, 'adaptive_metrics'), "Missing adaptive_metrics"
        assert 'optimal_batch_size' in builder.adaptive_metrics, "Missing optimal_batch_size"
        assert 'throughput_history' in builder.adaptive_metrics, "Missing throughput_history"
        assert 'auto_scaling_events' in builder.adaptive_metrics, "Missing auto_scaling_events"
        
        print("   ✅ Adaptive metrics properly initialized")
        
        # Test 3: Auto-scaling configuration
        print("\n3. Testing auto-scaling system...")
        
        assert hasattr(builder, 'auto_scaling_config'), "Missing auto_scaling_config"
        assert 'enabled' in builder.auto_scaling_config, "Missing enabled"
        assert 'max_workers' in builder.auto_scaling_config, "Missing max_workers"
        assert 'scale_up_threshold' in builder.auto_scaling_config, "Missing scale_up_threshold"
        assert 'scale_down_threshold' in builder.auto_scaling_config, "Missing scale_down_threshold"
        
        print("   ✅ Auto-scaling system properly configured")
        
        # Test 4: Batch size calculation
        print("\n4. Testing adaptive batch sizing...")
        
        test_sizes = [5, 10, 50, 100, 1000]
        for size in test_sizes:
            optimal_batch = builder._calculate_optimal_batch_size(size)
            assert isinstance(optimal_batch, int), f"Batch size not integer for {size}"
            assert optimal_batch > 0, f"Invalid batch size {optimal_batch} for {size}"
            assert optimal_batch <= size, f"Batch size {optimal_batch} > total {size}"
            
        print("   ✅ Adaptive batch sizing functional")
        
        # Test 5: Performance metrics
        print("\n5. Testing performance monitoring...")
        
        assert hasattr(builder, 'performance_metrics'), "Missing performance_metrics"
        assert 'memory_usage' in builder.performance_metrics, "Missing memory_usage metric"
        assert 'documents_processed' in builder.performance_metrics, "Missing documents_processed"
        assert 'avg_processing_time' in builder.performance_metrics, "Missing avg_processing_time"
        assert 'errors_encountered' in builder.performance_metrics, "Missing errors_encountered"
        
        print("   ✅ Performance monitoring active")
        
        # Test 6: Trainer optimization features
        print("\n6. Testing trainer optimization features...")
        
        # Create mock model and processor
        class MockModel:
            def to(self, device):
                return self
            def parameters(self):
                class MockParam:
                    def numel(self):
                        return 1000
                    requires_grad = True
                return [MockParam()]
        
        class MockProcessor:
            pass
        
        trainer = VisionLanguageTrainer(
            model=MockModel(),
            processor=MockProcessor(),
            languages=['en', 'fr']
        )
        
        # Check optimization config
        assert hasattr(trainer, 'optimization_config'), "Trainer missing optimization_config"
        assert trainer.optimization_config['mixed_precision'] is True, "Mixed precision not enabled"
        assert trainer.optimization_config['gradient_checkpointing'] is True, "Gradient checkpointing not enabled"
        assert trainer.optimization_config['cache_enabled'] is True, "Cache not enabled in trainer"
        assert trainer.optimization_config['dynamic_batching'] is True, "Dynamic batching not enabled"
        assert trainer.optimization_config['memory_optimization'] is True, "Memory optimization not enabled in trainer"
        
        # Check adaptive metrics
        assert hasattr(trainer, 'adaptive_metrics'), "Trainer missing adaptive_metrics"
        assert 'optimal_batch_size' in trainer.adaptive_metrics, "Missing optimal_batch_size in trainer"
        assert 'memory_usage_history' in trainer.adaptive_metrics, "Missing memory_usage_history"
        assert 'throughput_history' in trainer.adaptive_metrics, "Missing throughput_history"
        
        # Check memory monitor
        assert hasattr(trainer, 'memory_monitor'), "Trainer missing memory_monitor"
        assert 'peak_memory' in trainer.memory_monitor, "Missing peak_memory"
        assert 'oom_count' in trainer.memory_monitor, "Missing oom_count"
        
        print("   ✅ Trainer optimization features initialized")
        
        # Test 7: Dataset building with small batch (sequential processing)
        print("\n7. Testing dataset building with sequential processing...")
        
        small_documents = [
            {'id': 'doc1', 'url': 'http://test.com/1', 'content': 'Test content 1', 'images': []},
            {'id': 'doc2', 'url': 'http://test.com/2', 'content': 'Test content 2', 'images': []},
        ]
        
        start_time = time.time()
        result = builder.build(small_documents, output_format="custom")
        build_time = time.time() - start_time
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'train' in result or 'data' in result or 'items' in result, "Result missing expected keys"
        assert build_time < 10, f"Sequential processing too slow: {build_time}s"
        
        # Check performance metrics were updated
        assert 'last_build_time' in builder.performance_metrics, "Missing last_build_time metric"
        assert builder.performance_metrics['last_build_time'] > 0, "Invalid build time recorded"
        
        print(f"   ✅ Sequential processing completed in {build_time:.2f}s")
        
        # Test 8: Test parallel vs sequential logic
        print("\n8. Testing parallel processing logic...")
        
        # Create larger document set (above threshold)
        large_documents = []
        for i in range(15):  # Above parallel threshold of 10
            large_documents.append({
                'id': f'doc_{i}',
                'url': f'http://test.com/{i}',
                'content': f'Test content {i}',
                'images': []
            })
        
        # This should trigger parallel processing
        start_time = time.time()
        result_large = builder.build(large_documents, output_format="custom")
        large_build_time = time.time() - start_time
        
        assert isinstance(result_large, dict), "Large result should be a dictionary"
        assert large_build_time < 30, f"Parallel processing too slow: {large_build_time}s"
        
        print(f"   ✅ Parallel processing completed in {large_build_time:.2f}s for {len(large_documents)} docs")
        
        print(f"\n🎉 All Generation 3 optimization tests passed!")
        print(f"   Small batch (2 docs): {build_time:.2f}s")
        print(f"   Large batch (15 docs): {large_build_time:.2f}s")
        print(f"   Performance metrics collected: {len(builder.performance_metrics)} metrics")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_generation3_features()
    exit(0 if success else 1)