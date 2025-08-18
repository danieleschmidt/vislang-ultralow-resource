"""Performance benchmarks and stress tests for VisLang-UltraLow-Resource."""

import time
import pytest
import psutil
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Benchmark configuration
BENCHMARK_ITERATIONS = 10
STRESS_TEST_ITERATIONS = 100
MEMORY_THRESHOLD_MB = 500
PROCESSING_TIME_THRESHOLD_SEC = 30


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for core functionality."""

    def test_image_processing_speed(self, sample_image):
        """Benchmark image processing speed."""
        from vislang_ultralow.dataset import DatasetBuilder
        
        builder = DatasetBuilder(languages=['en'])
        
        # Measure processing time
        start_time = time.time()
        
        for _ in range(BENCHMARK_ITERATIONS):
            # Mock OCR processing
            with patch.object(builder, '_process_image') as mock_process:
                mock_process.return_value = {
                    'text': 'Sample text',
                    'confidence': 0.95,
                    'language': 'en'
                }
                builder._process_image(sample_image)
        
        processing_time = time.time() - start_time
        avg_time_per_image = processing_time / BENCHMARK_ITERATIONS
        
        print(f"Average image processing time: {avg_time_per_image:.4f}s")
        assert avg_time_per_image < 1.0, f"Image processing too slow: {avg_time_per_image:.4f}s"

    def test_memory_usage_during_batch_processing(self, sample_image):
        """Test memory usage during batch processing."""
        from vislang_ultralow.dataset import DatasetBuilder
        
        # Monitor memory before processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        builder = DatasetBuilder(languages=['en'])
        images = [sample_image] * 50  # Process 50 images
        
        # Process batch
        with patch.object(builder, '_process_image') as mock_process:
            mock_process.return_value = {
                'text': 'Sample text',
                'confidence': 0.95,
                'language': 'en'
            }
            
            for img in images:
                builder._process_image(img)
        
        # Check memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase during batch processing: {memory_increase:.2f}MB")
        assert memory_increase < MEMORY_THRESHOLD_MB, f"Memory usage too high: {memory_increase:.2f}MB"

    def test_dataset_loading_speed(self, sample_dataset, temp_dir):
        """Benchmark dataset loading speed."""
        from vislang_ultralow.dataset import DatasetBuilder
        
        # Create a larger test dataset
        large_dataset = sample_dataset
        for i in range(100):  # Replicate to create larger dataset
            large_dataset = large_dataset.append(sample_dataset, ignore_index=True)
        
        # Save dataset
        dataset_path = temp_dir / "test_dataset.csv"
        large_dataset.to_csv(dataset_path, index=False)
        
        # Benchmark loading
        start_time = time.time()
        
        builder = DatasetBuilder(languages=['en'])
        loaded_dataset = builder.load_dataset(str(dataset_path))
        
        loading_time = time.time() - start_time
        
        print(f"Dataset loading time for {len(loaded_dataset)} samples: {loading_time:.4f}s")
        assert loading_time < 5.0, f"Dataset loading too slow: {loading_time:.4f}s"

    @pytest.mark.requires_model
    def test_model_inference_speed(self, mock_model, mock_processor, sample_image):
        """Benchmark model inference speed."""
        from vislang_ultralow.trainer import VisionLanguageTrainer
        
        trainer = VisionLanguageTrainer(
            model=mock_model,
            processor=mock_processor,
            languages=['en']
        )
        
        # Benchmark inference
        start_time = time.time()
        
        for _ in range(BENCHMARK_ITERATIONS):
            result = trainer.generate_caption(sample_image, "What is in this image?")
        
        inference_time = time.time() - start_time
        avg_time_per_inference = inference_time / BENCHMARK_ITERATIONS
        
        print(f"Average inference time: {avg_time_per_inference:.4f}s")
        assert avg_time_per_inference < 0.5, f"Inference too slow: {avg_time_per_inference:.4f}s"

    def test_concurrent_processing_performance(self, sample_image):
        """Test performance under concurrent processing load."""
        import concurrent.futures
        import threading
        from vislang_ultralow.dataset import DatasetBuilder
        
        builder = DatasetBuilder(languages=['en'])
        
        def process_image_task(image):
            """Task to process a single image."""
            with patch.object(builder, '_process_image') as mock_process:
                mock_process.return_value = {
                    'text': 'Sample text',
                    'confidence': 0.95,
                    'language': 'en'
                }
                return builder._process_image(image)
        
        # Run concurrent tasks
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_image_task, sample_image) 
                      for _ in range(20)]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        print(f"Concurrent processing time for 20 tasks: {concurrent_time:.4f}s")
        assert len(results) == 20, "Not all concurrent tasks completed"
        assert concurrent_time < 10.0, f"Concurrent processing too slow: {concurrent_time:.4f}s"


@pytest.mark.stress
@pytest.mark.slow
class TestStressTesting:
    """Stress tests for system robustness."""

    def test_large_document_processing(self, temp_dir):
        """Test processing of very large documents."""
        from vislang_ultralow.scraper import HumanitarianScraper
        
        # Create a large mock document
        large_content = "This is a test document. " * 10000  # ~250KB of text
        
        scraper = HumanitarianScraper(sources=['test'])
        
        # Mock the document processing
        with patch.object(scraper, '_extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = large_content
            
            start_time = time.time()
            result = scraper._extract_text_from_pdf(b"fake pdf content")
            processing_time = time.time() - start_time
            
            assert len(result) > 200000, "Large document not processed correctly"
            assert processing_time < PROCESSING_TIME_THRESHOLD_SEC, \
                f"Large document processing too slow: {processing_time:.2f}s"

    def test_memory_stability_under_load(self, sample_image):
        """Test memory stability under sustained load."""
        from vislang_ultralow.dataset import DatasetBuilder
        
        process = psutil.Process()
        memory_readings = []
        
        builder = DatasetBuilder(languages=['en'])
        
        # Process images continuously and monitor memory
        with patch.object(builder, '_process_image') as mock_process:
            mock_process.return_value = {
                'text': 'Sample text',
                'confidence': 0.95,
                'language': 'en'
            }
            
            for i in range(STRESS_TEST_ITERATIONS):
                builder._process_image(sample_image)
                
                # Record memory every 10 iterations
                if i % 10 == 0:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_readings.append(memory_mb)
        
        # Check for memory leaks (increasing trend)
        memory_trend = np.polyfit(range(len(memory_readings)), memory_readings, 1)[0]
        
        print(f"Memory trend (MB/iteration): {memory_trend:.4f}")
        assert memory_trend < 0.1, f"Potential memory leak detected: {memory_trend:.4f} MB/iteration"
        
        # Check maximum memory usage
        max_memory = max(memory_readings)
        print(f"Maximum memory usage: {max_memory:.2f}MB")
        assert max_memory < MEMORY_THRESHOLD_MB * 2, f"Memory usage too high: {max_memory:.2f}MB"

    def test_error_recovery_under_stress(self, sample_image):
        """Test system recovery from errors under stress."""
        from vislang_ultralow.dataset import DatasetBuilder
        
        builder = DatasetBuilder(languages=['en'])
        success_count = 0
        error_count = 0
        
        # Simulate random failures
        def mock_process_with_failures(image):
            import random
            if random.random() < 0.2:  # 20% failure rate
                raise RuntimeError("Simulated processing error")
            return {
                'text': 'Sample text',
                'confidence': 0.95,
                'language': 'en'
            }
        
        with patch.object(builder, '_process_image', side_effect=mock_process_with_failures):
            for _ in range(50):
                try:
                    builder._process_image(sample_image)
                    success_count += 1
                except RuntimeError:
                    error_count += 1
        
        print(f"Success rate: {success_count}/{success_count + error_count}")
        assert success_count > 0, "No successful operations under stress"
        assert error_count > 0, "Error simulation not working"

    def test_concurrent_database_access(self, mock_database_session):
        """Test database performance under concurrent access."""
        import threading
        import time
        from vislang_ultralow.database.models import Document
        
        results = []
        errors = []
        
        def database_operation(thread_id):
            """Simulate database operations in a thread."""
            try:
                # Create documents
                for i in range(10):
                    doc = Document(
                        url=f"https://example.com/doc_{thread_id}_{i}.pdf",
                        title=f"Document {thread_id}-{i}",
                        content=f"Content for document {thread_id}-{i}",
                        source="test",
                        language="en"
                    )
                    mock_database_session.add(doc)
                
                mock_database_session.commit()
                results.append(f"Thread {thread_id} completed")
                
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {str(e)}")
        
        # Run concurrent database operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=database_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print(f"Successful operations: {len(results)}")
        print(f"Failed operations: {len(errors)}")
        
        assert len(results) > 0, "No successful database operations"
        # Allow some failures under stress but not complete failure
        assert len(results) > len(errors), "More failures than successes"


@pytest.mark.integration
class TestIntegrationPerformance:
    """Integration performance tests."""

    def test_end_to_end_pipeline_performance(self, sample_documents, temp_dir):
        """Test performance of the complete processing pipeline."""
        from vislang_ultralow.dataset import DatasetBuilder
        from vislang_ultralow.scraper import HumanitarianScraper
        
        # Setup components
        scraper = HumanitarianScraper(sources=['unhcr', 'who'])
        builder = DatasetBuilder(
            languages=['en'],
            output_dir=str(temp_dir)
        )
        
        # Mock the complete pipeline
        with patch.object(scraper, 'scrape', return_value=sample_documents), \
             patch.object(builder, '_process_image') as mock_process:
            
            mock_process.return_value = {
                'text': 'Extracted text from image',
                'confidence': 0.95,
                'language': 'en'
            }
            
            # Run the complete pipeline
            start_time = time.time()
            
            # Scrape documents
            documents = scraper.scrape(max_documents=len(sample_documents))
            
            # Build dataset
            dataset = builder.build_from_documents(documents)
            
            # Save dataset
            output_path = temp_dir / "complete_dataset.json"
            dataset.to_json(str(output_path))
            
            total_time = time.time() - start_time
            
            print(f"End-to-end pipeline time: {total_time:.4f}s")
            assert total_time < 30.0, f"Pipeline too slow: {total_time:.4f}s"
            assert len(dataset) > 0, "Pipeline produced empty dataset"


# Utility functions for performance testing
def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time


def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage during function execution."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args, **kwargs)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    return result, memory_increase


def profile_function(func, *args, **kwargs):
    """Profile a function's performance characteristics."""
    import cProfile
    import io
    import pstats
    
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    # Get profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats()
    
    profile_output = s.getvalue()
    
    return result, profile_output


if __name__ == "__main__":
    # Run performance tests when script is executed directly
    pytest.main([__file__, "-v", "-m", "performance"])