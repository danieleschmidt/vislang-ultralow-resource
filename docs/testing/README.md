# Testing Guide

This document provides comprehensive guidance on testing the VisLang-UltraLow-Resource framework.

## Test Structure

Our testing strategy follows a three-tier approach:

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions
├── e2e/                 # End-to-end tests for complete workflows
├── fixtures/            # Test data and mock objects
├── conftest.py          # Shared pytest configuration and fixtures
└── performance/         # Performance and load tests
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **Scraper Tests** (`test_scraper.py`): HumanitarianScraper functionality
- **Dataset Tests** (`test_dataset.py`): DatasetBuilder components
- **Trainer Tests** (`test_trainer.py`): VisionLanguageTrainer methods
- **OCR Tests** (`test_ocr.py`): OCR engine implementations
- **Utils Tests** (`test_utils.py`): Utility functions

### Integration Tests (`tests/integration/`)

Test component interactions:

- **Pipeline Tests** (`test_full_pipeline.py`): End-to-end workflow testing
- **Data Flow Tests** (`test_data_flow.py`): Data transformation between components
- **Model Integration** (`test_model_integration.py`): Model loading and training

### End-to-End Tests (`tests/e2e/`)

Test complete user workflows:

- **CLI Tests** (`test_cli.py`): Command-line interface functionality
- **API Tests** (`test_api.py`): REST API endpoints
- **Deployment Tests** (`test_deployment.py`): Production deployment scenarios

## Running Tests

### Quick Start

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test categories
./scripts/run_tests.sh -t unit
./scripts/run_tests.sh -t integration
./scripts/run_tests.sh -t e2e
```

### Using pytest directly

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test files
pytest tests/unit/test_scraper.py

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
pytest -m "gpu"  # Run only GPU tests

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v
```

### Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.requires_model` - Tests requiring model downloads
- `@pytest.mark.requires_internet` - Tests requiring internet access
- `@pytest.mark.multilingual` - Multilingual functionality tests
- `@pytest.mark.performance` - Performance tests

## Test Configuration

### Environment Setup

Tests use the following environment variables:

```bash
# Required for testing
export APP_ENV=test
export LOG_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=-1  # Disable GPU for reproducibility

# Optional for specific tests
export TEST_DATA_PATH=/path/to/test/data
export SKIP_SLOW_TESTS=true
export MOCK_EXTERNAL_APIS=true
```

### Test Data

Test fixtures are provided in `tests/fixtures/`:

- **Sample Data** (`sample_data.py`): Generates realistic test data
- **Mock Objects** (`conftest.py`): Shared mock objects and fixtures
- **Test Images** (`fixtures/images/`): Sample images for vision tests

## Writing Tests

### Test Naming Convention

```python
def test_component_function_condition():
    """Test description following the pattern: test_[component]_[function]_[condition]"""
    pass

# Examples:
def test_scraper_fetch_document_success():
def test_dataset_builder_process_image_invalid_input():
def test_trainer_train_model_with_gpu():
```

### Using Fixtures

```python
def test_scraper_with_mock_data(mock_scraper, sample_dataset):
    """Example using fixtures from conftest.py"""
    result = mock_scraper.scrape()
    assert len(result) > 0

@pytest.fixture
def custom_test_data():
    """Custom fixture for specific test needs"""
    return {"test": "data"}
```

### Mocking External Dependencies

```python
from unittest.mock import patch, Mock

@patch('requests.get')
def test_api_call_success(mock_get):
    """Mock external API calls"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"data": "test"}
    
    # Test your code that makes the API call
    result = my_function_that_calls_api()
    assert result["data"] == "test"
```

### Testing Multilingual Content

```python
@pytest.mark.multilingual
def test_multilingual_processing():
    """Test with multiple languages"""
    test_texts = {
        "en": "English text",
        "sw": "Maandishi ya Kiswahili",
        "am": "የአማርኛ ጽሁፍ",
        "ar": "النص العربي"
    }
    
    for lang, text in test_texts.items():
        result = process_text(text, language=lang)
        assert result.language == lang
```

## Performance Testing

### Memory Usage

```python
import psutil
import os

def test_memory_usage():
    """Test memory consumption during processing"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run your memory-intensive operation
    large_dataset = create_large_dataset()
    process_dataset(large_dataset)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert reasonable memory usage (example: < 1GB increase)
    assert memory_increase < 1024 * 1024 * 1024
```

### Timing Tests

```python
import time

@pytest.mark.performance
def test_processing_speed():
    """Test processing speed requirements"""
    start_time = time.time()
    
    # Process 100 documents
    results = process_documents(test_documents[:100])
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Should process 100 documents in under 60 seconds
    assert processing_time < 60
    assert len(results) == 100
```

## Coverage Requirements

We maintain high test coverage standards:

- **Minimum Coverage**: 80% overall
- **Unit Tests**: 90% for core components
- **Integration Tests**: Cover all major workflows
- **Critical Path**: 100% coverage for data processing pipeline

### Viewing Coverage

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate terminal report
pytest --cov=src --cov-report=term-missing

# Check coverage percentage
coverage report --show-missing
```

## Continuous Integration

### GitHub Actions

Our CI pipeline runs tests on:

- Python 3.8, 3.9, 3.10, 3.11
- Ubuntu, macOS, Windows
- With and without GPU support

### Pre-commit Hooks

Automated testing is triggered by:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Debugging Tests

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes `src/`
2. **Missing Dependencies**: Install test dependencies with `pip install -e ".[dev]"`
3. **GPU Tests Failing**: Set `CUDA_VISIBLE_DEVICES=-1` for CPU-only testing
4. **Timeout Issues**: Increase timeout for slow tests or mark with `@pytest.mark.slow`

### Debug Mode

```bash
# Run single test with debugging
pytest tests/unit/test_scraper.py::TestHumanitarianScraper::test_fetch_document_success -v -s

# Drop into debugger on failure
pytest --pdb

# Keep temporary files for inspection
pytest --basetemp=/tmp/pytest-debug
```

### Logging

Enable detailed logging during tests:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_with_logging():
    """Test with debug logging enabled"""
    logger = logging.getLogger(__name__)
    logger.debug("Debug information")
    # Your test code
```

## Best Practices

### Test Organization

1. **One test class per component**
2. **Descriptive test names**
3. **Independent tests** (no test dependencies)
4. **Fast unit tests** (< 1 second each)
5. **Comprehensive edge case testing**

### Mock Strategy

1. **Mock external services** (APIs, file systems)
2. **Use fixtures for complex test data**
3. **Test behavior, not implementation**
4. **Mock at appropriate boundaries**

### Data Management

1. **Use deterministic test data**
2. **Clean up after tests**
3. **Isolate test environments**
4. **Version control test data**

## Troubleshooting

### Common Test Failures

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing PYTHONPATH | Set `PYTHONPATH=src` |
| `CUDA out of memory` | GPU tests on limited hardware | Use `CUDA_VISIBLE_DEVICES=-1` |
| `Timeout` | Slow network/computation | Mark as `@pytest.mark.slow` |
| `FileNotFoundError` | Missing test data | Run `python tests/fixtures/sample_data.py` |

### Getting Help

1. Check test logs for detailed error messages
2. Run individual tests with `-v -s` for verbose output
3. Use `--pdb` to debug test failures
4. Consult the [troubleshooting guide](../troubleshooting/common-issues.md)

---

For more information on specific testing scenarios, see:

- [Performance Testing](./performance-testing.md)
- [Multilingual Testing](./multilingual-testing.md)
- [GPU Testing](./gpu-testing.md)
- [API Testing](./api-testing.md)