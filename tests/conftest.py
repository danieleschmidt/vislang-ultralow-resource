"""Pytest configuration and shared fixtures for VisLang-UltraLow-Resource tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest
import pandas as pd
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from redis import Redis
from unittest.mock import patch


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample PIL Image for testing."""
    # Create a simple RGB image with some text-like pattern
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some structure to make it look more like a document
    img_array[50:70, 20:200] = 255  # White background for text area
    img_array[100:120, 20:200] = 255  # Another text line
    img_array[150:170, 20:200] = 255  # Third text line
    
    return Image.fromarray(img_array)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Create sample PDF content for testing."""
    # Minimal PDF content (this is a very basic PDF structure)
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test document) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000281 00000 n 
0000000377 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
456
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_ocr_result() -> Dict[str, Any]:
    """Create a sample OCR result for testing."""
    return {
        "text": "Sample text extracted from image",
        "confidence": 0.95,
        "bounding_boxes": [
            {"text": "Sample", "bbox": [10, 10, 60, 30], "confidence": 0.98},
            {"text": "text", "bbox": [65, 10, 95, 30], "confidence": 0.97},
            {"text": "extracted", "bbox": [100, 10, 170, 30], "confidence": 0.93},
            {"text": "from", "bbox": [175, 10, 205, 30], "confidence": 0.96},
            {"text": "image", "bbox": [210, 10, 250, 30], "confidence": 0.94},
        ],
        "language": "en",
    }


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    return pd.DataFrame({
        "image_path": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        "text": ["Sample text 1", "Sample text 2"],
        "language": ["en", "sw"],
        "source": ["unhcr", "who"],
        "quality_score": [0.95, 0.87],
        "metadata": [
            {"title": "Document 1", "date": "2023-01-01"},
            {"title": "Document 2", "date": "2023-01-02"},
        ],
    })


@pytest.fixture
def mock_model():
    """Create a mock vision-language model for testing."""
    model = Mock()
    model.config = Mock()
    model.config.vision_config = Mock()
    model.config.text_config = Mock()
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)
    return model


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    processor = Mock()
    processor.tokenizer = Mock()
    processor.image_processor = Mock()
    processor.decode = Mock(return_value="Generated text")
    processor.batch_decode = Mock(return_value=["Generated text"])
    return processor


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="Decoded text")
    tokenizer.batch_decode = Mock(return_value=["Decoded text"])
    tokenizer.vocab_size = 50000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer


@pytest.fixture
def mock_scraper():
    """Create a mock humanitarian scraper for testing."""
    scraper = Mock()
    scraper.sources = ["unhcr", "who", "unicef"]
    scraper.scrape = Mock(return_value=[
        {
            "url": "https://example.com/doc1.pdf",
            "title": "Sample Document 1",
            "content": b"Sample PDF content",
            "metadata": {"date": "2023-01-01", "source": "unhcr"},
        }
    ])
    return scraper


@pytest.fixture
def mock_ocr_engine():
    """Create a mock OCR engine for testing."""
    engine = Mock()
    engine.extract_text = Mock(return_value={
        "text": "Extracted text",
        "confidence": 0.95,
        "language": "en",
    })
    engine.supported_languages = ["en", "ar", "hi", "sw", "am"]
    return engine


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        "data": {
            "sources": ["unhcr", "who"],
            "languages": ["en", "sw", "am"],
            "output_format": "hf_dataset",
            "min_quality_score": 0.8,
        },
        "ocr": {
            "engines": ["tesseract", "easyocr"],
            "confidence_threshold": 0.7,
            "language_threshold": 0.8,
        },
        "training": {
            "model_name": "facebook/mblip-mt0-xl",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "warmup_steps": 100,
        },
        "quality": {
            "enable_validation": True,
            "validation_sample_rate": 0.1,
            "duplicate_threshold": 0.95,
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Setup test environment with temporary directories and mock environment variables."""
    # Set test environment variables
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("STORAGE_PATH", str(temp_dir))
    monkeypatch.setenv("HF_CACHE_DIR", str(temp_dir / "hf_cache"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")  # Disable CUDA for tests
    
    # Create necessary test directories
    (temp_dir / "data").mkdir(exist_ok=True)
    (temp_dir / "models").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    (temp_dir / "hf_cache").mkdir(exist_ok=True)


@pytest.fixture
def disable_gpu(monkeypatch):
    """Disable GPU usage for tests to ensure reproducibility."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    
    # Mock torch.cuda methods
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require downloading models"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "slow" in item.keywords or "e2e" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark e2e tests
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)


# Session-scoped fixtures for expensive operations
@pytest.fixture(scope="session")
def real_model_for_integration():
    """Load a real model for integration tests (only when needed)."""
    try:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-BlipModel")
        processor = AutoProcessor.from_pretrained("hf-internal-testing/tiny-random-BlipModel")
        return {"model": model, "processor": processor}
    except Exception as e:
        pytest.skip(f"Could not load real model for integration tests: {e}")


@pytest.fixture
def mock_database_session():
    """Create a mock database session for testing."""
    from vislang_ultralow.database.models import Base
    
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture 
def mock_redis():
    """Create a mock Redis client for testing."""
    redis_mock = Mock(spec=Redis)
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.expire.return_value = True
    redis_mock.ttl.return_value = 3600
    redis_mock.keys.return_value = []
    redis_mock.mget.return_value = []
    redis_mock.mset.return_value = True
    redis_mock.incrby.return_value = 1
    redis_mock.info.return_value = {
        'used_memory_human': '1M',
        'connected_clients': 1,
        'total_commands_processed': 100,
        'keyspace_hits': 50,
        'keyspace_misses': 50
    }
    return redis_mock


@pytest.fixture
def sample_documents():
    """Create sample document data for testing."""
    return [
        {
            "url": "https://example.unhcr.org/report1.pdf",
            "title": "UNHCR Global Report 2024",
            "source": "unhcr",
            "language": "en",
            "content": "This is a comprehensive report on global refugee situations.",
            "content_type": "pdf",
            "word_count": 150,
            "quality_score": 0.92,
            "images": [
                {
                    "src": "https://example.com/chart1.png",
                    "alt": "Refugee statistics chart",
                    "width": 800,
                    "height": 600,
                    "page": 1
                }
            ]
        },
        {
            "url": "https://example.who.int/health-report.pdf",
            "title": "WHO Health Emergency Report",
            "source": "who",
            "language": "en",
            "content": "Emergency health situation analysis and recommendations.",
            "content_type": "pdf",
            "word_count": 200,
            "quality_score": 0.88,
            "images": [
                {
                    "src": "https://example.com/map1.png",
                    "alt": "Health emergency map",
                    "width": 1000,
                    "height": 700,
                    "page": 2
                }
            ]
        }
    ]


@pytest.fixture
def sample_training_config():
    """Create sample training configuration."""
    return {
        'model_name': 'facebook/mblip-mt0-xl',
        'learning_rate': 5e-5,
        'batch_size': 8,
        'num_epochs': 3,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'gradient_checkpointing': True,
        'save_steps': 500,
        'eval_steps': 250,
        'logging_steps': 50,
        'early_stopping_patience': 3
    }


# Helper functions for tests
def assert_valid_dataset(dataset):
    """Assert that a dataset has the expected structure."""
    required_columns = ["image_path", "text", "language", "quality_score"]
    for col in required_columns:
        assert col in dataset.columns, f"Missing required column: {col}"
    
    assert len(dataset) > 0, "Dataset should not be empty"
    assert dataset["quality_score"].min() >= 0, "Quality scores should be non-negative"
    assert dataset["quality_score"].max() <= 1, "Quality scores should be <= 1"


def assert_valid_ocr_result(result):
    """Assert that an OCR result has the expected structure."""
    required_keys = ["text", "confidence", "language"]
    for key in required_keys:
        assert key in result, f"Missing required key in OCR result: {key}"
    
    assert isinstance(result["text"], str), "OCR text should be a string"
    assert 0 <= result["confidence"] <= 1, "OCR confidence should be between 0 and 1"
    assert isinstance(result["language"], str), "OCR language should be a string"