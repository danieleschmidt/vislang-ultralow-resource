"""Utility functions and helpers for testing."""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import pandas as pd


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_test_image(
        width: int = 224,
        height: int = 224,
        text_regions: Optional[List[Dict]] = None
    ) -> Image.Image:
        """Create a test image with optional text regions."""
        # Create base image
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add text regions if specified
        if text_regions:
            for region in text_regions:
                x1, y1, x2, y2 = region.get('bbox', [50, 50, 150, 70])
                img_array[y1:y2, x1:x2] = 255  # White background for text
        else:
            # Default text-like regions
            img_array[50:70, 20:200] = 255
            img_array[100:120, 20:200] = 255
            img_array[150:170, 20:200] = 255
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def create_test_dataset(
        num_samples: int = 10,
        languages: List[str] = ['en', 'sw', 'am']
    ) -> pd.DataFrame:
        """Create a test dataset with specified characteristics."""
        data = []
        
        for i in range(num_samples):
            lang = languages[i % len(languages)]
            data.append({
                'image_path': f'/fake/path/image_{i}.jpg',
                'text': f'Sample text {i} in {lang}',
                'language': lang,
                'source': ['unhcr', 'who', 'unicef'][i % 3],
                'quality_score': 0.7 + (i % 3) * 0.1,
                'metadata': {
                    'title': f'Document {i}',
                    'date': f'2024-01-{(i % 28) + 1:02d}',
                    'word_count': 50 + i * 10,
                    'image_count': (i % 5) + 1
                }
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_test_documents(
        num_docs: int = 5,
        include_images: bool = True
    ) -> List[Dict[str, Any]]:
        """Create test document data."""
        documents = []
        
        for i in range(num_docs):
            doc = {
                'url': f'https://example.org/doc_{i}.pdf',
                'title': f'Test Document {i}',
                'source': ['unhcr', 'who', 'unicef', 'wfp'][i % 4],
                'language': ['en', 'es', 'fr', 'ar'][i % 4],
                'content': f'This is the content of test document {i}. ' * 20,
                'content_type': 'pdf',
                'word_count': 100 + i * 50,
                'quality_score': 0.8 + (i % 2) * 0.1,
                'date_published': f'2024-{(i % 12) + 1:02d}-15',
                'metadata': {
                    'author': f'Author {i}',
                    'pages': i + 1,
                    'file_size': f'{(i + 1) * 1024}KB'
                }
            }
            
            if include_images:
                doc['images'] = [
                    {
                        'src': f'https://example.org/image_{i}_{j}.png',
                        'alt': f'Chart {j} from document {i}',
                        'width': 800 + j * 100,
                        'height': 600 + j * 50,
                        'page': j + 1,
                        'type': ['chart', 'map', 'infographic'][j % 3]
                    }
                    for j in range((i % 3) + 1)
                ]
            
            documents.append(doc)
        
        return documents


class MockServerResponse:
    """Mock HTTP response for testing API interactions."""
    
    def __init__(self, json_data: Dict, status_code: int = 200, headers: Dict = None):
        self.json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TemporaryTestEnvironment:
    """Context manager for temporary test environments."""
    
    def __init__(self, create_dirs: List[str] = None):
        self.create_dirs = create_dirs or []
        self.temp_dir = None
        self.original_env = {}
    
    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create requested directories
        for dir_name in self.create_dirs:
            (self.temp_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class TestAssertions:
    """Custom assertion helpers for testing."""
    
    @staticmethod
    def assert_valid_dataset_structure(dataset: pd.DataFrame):
        """Assert dataset has valid structure for VisLang."""
        required_columns = ['image_path', 'text', 'language', 'quality_score']
        
        for col in required_columns:
            assert col in dataset.columns, f"Missing required column: {col}"
        
        assert len(dataset) > 0, "Dataset should not be empty"
        assert dataset['quality_score'].dtype in [np.float64, np.float32], \
            "Quality scores should be numeric"
        assert (dataset['quality_score'] >= 0).all(), \
            "Quality scores should be non-negative"
        assert (dataset['quality_score'] <= 1).all(), \
            "Quality scores should be <= 1"
    
    @staticmethod
    def assert_valid_ocr_result(result: Dict[str, Any]):
        """Assert OCR result has expected structure."""
        required_keys = ['text', 'confidence', 'language']
        
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        assert isinstance(result['text'], str), "OCR text should be string"
        assert isinstance(result['confidence'], (int, float)), \
            "OCR confidence should be numeric"
        assert 0 <= result['confidence'] <= 1, \
            "OCR confidence should be between 0 and 1"
        assert isinstance(result['language'], str), \
            "OCR language should be string"
    
    @staticmethod
    def assert_valid_model_output(output: Any, expected_type: str = "text"):
        """Assert model output is valid."""
        if expected_type == "text":
            assert isinstance(output, str), "Model output should be string"
            assert len(output.strip()) > 0, "Model output should not be empty"
        elif expected_type == "tensor":
            import torch
            assert isinstance(output, torch.Tensor), "Model output should be tensor"
            assert output.numel() > 0, "Model output tensor should not be empty"
    
    @staticmethod
    def assert_performance_within_limits(
        execution_time: float,
        max_time: float,
        operation_name: str = "operation"
    ):
        """Assert operation completed within time limits."""
        assert execution_time <= max_time, \
            f"{operation_name} took {execution_time:.4f}s (limit: {max_time}s)"
    
    @staticmethod
    def assert_memory_usage_reasonable(
        memory_increase: float,
        max_increase_mb: float,
        operation_name: str = "operation"
    ):
        """Assert memory usage is within reasonable limits."""
        assert memory_increase <= max_increase_mb, \
            f"{operation_name} used {memory_increase:.2f}MB (limit: {max_increase_mb}MB)"


class DatabaseTestMixin:
    """Mixin for database-related test functionality."""
    
    @staticmethod
    def create_test_database_session():
        """Create a test database session with in-memory SQLite."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from vislang_ultralow.database.models import Base
        
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()
    
    @staticmethod
    def populate_test_database(session, num_documents: int = 5):
        """Populate test database with sample data."""
        from vislang_ultralow.database.models import Document, Dataset, ProcessingJob
        
        # Add documents
        documents = []
        for i in range(num_documents):
            doc = Document(
                url=f"https://example.org/doc_{i}.pdf",
                title=f"Test Document {i}",
                content=f"Content for document {i}",
                source=['unhcr', 'who'][i % 2],
                language=['en', 'sw'][i % 2],
                quality_score=0.8 + (i % 2) * 0.1
            )
            documents.append(doc)
            session.add(doc)
        
        session.commit()
        return documents


class ModelTestMixin:
    """Mixin for model-related test functionality."""
    
    @staticmethod
    def create_mock_model():
        """Create a comprehensive mock model for testing."""
        model = Mock()
        model.config = Mock()
        model.config.vision_config = Mock()
        model.config.text_config = Mock()
        model.config.vocab_size = 50000
        
        # Mock generation methods
        import torch
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        model.forward = Mock(return_value=Mock(logits=torch.randn(1, 10, 50000)))
        model.eval = Mock(return_value=model)
        model.train = Mock(return_value=model)
        model.to = Mock(return_value=model)
        model.parameters = Mock(return_value=[torch.randn(100, 100, requires_grad=True)])
        
        return model
    
    @staticmethod
    def create_mock_processor():
        """Create a comprehensive mock processor for testing."""
        processor = Mock()
        
        # Mock tokenizer
        processor.tokenizer = Mock()
        processor.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        processor.tokenizer.decode = Mock(return_value="Generated text")
        processor.tokenizer.batch_decode = Mock(return_value=["Generated text"])
        processor.tokenizer.vocab_size = 50000
        processor.tokenizer.pad_token_id = 0
        processor.tokenizer.eos_token_id = 1
        
        # Mock image processor
        processor.image_processor = Mock()
        processor.image_processor.preprocess = Mock(return_value={
            'pixel_values': np.random.randn(1, 3, 224, 224)
        })
        
        # Mock main methods
        processor.decode = Mock(return_value="Decoded text")
        processor.batch_decode = Mock(return_value=["Decoded text"])
        processor.__call__ = Mock(return_value={
            'pixel_values': np.random.randn(1, 3, 224, 224),
            'input_ids': np.array([[1, 2, 3, 4, 5]])
        })
        
        return processor


# Test decorators
def requires_gpu(func):
    """Decorator to skip tests that require GPU if not available."""
    import torch
    import pytest
    
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return func(*args, **kwargs)
    
    return wrapper


def requires_internet(func):
    """Decorator to skip tests that require internet connection."""
    import socket
    import pytest
    
    def wrapper(*args, **kwargs):
        try:
            socket.create_connection(("8.8.8.8", 53), 1)
        except OSError:
            pytest.skip("Internet connection not available")
        return func(*args, **kwargs)
    
    return wrapper


def slow_test(func):
    """Decorator to mark tests as slow."""
    import pytest
    return pytest.mark.slow(func)


def integration_test(func):
    """Decorator to mark tests as integration tests."""
    import pytest
    return pytest.mark.integration(func)


# Utility functions
def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """Compare two datasets with tolerance for floating point values."""
    if df1.shape != df2.shape:
        return False
    
    if list(df1.columns) != list(df2.columns):
        return False
    
    for col in df1.columns:
        if df1[col].dtype in [np.float64, np.float32]:
            if not np.allclose(df1[col], df2[col], atol=tolerance, rtol=tolerance):
                return False
        else:
            if not df1[col].equals(df2[col]):
                return False
    
    return True


def create_test_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a test configuration with optional overrides."""
    config = {
        'data': {
            'sources': ['test'],
            'languages': ['en'],
            'output_format': 'json',
            'min_quality_score': 0.5
        },
        'ocr': {
            'engines': ['tesseract'],
            'confidence_threshold': 0.5,
            'language_threshold': 0.7
        },
        'training': {
            'model_name': 'hf-internal-testing/tiny-random-BlipModel',
            'batch_size': 2,
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'warmup_steps': 10
        },
        'quality': {
            'enable_validation': False,
            'validation_sample_rate': 0.1,
            'duplicate_threshold': 0.95
        }
    }
    
    if overrides:
        config.update(overrides)
    
    return config