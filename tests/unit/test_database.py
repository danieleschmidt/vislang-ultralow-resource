"""Tests for database models and repositories."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError

from vislang_ultralow.database.models import Document, Image, DatasetItem, TrainingRun
from vislang_ultralow.database.repositories import (
    DocumentRepository, ImageRepository, DatasetRepository, TrainingRepository
)


class TestDatabaseModels:
    """Test database model classes."""
    
    def test_document_model_creation(self, mock_database_session):
        """Test Document model creation and validation."""
        doc = Document(
            url="https://example.com/test.pdf",
            title="Test Document",
            source="unhcr",
            language="en",
            content="Sample content"
        )
        
        mock_database_session.add(doc)
        mock_database_session.commit()
        
        assert doc.id is not None
        assert doc.url == "https://example.com/test.pdf"
        assert doc.source == "unhcr"
        assert doc.quality_score == 0.0  # Default value
    
    def test_document_source_validation(self, mock_database_session):
        """Test document source validation."""
        doc = Document(
            url="https://example.com/test.pdf",
            title="Test Document",
            source="invalid_source",  # Invalid source
            language="en"
        )
        
        with pytest.raises(ValueError, match="Invalid source"):
            mock_database_session.add(doc)
            mock_database_session.commit()
    
    def test_image_model_creation(self, mock_database_session):
        """Test Image model creation."""
        # First create a document
        doc = Document(
            url="https://example.com/test.pdf",
            title="Test Document", 
            source="unhcr",
            language="en"
        )
        mock_database_session.add(doc)
        mock_database_session.flush()
        
        # Create image linked to document
        image = Image(
            document_id=doc.id,
            src="https://example.com/image.jpg",
            alt_text="Test image",
            width=800,
            height=600,
            image_type="chart"
        )
        
        mock_database_session.add(image)
        mock_database_session.commit()
        
        assert image.id is not None
        assert image.document_id == doc.id
        assert image.image_type == "chart"
    
    def test_dataset_item_model_creation(self, mock_database_session):
        """Test DatasetItem model creation."""
        # Create document and image first
        doc = Document(
            url="https://example.com/test.pdf",
            title="Test Document",
            source="unhcr", 
            language="en"
        )
        mock_database_session.add(doc)
        mock_database_session.flush()
        
        image = Image(
            document_id=doc.id,
            src="https://example.com/image.jpg",
            image_type="chart"
        )
        mock_database_session.add(image)
        mock_database_session.flush()
        
        # Create dataset item
        item = DatasetItem(
            document_id=doc.id,
            image_id=image.id,
            instruction="What does this chart show?",
            response="This chart shows humanitarian statistics.",
            quality_score=0.85,
            target_language="sw"
        )
        
        mock_database_session.add(item)
        mock_database_session.commit()
        
        assert item.id is not None
        assert item.quality_score == 0.85
        assert item.split == "train"  # Default value
    
    def test_training_run_model_creation(self, mock_database_session):
        """Test TrainingRun model creation."""
        run = TrainingRun(
            name="Test Training Run",
            base_model="facebook/mblip-mt0-xl",
            languages=["en", "sw", "am"],
            dataset_version="v1.0"
        )
        
        mock_database_session.add(run)
        mock_database_session.commit()
        
        assert run.id is not None
        assert run.status == "pending"  # Default value
        assert run.languages == ["en", "sw", "am"]
    
    def test_training_run_progress_calculation(self, mock_database_session):
        """Test training run progress calculation."""
        run = TrainingRun(
            name="Test Run",
            base_model="test-model",
            languages=["en"],
            dataset_version="v1.0",
            total_steps=1000,
            current_step=250
        )
        
        assert run.progress_percentage == 25.0
        
        # Test completed run
        run.current_step = 1000
        assert run.progress_percentage == 100.0
        
        # Test no steps
        run.total_steps = 0
        assert run.progress_percentage == 0.0


class TestDocumentRepository:
    """Test DocumentRepository methods."""
    
    def test_create_document(self, mock_database_session):
        """Test document creation via repository."""
        repo = DocumentRepository(mock_database_session)
        
        doc = repo.create(
            url="https://example.com/test.pdf",
            title="Test Document",
            source="unhcr",
            language="en",
            content="Sample content"
        )
        
        assert doc.url == "https://example.com/test.pdf"
        assert doc.source == "unhcr"
    
    def test_get_by_url(self, mock_database_session):
        """Test getting document by URL."""
        repo = DocumentRepository(mock_database_session)
        
        # Mock query result
        mock_database_session.query.return_value.filter.return_value.first.return_value = Mock(
            url="https://example.com/test.pdf",
            title="Test Document"
        )
        
        doc = repo.get_by_url("https://example.com/test.pdf")
        assert doc.url == "https://example.com/test.pdf"
    
    def test_get_by_source(self, mock_database_session):
        """Test getting documents by source."""
        repo = DocumentRepository(mock_database_session)
        
        # Mock query result
        mock_database_session.query.return_value.filter.return_value.limit.return_value.all.return_value = [
            Mock(source="unhcr", title="Doc 1"),
            Mock(source="unhcr", title="Doc 2")
        ]
        
        docs = repo.get_by_source("unhcr")
        assert len(docs) == 2
        assert all(doc.source == "unhcr" for doc in docs)
    
    def test_get_statistics(self, mock_database_session):
        """Test getting document statistics."""
        repo = DocumentRepository(mock_database_session)
        
        # Mock various query results for statistics
        mock_database_session.query.return_value.count.return_value = 100
        mock_database_session.query.return_value.group_by.return_value.all.return_value = [
            ("unhcr", 50), ("who", 30), ("unicef", 20)
        ]
        
        stats = repo.get_statistics()
        
        assert stats["total_documents"] == 100
        assert "by_source" in stats
        assert "quality_distribution" in stats


class TestImageRepository:
    """Test ImageRepository methods."""
    
    def test_get_by_document(self, mock_database_session):
        """Test getting images by document ID."""
        repo = ImageRepository(mock_database_session)
        
        mock_database_session.query.return_value.filter.return_value.all.return_value = [
            Mock(document_id="doc123", image_type="chart"),
            Mock(document_id="doc123", image_type="map")
        ]
        
        images = repo.get_by_document("doc123")
        assert len(images) == 2
        assert all(img.document_id == "doc123" for img in images)
    
    def test_update_ocr_results(self, mock_database_session):
        """Test updating OCR results for an image."""
        repo = ImageRepository(mock_database_session)
        
        # Mock existing image
        mock_image = Mock()
        mock_database_session.query.return_value.filter.return_value.first.return_value = mock_image
        
        updated_image = repo.update_ocr_results(
            "image123",
            "Extracted text",
            0.95,
            ["tesseract", "easyocr"],
            {"tesseract": {"text": "Text", "confidence": 0.9}}
        )
        
        assert updated_image.ocr_text == "Extracted text"
        assert updated_image.ocr_confidence == 0.95
        assert updated_image.ocr_engines_used == ["tesseract", "easyocr"]


class TestDatasetRepository:
    """Test DatasetRepository methods."""
    
    def test_get_by_split(self, mock_database_session):
        """Test getting dataset items by split."""
        repo = DatasetRepository(mock_database_session)
        
        mock_database_session.query.return_value.filter.return_value.limit.return_value.all.return_value = [
            Mock(split="train", quality_score=0.9),
            Mock(split="train", quality_score=0.8)
        ]
        
        items = repo.get_by_split("train")
        assert len(items) == 2
        assert all(item.split == "train" for item in items)
    
    def test_get_for_training(self, mock_database_session):
        """Test getting dataset items formatted for training."""
        repo = DatasetRepository(mock_database_session)
        
        # Mock base query
        mock_query = Mock()
        mock_database_session.query.return_value.filter.return_value = mock_query
        
        # Mock filtered queries for each split
        mock_query.filter.return_value.all.side_effect = [
            [Mock(split="train")],  # train split
            [Mock(split="validation")],  # validation split  
            [Mock(split="test")]  # test split
        ]
        
        result = repo.get_for_training(["en", "sw"], min_quality=0.8)
        
        assert "train" in result
        assert "validation" in result
        assert "test" in result
    
    def test_create_dataset_split(self, mock_database_session):
        """Test creating dataset splits."""
        repo = DatasetRepository(mock_database_session)
        
        # Create mock items
        items = [Mock(split=None) for _ in range(10)]
        
        repo.create_dataset_split(items, 0.8, 0.1, 0.1)
        
        # Check that splits were assigned
        train_count = sum(1 for item in items if item.split == "train")
        val_count = sum(1 for item in items if item.split == "validation") 
        test_count = sum(1 for item in items if item.split == "test")
        
        assert train_count == 8
        assert val_count == 1
        assert test_count == 1


class TestTrainingRepository:
    """Test TrainingRepository methods."""
    
    def test_get_by_status(self, mock_database_session):
        """Test getting training runs by status."""
        repo = TrainingRepository(mock_database_session)
        
        mock_database_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            Mock(status="running", name="Run 1"),
            Mock(status="running", name="Run 2")
        ]
        
        runs = repo.get_by_status("running")
        assert len(runs) == 2
        assert all(run.status == "running" for run in runs)
    
    def test_update_progress(self, mock_database_session):
        """Test updating training progress."""
        repo = TrainingRepository(mock_database_session)
        
        # Mock existing run
        mock_run = Mock()
        mock_run.metrics_history = {}
        mock_database_session.query.return_value.filter.return_value.first.return_value = mock_run
        
        metrics = {"train_loss": 0.5, "eval_loss": 0.6}
        updated_run = repo.update_progress("run123", 100, metrics)
        
        assert updated_run.current_step == 100
        assert updated_run.train_loss == 0.5
        assert updated_run.eval_loss == 0.6
    
    def test_complete_training(self, mock_database_session):
        """Test marking training as completed."""
        repo = TrainingRepository(mock_database_session)
        
        # Mock existing run
        mock_run = Mock()
        mock_run.started_at = datetime.now() - timedelta(hours=2)
        mock_database_session.query.return_value.filter.return_value.first.return_value = mock_run
        
        final_metrics = {"final_loss": 0.3, "final_bleu": 0.7}
        completed_run = repo.complete_training(
            "run123", final_metrics, "/path/to/model"
        )
        
        assert completed_run.status == "completed"
        assert completed_run.model_path == "/path/to/model"
        assert completed_run.final_loss == 0.3
        assert completed_run.duration_seconds > 0


@pytest.mark.integration  
def test_repository_integration(mock_database_session):
    """Integration test for repository operations."""
    doc_repo = DocumentRepository(mock_database_session)
    img_repo = ImageRepository(mock_database_session)
    dataset_repo = DatasetRepository(mock_database_session)
    
    # Create document
    doc = doc_repo.create(
        url="https://example.com/integration.pdf",
        title="Integration Test Doc",
        source="unhcr",
        language="en"
    )
    
    # Create image
    image = img_repo.create(
        document_id=doc.id,
        src="https://example.com/chart.jpg",
        image_type="chart"
    )
    
    # Create dataset item
    item = dataset_repo.create(
        document_id=doc.id,
        image_id=image.id,
        instruction="What does this show?",
        response="Statistical data",
        quality_score=0.9,
        target_language="sw"
    )
    
    # Verify relationships work
    assert item.document_id == doc.id
    assert item.image_id == image.id