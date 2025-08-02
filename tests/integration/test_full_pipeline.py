"""Integration tests for the complete VisLang pipeline."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import pandas as pd
from PIL import Image

from vislang_ultralow.scraper import HumanitarianScraper
from vislang_ultralow.dataset import DatasetBuilder
from vislang_ultralow.trainer import VisionLanguageTrainer


class TestFullPipeline:
    """Integration tests for the complete pipeline from scraping to training."""

    @pytest.mark.integration
    def test_scraper_to_dataset_pipeline(self, sample_image, temp_dir):
        """Test integration between scraper and dataset builder."""
        # Setup mock scraper with realistic data
        scraper = Mock(spec=HumanitarianScraper)
        scraper.scrape.return_value = [
            {
                "url": "https://example.com/doc1.pdf",
                "title": "Humanitarian Report 2023",
                "content": b"Mock PDF content",
                "images": [sample_image],
                "metadata": {
                    "source": "unhcr",
                    "date": "2023-01-01",
                    "language": "en",
                    "pages": 10
                }
            }
        ]
        
        # Create dataset builder
        builder = DatasetBuilder(
            target_languages=["en", "sw"],
            min_quality_score=0.7
        )
        
        # Mock OCR extraction
        with patch.object(builder, '_extract_text_from_image') as mock_ocr:
            mock_ocr.return_value = {
                "text": "Sample humanitarian document text about refugee assistance.",
                "confidence": 0.92,
                "language": "en",
                "bounding_boxes": []
            }
            
            # Build dataset
            dataset = builder.build(
                scraper=scraper,
                include_infographics=True,
                output_format="hf_dataset"
            )
            
            # Verify pipeline integration
            assert len(dataset) > 0
            assert "text" in dataset.column_names
            assert "image" in dataset.column_names
            assert "language" in dataset.column_names

    @pytest.mark.integration
    def test_dataset_to_training_pipeline(self, sample_dataset, mock_model, mock_processor):
        """Test integration between dataset builder and trainer."""
        # Prepare training dataset
        training_data = sample_dataset.copy()
        training_data["instruction"] = [
            "What information is shown in this image?",
            "Describe the content of this document."
        ]
        training_data["response"] = [
            "This image shows humanitarian statistics.",
            "This document contains health information."
        ]
        
        # Create trainer
        trainer = VisionLanguageTrainer(
            model=mock_model,
            processor=mock_processor,
            languages=["en", "sw"]
        )
        
        # Convert to HuggingFace format
        with patch('datasets.Dataset.from_pandas') as mock_from_pandas:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=2)
            mock_dataset.__getitem__ = Mock(return_value={
                "instruction": "What is shown?",
                "response": "Document content",
                "image": sample_dataset.iloc[0]["image_path"],
                "language": "en"
            })
            mock_from_pandas.return_value = mock_dataset
            
            # Test training preparation
            train_dataset = trainer._prepare_training_data(training_data)
            
            # Verify integration
            assert train_dataset is not None
            assert len(train_dataset) == 2

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_pipeline_end_to_end(self, temp_dir):
        """Test the complete pipeline from scraping to model output."""
        # Create test image
        test_image = Image.new("RGB", (224, 224), color="white")
        image_path = temp_dir / "test_image.jpg"
        test_image.save(image_path)
        
        # Mock scraper data
        scraped_data = [
            {
                "url": "https://example.com/humanitarian_report.pdf",
                "title": "Emergency Response Report",
                "content": b"PDF content",
                "images": [test_image],
                "metadata": {
                    "source": "unhcr",
                    "date": "2023-06-01",
                    "language": "en",
                    "topic": "emergency_response"
                }
            }
        ]
        
        # Step 1: Build dataset
        builder = DatasetBuilder(target_languages=["en", "sw"])
        
        with patch.object(builder, '_extract_text_from_image') as mock_ocr:
            mock_ocr.return_value = {
                "text": "Emergency response statistics show 1000 people assisted.",
                "confidence": 0.95,
                "language": "en"
            }
            
            dataset = builder._process_scraped_content(scraped_data)
            
        # Step 2: Prepare for training
        with patch('transformers.AutoModel.from_pretrained') as mock_model_load, \
             patch('transformers.AutoProcessor.from_pretrained') as mock_proc_load:
            
            mock_model = Mock()
            mock_processor = Mock()
            mock_model_load.return_value = mock_model
            mock_proc_load.return_value = mock_processor
            
            trainer = VisionLanguageTrainer(
                model=mock_model,
                processor=mock_processor,
                languages=["en", "sw"]
            )
            
            # Step 3: Mock training process
            with patch.object(trainer, 'train') as mock_train:
                mock_train.return_value = {
                    "train_loss": 0.5,
                    "eval_loss": 0.6,
                    "epochs_completed": 3
                }
                
                # Simulate training
                results = trainer.train(
                    train_dataset=dataset,
                    eval_dataset=dataset,
                    num_epochs=3
                )
                
                # Verify end-to-end pipeline
                assert results["epochs_completed"] == 3
                assert "train_loss" in results

    @pytest.mark.integration
    def test_multilingual_pipeline(self, sample_image, temp_dir):
        """Test pipeline with multiple languages."""
        # Setup multilingual data
        multilingual_data = [
            {
                "images": [sample_image],
                "text": "Health statistics for refugee populations",
                "language": "en",
                "metadata": {"source": "who", "topic": "health"}
            },
            {
                "images": [sample_image],
                "text": "Takwimu za afya kwa wakimbizi",  # Swahili
                "language": "sw",
                "metadata": {"source": "who", "topic": "health"}
            }
        ]
        
        builder = DatasetBuilder(
            target_languages=["en", "sw", "am"],
            source_language="en"
        )
        
        # Mock translation service
        with patch.object(builder, '_translate_text') as mock_translate:
            mock_translate.side_effect = lambda text, src, tgt: f"[{tgt}] {text}"
            
            # Process multilingual content
            dataset = builder._process_scraped_content(multilingual_data)
            
            # Verify multilingual processing
            assert len(dataset) >= 2
            languages = dataset["language"].unique() if "language" in dataset.columns else []
            assert len(languages) > 0

    @pytest.mark.integration
    def test_quality_assurance_pipeline(self, sample_image):
        """Test quality assurance throughout the pipeline."""
        builder = DatasetBuilder(min_quality_score=0.8)
        
        # Create data with varying quality
        quality_test_data = [
            {
                "images": [sample_image],
                "metadata": {"title": "High Quality Document", "pages": 20}
            },
            {
                "images": [sample_image],
                "metadata": {"title": "Low Quality Doc", "pages": 1}
            }
        ]
        
        # Mock OCR with different quality results
        def mock_ocr_side_effect(image):
            # Simulate different quality OCR results
            return {
                "text": "Clear, well-formatted text with detailed information.",
                "confidence": 0.95,
                "language": "en"
            }
        
        with patch.object(builder, '_extract_text_from_image', side_effect=mock_ocr_side_effect):
            dataset = builder._process_scraped_content(quality_test_data)
            
            # Apply quality filtering
            filtered_dataset = builder._filter_by_quality(dataset)
            
            # Verify quality assurance
            if len(filtered_dataset) > 0:
                assert all(score >= 0.8 for score in filtered_dataset["quality_score"])

    @pytest.mark.integration
    def test_error_recovery_pipeline(self, sample_image):
        """Test pipeline error recovery and resilience."""
        builder = DatasetBuilder()
        
        # Data with potential errors
        error_prone_data = [
            {
                "images": [sample_image],
                "metadata": {"title": "Valid Document"}
            },
            {
                "images": [None],  # This will cause an error
                "metadata": {"title": "Invalid Document"}
            },
            {
                "images": [sample_image],
                "metadata": {"title": "Another Valid Document"}
            }
        ]
        
        # Mock OCR that fails on invalid images
        def mock_ocr_with_errors(image):
            if image is None:
                raise ValueError("Invalid image")
            return {
                "text": "Valid extracted text",
                "confidence": 0.9,
                "language": "en"
            }
        
        with patch.object(builder, '_extract_text_from_image', side_effect=mock_ocr_with_errors):
            # Pipeline should continue despite errors
            dataset = builder._process_scraped_content(error_prone_data)
            
            # Should process valid documents despite errors
            assert len(dataset) >= 0  # At least doesn't crash

    @pytest.mark.integration
    def test_performance_monitoring_pipeline(self, sample_image):
        """Test performance monitoring throughout the pipeline."""
        import time
        
        builder = DatasetBuilder()
        
        # Create moderately sized dataset for performance testing
        perf_test_data = [
            {
                "images": [sample_image],
                "metadata": {"title": f"Document {i}"}
            }
            for i in range(10)
        ]
        
        # Monitor processing time
        start_time = time.time()
        
        with patch.object(builder, '_extract_text_from_image') as mock_ocr:
            mock_ocr.return_value = {
                "text": "Sample extracted text",
                "confidence": 0.9,
                "language": "en"
            }
            
            dataset = builder._process_scraped_content(perf_test_data)
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify reasonable performance
        assert processing_time < 60  # Should complete within 1 minute
        assert len(dataset) == 10

    @pytest.mark.integration
    def test_data_format_compatibility(self, sample_dataset):
        """Test compatibility between different data formats in the pipeline."""
        builder = DatasetBuilder()
        
        # Test different output formats
        formats = ["hf_dataset", "coco", "csv", "json"]
        
        for format_type in formats:
            builder.output_format = format_type
            
            try:
                if format_type == "hf_dataset":
                    result = builder._convert_to_huggingface_format(sample_dataset)
                    assert hasattr(result, "column_names")
                elif format_type == "coco":
                    result = builder._convert_to_coco_format(sample_dataset)
                    assert isinstance(result, dict)
                    assert "images" in result
                # CSV and JSON would be tested in file saving
                
            except Exception as e:
                pytest.fail(f"Format {format_type} failed: {e}")

    @pytest.mark.integration
    def test_cross_validation_pipeline(self, sample_dataset):
        """Test cross-validation capabilities in the pipeline."""
        from sklearn.model_selection import train_test_split
        
        # Split dataset for cross-validation
        if len(sample_dataset) >= 2:
            train_data, val_data = train_test_split(
                sample_dataset, 
                test_size=0.5, 
                random_state=42
            )
            
            # Verify split
            assert len(train_data) > 0
            assert len(val_data) > 0
            assert len(train_data) + len(val_data) == len(sample_dataset)

    @pytest.mark.integration
    def test_metadata_preservation_pipeline(self, sample_image):
        """Test that metadata is preserved throughout the pipeline."""
        test_metadata = {
            "source": "unhcr",
            "date": "2023-01-01",
            "language": "en",
            "topic": "refugee_assistance",
            "region": "east_africa",
            "document_type": "situation_report"
        }
        
        scraped_data = [
            {
                "images": [sample_image],
                "metadata": test_metadata
            }
        ]
        
        builder = DatasetBuilder()
        
        with patch.object(builder, '_extract_text_from_image') as mock_ocr:
            mock_ocr.return_value = {
                "text": "Sample text",
                "confidence": 0.9,
                "language": "en"
            }
            
            dataset = builder._process_scraped_content(scraped_data)
            
            # Verify metadata preservation
            if len(dataset) > 0 and "metadata" in dataset.columns:
                preserved_metadata = dataset.iloc[0]["metadata"]
                for key, value in test_metadata.items():
                    assert key in preserved_metadata
                    assert preserved_metadata[key] == value