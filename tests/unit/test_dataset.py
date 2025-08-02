"""Unit tests for the DatasetBuilder class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

from vislang_ultralow.dataset import DatasetBuilder


class TestDatasetBuilder:
    """Test cases for DatasetBuilder."""

    def test_init_with_default_params(self):
        """Test DatasetBuilder initialization with default parameters."""
        builder = DatasetBuilder()
        
        assert builder.target_languages == ["en"]
        assert builder.source_language == "en"
        assert builder.min_quality_score == 0.8
        assert builder.output_format == "hf_dataset"

    def test_init_with_custom_params(self):
        """Test DatasetBuilder initialization with custom parameters."""
        builder = DatasetBuilder(
            target_languages=["sw", "am", "ha"],
            source_language="en",
            min_quality_score=0.9,
            output_format="coco"
        )
        
        assert builder.target_languages == ["sw", "am", "ha"]
        assert builder.source_language == "en"
        assert builder.min_quality_score == 0.9
        assert builder.output_format == "coco"

    def test_process_image_with_pil_image(self, sample_image):
        """Test image processing with PIL Image input."""
        builder = DatasetBuilder()
        result = builder._process_image(sample_image)
        
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)  # Should maintain size
        assert result.mode == "RGB"

    def test_process_image_with_file_path(self, sample_image, temp_dir):
        """Test image processing with file path input."""
        # Save image to temp file
        image_path = temp_dir / "test_image.jpg"
        sample_image.save(image_path)
        
        builder = DatasetBuilder()
        result = builder._process_image(str(image_path))
        
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_process_image_invalid_input(self):
        """Test image processing with invalid input."""
        builder = DatasetBuilder()
        
        with pytest.raises(ValueError):
            builder._process_image("nonexistent_file.jpg")

    @patch('vislang_ultralow.dataset.DatasetBuilder._run_ocr')
    def test_extract_text_from_image(self, mock_ocr, sample_image, sample_ocr_result):
        """Test text extraction from images using OCR."""
        mock_ocr.return_value = sample_ocr_result
        
        builder = DatasetBuilder()
        result = builder._extract_text_from_image(sample_image)
        
        assert result["text"] == "Sample text extracted from image"
        assert result["confidence"] == 0.95
        assert result["language"] == "en"
        mock_ocr.assert_called_once_with(sample_image)

    def test_run_ocr_with_multiple_engines(self, sample_image):
        """Test OCR execution with multiple engines."""
        builder = DatasetBuilder()
        
        # Mock OCR engines
        mock_tesseract = Mock()
        mock_tesseract.extract_text.return_value = {
            "text": "Tesseract result",
            "confidence": 0.8,
            "language": "en"
        }
        
        mock_easyocr = Mock()
        mock_easyocr.extract_text.return_value = {
            "text": "EasyOCR result",
            "confidence": 0.9,
            "language": "en"
        }
        
        builder.ocr_engines = {"tesseract": mock_tesseract, "easyocr": mock_easyocr}
        
        result = builder._run_ocr(sample_image)
        
        # Should return the result with highest confidence
        assert result["text"] == "EasyOCR result"
        assert result["confidence"] == 0.9

    def test_generate_instruction_templates(self):
        """Test instruction template generation."""
        builder = DatasetBuilder()
        
        text = "Sample extracted text"
        image_description = "A document with text"
        
        instructions = builder._generate_instruction_templates(text, image_description)
        
        assert len(instructions) > 0
        assert any("text" in instr.lower() for instr in instructions)
        assert any("image" in instr.lower() for instr in instructions)

    def test_translate_text(self):
        """Test text translation between languages."""
        builder = DatasetBuilder(target_languages=["sw", "am"])
        
        # Mock translator
        mock_translator = Mock()
        mock_translator.translate.return_value = "Maandishi ya mfano"
        builder.translator = mock_translator
        
        result = builder._translate_text("Sample text", "en", "sw")
        
        assert result == "Maandishi ya mfano"
        mock_translator.translate.assert_called_once()

    def test_assess_quality_high_quality(self):
        """Test quality assessment for high-quality data."""
        builder = DatasetBuilder()
        
        data_point = {
            "text": "This is a clear, well-formatted text with good information.",
            "ocr_confidence": 0.95,
            "image_quality": 0.9,
            "language_confidence": 0.85
        }
        
        score = builder._assess_quality(data_point)
        
        assert score >= 0.8  # Should be high quality
        assert 0 <= score <= 1

    def test_assess_quality_low_quality(self):
        """Test quality assessment for low-quality data."""
        builder = DatasetBuilder()
        
        data_point = {
            "text": "txt wth mny errrs",  # Poor text quality
            "ocr_confidence": 0.3,  # Low OCR confidence
            "image_quality": 0.4,   # Poor image quality
            "language_confidence": 0.2  # Low language confidence
        }
        
        score = builder._assess_quality(data_point)
        
        assert score < 0.5  # Should be low quality
        assert 0 <= score <= 1

    def test_filter_by_quality(self, sample_dataset):
        """Test filtering dataset by quality score."""
        builder = DatasetBuilder(min_quality_score=0.9)
        
        # Add quality scores to sample dataset
        sample_dataset["quality_score"] = [0.95, 0.87]
        
        filtered = builder._filter_by_quality(sample_dataset)
        
        # Only the first row should pass the 0.9 threshold
        assert len(filtered) == 1
        assert filtered.iloc[0]["quality_score"] == 0.95

    def test_create_instruction_pairs(self):
        """Test creation of instruction-response pairs."""
        builder = DatasetBuilder(target_languages=["sw"])
        
        data_point = {
            "text": "Sample text",
            "image": Image.new("RGB", (100, 100)),
            "language": "en"
        }
        
        # Mock translation
        with patch.object(builder, '_translate_text') as mock_translate:
            mock_translate.return_value = "Maandishi ya mfano"
            
            pairs = builder._create_instruction_pairs(data_point)
            
            assert len(pairs) > 0
            assert any("sw" in pair.get("target_language", "") for pair in pairs)

    @patch('vislang_ultralow.dataset.DatasetBuilder._extract_text_from_image')
    def test_process_scraped_content(self, mock_extract, sample_image):
        """Test processing of scraped content into dataset format."""
        mock_extract.return_value = {
            "text": "Extracted text",
            "confidence": 0.9,
            "language": "en"
        }
        
        builder = DatasetBuilder()
        
        scraped_data = [
            {
                "images": [sample_image],
                "metadata": {
                    "title": "Test Document",
                    "source": "unhcr",
                    "url": "https://example.com/doc.pdf"
                }
            }
        ]
        
        result = builder._process_scraped_content(scraped_data)
        
        assert len(result) > 0
        assert "text" in result.columns
        assert "image_path" in result.columns
        assert "quality_score" in result.columns

    def test_convert_to_huggingface_format(self, sample_dataset):
        """Test conversion to HuggingFace dataset format."""
        builder = DatasetBuilder(output_format="hf_dataset")
        
        hf_dataset = builder._convert_to_huggingface_format(sample_dataset)
        
        # Should return a HuggingFace Dataset object
        assert hasattr(hf_dataset, 'train_test_split')
        assert len(hf_dataset) == len(sample_dataset)

    def test_convert_to_coco_format(self, sample_dataset):
        """Test conversion to COCO dataset format."""
        builder = DatasetBuilder(output_format="coco")
        
        coco_data = builder._convert_to_coco_format(sample_dataset)
        
        # Should have COCO structure
        assert "images" in coco_data
        assert "annotations" in coco_data
        assert "categories" in coco_data

    def test_save_dataset_csv(self, sample_dataset, temp_dir):
        """Test saving dataset in CSV format."""
        builder = DatasetBuilder()
        output_path = temp_dir / "dataset.csv"
        
        builder._save_dataset(sample_dataset, output_path, format="csv")
        
        assert output_path.exists()
        
        # Verify content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(sample_dataset)

    def test_save_dataset_json(self, sample_dataset, temp_dir):
        """Test saving dataset in JSON format."""
        builder = DatasetBuilder()
        output_path = temp_dir / "dataset.json"
        
        builder._save_dataset(sample_dataset, output_path, format="json")
        
        assert output_path.exists()

    @patch.object(DatasetBuilder, '_process_scraped_content')
    @patch.object(DatasetBuilder, '_filter_by_quality')
    def test_build_dataset_end_to_end(self, mock_filter, mock_process, mock_scraper):
        """Test end-to-end dataset building process."""
        # Setup mocks
        mock_dataset = pd.DataFrame({
            "text": ["Sample text"],
            "image_path": ["/path/to/image.jpg"],
            "language": ["en"],
            "quality_score": [0.9]
        })
        
        mock_process.return_value = mock_dataset
        mock_filter.return_value = mock_dataset
        
        builder = DatasetBuilder()
        
        # Mock scraper data
        scraped_data = [{"content": "mock content"}]
        
        result = builder.build(
            scraper=mock_scraper,
            scraped_data=scraped_data,
            include_infographics=True
        )
        
        assert len(result) > 0
        mock_process.assert_called_once()
        mock_filter.assert_called_once()

    def test_validate_dataset_structure(self, sample_dataset):
        """Test validation of dataset structure."""
        builder = DatasetBuilder()
        
        # Valid dataset should pass
        assert builder._validate_dataset_structure(sample_dataset) is True
        
        # Invalid dataset (missing required columns) should fail
        invalid_dataset = pd.DataFrame({"only_text": ["sample"]})
        assert builder._validate_dataset_structure(invalid_dataset) is False

    def test_detect_duplicates(self):
        """Test duplicate detection in dataset."""
        builder = DatasetBuilder()
        
        dataset = pd.DataFrame({
            "text": ["Same text", "Same text", "Different text"],
            "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "language": ["en", "en", "en"],
            "quality_score": [0.9, 0.9, 0.9]
        })
        
        duplicates = builder._detect_duplicates(dataset)
        
        assert len(duplicates) == 1  # Should find one duplicate pair

    def test_cross_lingual_alignment(self):
        """Test cross-lingual alignment of instruction pairs."""
        builder = DatasetBuilder(target_languages=["sw", "am"])
        
        source_data = {
            "instruction": "What is shown in this image?",
            "response": "This image shows a document with text.",
            "language": "en"
        }
        
        aligned_data = builder._create_cross_lingual_alignment(source_data)
        
        assert len(aligned_data) == 2  # One for each target language
        assert any(item["language"] == "sw" for item in aligned_data)
        assert any(item["language"] == "am" for item in aligned_data)

    def test_augment_dataset(self, sample_dataset):
        """Test dataset augmentation techniques."""
        builder = DatasetBuilder()
        
        augmented = builder._augment_dataset(sample_dataset, augmentation_factor=2)
        
        # Should have more samples than original
        assert len(augmented) >= len(sample_dataset)

    @pytest.mark.slow
    def test_build_with_large_dataset(self):
        """Test building with a large dataset (performance test)."""
        builder = DatasetBuilder()
        
        # Create a large mock dataset
        large_data = []
        for i in range(1000):
            large_data.append({
                "images": [Image.new("RGB", (100, 100))],
                "metadata": {"title": f"Document {i}"}
            })
        
        with patch.object(builder, '_extract_text_from_image') as mock_extract:
            mock_extract.return_value = {
                "text": f"Text {i}",
                "confidence": 0.9,
                "language": "en"
            }
            
            result = builder._process_scraped_content(large_data[:10])  # Process subset
            
            assert len(result) == 10

    def test_handle_corrupted_images(self):
        """Test handling of corrupted or invalid images."""
        builder = DatasetBuilder()
        
        # Create corrupted image data
        corrupted_data = [
            {
                "images": [None],  # Invalid image
                "metadata": {"title": "Corrupted Document"}
            }
        ]
        
        result = builder._process_scraped_content(corrupted_data)
        
        # Should handle gracefully and return empty or filtered results
        assert isinstance(result, pd.DataFrame)

    def test_multilingual_text_validation(self):
        """Test validation of multilingual text content."""
        builder = DatasetBuilder(target_languages=["ar", "hi", "sw"])
        
        texts = [
            "English text",
            "النص العربي",  # Arabic
            "हिंदी पाठ",     # Hindi
            "Maandishi ya Kiswahili"  # Swahili
        ]
        
        for text in texts:
            is_valid = builder._validate_text_quality(text)
            assert isinstance(is_valid, bool)

    def test_batch_processing(self, sample_image):
        """Test batch processing of multiple images."""
        builder = DatasetBuilder()
        
        images = [sample_image] * 5
        
        with patch.object(builder, '_extract_text_from_image') as mock_extract:
            mock_extract.return_value = {
                "text": "Sample text",
                "confidence": 0.9,
                "language": "en"
            }
            
            results = builder._process_image_batch(images)
            
            assert len(results) == 5
            assert mock_extract.call_count == 5