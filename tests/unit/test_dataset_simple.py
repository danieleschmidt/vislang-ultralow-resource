"""Simple tests for DatasetBuilder functionality."""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from vislang_ultralow.dataset import DatasetBuilder


class TestDatasetBuilderSimple:
    """Simple test cases for DatasetBuilder class."""
    
    @patch.object(DatasetBuilder, '_initialize_ocr_engines')
    @patch.object(DatasetBuilder, '_initialize_translation')
    def test_initialization(self, mock_translation, mock_ocr):
        """Test DatasetBuilder initialization."""
        builder = DatasetBuilder(
            target_languages=["sw", "am"],
            source_language="en",
            min_quality_score=0.9
        )
        
        assert builder.target_languages == ["sw", "am"]
        assert builder.source_language == "en"
        assert builder.min_quality_score == 0.9
        mock_ocr.assert_called_once()
        mock_translation.assert_called_once()
    
    def test_classify_image_type(self):
        """Test image type classification."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            # Test chart classification
            chart_info = {
                'alt': 'Chart showing statistics',
                'src': 'chart.png',
                'width': 800,
                'height': 400
            }
            assert builder._classify_image_type(chart_info) == 'chart'
            
            # Test map classification
            map_info = {'alt': 'Map of regions', 'src': 'map.jpg'}
            assert builder._classify_image_type(map_info) == 'map'
            
            # Test wide aspect ratio
            wide_info = {'width': 600, 'height': 200, 'alt': 'wide image'}
            assert builder._classify_image_type(wide_info) == 'chart'
    
    @patch('requests.get')
    def test_load_image_from_url(self, mock_get):
        """Test loading image from URL."""
        # Create test image
        test_img = Image.new('RGB', (100, 100), 'red')
        from io import BytesIO
        img_bytes = BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Mock response
        mock_response = Mock()
        mock_response.content = img_bytes.getvalue()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            image_info = {'src': 'https://example.com/test.png'}
            loaded = builder._load_image(image_info)
            
            assert loaded is not None
            assert loaded.mode == 'RGB'
            assert loaded.size == (100, 100)
    
    def test_consensus_ocr_results(self):
        """Test OCR consensus mechanism."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            results = [
                ('tesseract', {'text': 'Sample text', 'confidence': 0.8}),
                ('easyocr', {'text': 'Sample text', 'confidence': 0.9}),
                ('paddleocr', {'text': 'Sample text', 'confidence': 0.85})
            ]
            
            consensus = builder._consensus_ocr_result(results)
            
            assert consensus['text'] == 'Sample text'
            assert consensus['confidence'] == 0.9  # Highest confidence
            assert len(consensus['engines']) == 3
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            # High quality example
            good_pair = {
                'instruction': 'What information is shown in this chart?',
                'response': 'This chart displays refugee statistics showing displacement trends over time.'
            }
            good_ocr = {'confidence': 0.95, 'text': 'Refugee Statistics Chart 2023'}
            
            score = builder._calculate_quality_score(good_pair, good_ocr)
            assert score > 0.8
            
            # Low quality example
            bad_pair = {'instruction': 'What?', 'response': 'Text'}
            bad_ocr = {'confidence': 0.3, 'text': ''}
            
            score = builder._calculate_quality_score(bad_pair, bad_ocr)
            assert score < 0.5
    
    def test_validate_quality(self):
        """Test quality validation."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"], min_quality_score=0.8)
            
            # Valid high-quality item
            valid_item = {
                'instruction': 'Describe this humanitarian infographic.',
                'response': 'This infographic shows statistics about refugee displacement.',
                'quality_score': 0.85,
                'ocr_confidence': 0.9
            }
            assert builder.validate_quality(valid_item) is True
            
            # Invalid low-quality item
            invalid_item = {
                'instruction': 'What?',
                'response': 'Text',
                'quality_score': 0.6,  # Below threshold
                'ocr_confidence': 0.9
            }
            assert builder.validate_quality(invalid_item) is False
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            items = [{'id': f'item_{i}', 'score': 0.9} for i in range(100)]
            
            splits = builder._split_dataset(items, 0.8, 0.1, 0.1)
            
            assert len(splits['train']) == 80
            assert len(splits['validation']) == 10
            assert len(splits['test']) == 10
            
            # Check no duplicates across splits
            all_ids = []
            for split in splits.values():
                all_ids.extend([item['id'] for item in split])
            assert len(set(all_ids)) == 100
    
    def test_instruction_generation(self):
        """Test instruction-response generation."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            image_info = {'alt': 'Refugee statistics chart', 'width': 800}
            ocr_results = {'text': 'Refugees: 50M worldwide', 'confidence': 0.9}
            doc = {'title': 'Global Report', 'content': 'Annual refugee statistics...'}
            
            pairs = builder._generate_instructions(image_info, ocr_results, doc, 'chart')
            
            assert len(pairs) > 0
            assert all('instruction' in p and 'response' in p for p in pairs)
            
            # Check that chart-specific templates are used
            responses = [p['response'] for p in pairs]
            assert any('chart' in r.lower() for r in responses)
    
    @patch('datasets.Dataset.from_list')
    def test_huggingface_formatting(self, mock_from_list):
        """Test HuggingFace dataset formatting."""
        mock_from_list.return_value = Mock()
        
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            splits = {
                'train': [{
                    'id': 'test_1',
                    'instruction': 'What is shown?',
                    'response': 'A chart',
                    'image_type': 'chart',
                    'language': 'en',
                    'source': 'unhcr',
                    'quality_score': 0.9,
                    'ocr_confidence': 0.85
                }]
            }
            
            result = builder._format_huggingface_dataset(splits)
            
            mock_from_list.assert_called_once()
            # Check that the data was formatted correctly
            call_data = mock_from_list.call_args[0][0]
            assert len(call_data) == 1
            assert call_data[0]['instruction'] == 'What is shown?'
    
    def test_coco_formatting(self):
        """Test COCO format dataset creation."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            splits = {
                'train': [{
                    'id': 'test_1',
                    'instruction': 'Describe this image',
                    'response': 'Shows refugee data',
                    'image_type': 'chart',
                    'language': 'en',
                    'quality_score': 0.9,
                    'image_info': {'width': 800, 'height': 600}
                }]
            }
            
            coco_data = builder._format_coco_dataset(splits)
            
            assert 'info' in coco_data
            assert 'images' in coco_data
            assert 'annotations' in coco_data
            assert 'categories' in coco_data
            
            assert len(coco_data['images']) == 1
            assert len(coco_data['annotations']) == 1
            assert coco_data['images'][0]['width'] == 800
            assert coco_data['annotations'][0]['instruction'] == 'Describe this image'
    
    def test_dataset_statistics(self):
        """Test dataset statistics generation."""
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en", "sw"])
            
            # Create mock HuggingFace dataset
            class MockDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __iter__(self):
                    return iter(self.data)
            
            mock_dataset = {
                'train': MockDataset([
                    {'language': 'en', 'image_type': 'chart', 'source': 'unhcr', 'quality_score': 0.9},
                    {'language': 'sw', 'image_type': 'map', 'source': 'who', 'quality_score': 0.8}
                ]),
                'test': MockDataset([
                    {'language': 'en', 'image_type': 'infographic', 'source': 'unicef', 'quality_score': 0.7}
                ])
            }
            
            stats = builder.get_dataset_statistics(mock_dataset)
            
            assert stats['total_items'] == 3
            assert stats['splits']['train'] == 2
            assert stats['splits']['test'] == 1
    
    @patch.object(DatasetBuilder, '_process_document')
    def test_build_integration(self, mock_process):
        """Test dataset building integration."""
        mock_process.return_value = [{
            'id': 'item_1',
            'instruction': 'What does this show?',
            'response': 'A humanitarian report chart',
            'quality_score': 0.9,
            'ocr_confidence': 0.85,
            'image_type': 'chart',
            'language': 'en',
            'source': 'unhcr'
        }]
        
        with patch.object(DatasetBuilder, '_initialize_ocr_engines'), \
             patch.object(DatasetBuilder, '_initialize_translation'):
            builder = DatasetBuilder(target_languages=["en"])
            
            documents = [{
                'url': 'https://test.com/doc.pdf',
                'title': 'Test Document',
                'source': 'unhcr',
                'language': 'en',
                'content': 'Test content',
                'images': [{'src': 'test.jpg', 'alt': 'chart'}]
            }]
            
            result = builder.build(documents, output_format="custom")
            
            assert 'train' in result
            assert 'validation' in result  
            assert 'test' in result
            
            # Check that processing was called for each document
            mock_process.assert_called_once()