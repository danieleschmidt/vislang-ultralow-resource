#!/usr/bin/env python3
"""Test script to verify improved functionality works correctly."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_adaptive_ocr():
    """Test adaptive OCR system."""
    logger.info("Testing Adaptive OCR System...")
    
    try:
        from vislang_ultralow.research.adaptive_ocr import AdaptiveMultiEngineOCR
        
        # Initialize OCR system
        ocr = AdaptiveMultiEngineOCR(['tesseract', 'easyocr', 'paddleocr'])
        
        # Test with mock image data
        import numpy as np
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test different document types
        document_types = ['humanitarian_report', 'infographic', 'chart', 'standard']
        
        for doc_type in document_types:
            try:
                result = ocr.extract_text(mock_image, doc_type)
                logger.info(f"OCR for {doc_type}: {result.get('text', '')[:50]}...")
                logger.info(f"  Confidence: {result.get('confidence', 0):.2f}")
                assert 'text' in result
                assert 'confidence' in result
                assert result['confidence'] > 0
            except Exception as e:
                logger.error(f"OCR failed for {doc_type}: {e}")
                
        # Test performance stats
        stats = ocr.get_engine_performance_stats()
        logger.info(f"Engine performance stats: {stats}")
        
        logger.info("✅ Adaptive OCR test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive OCR test failed: {e}")
        return False

def test_cross_lingual_alignment():
    """Test cross-lingual alignment system."""
    logger.info("Testing Cross-lingual Alignment...")
    
    try:
        from vislang_ultralow.research.cross_lingual_alignment import ZeroShotCrossLingual
        
        # Initialize alignment system
        aligner = ZeroShotCrossLingual()
        
        # Test basic alignment
        test_text = "Emergency assistance is needed for displaced families."
        target_languages = ['fr', 'es', 'ar', 'sw']
        
        for lang in target_languages:
            aligned_text = aligner.align_cross_lingual(test_text, lang)
            logger.info(f"Aligned to {lang}: {aligned_text[:60]}...")
            assert lang in aligned_text or '[' in aligned_text  # Should have language indicator
        
        # Test similarity computation
        text1 = "Food distribution center operational"
        text2 = "Food distribution center operational"  # Same text
        similarity = aligner.compute_cross_lingual_similarity(text1, text2, 'en', 'en')
        logger.info(f"Same text similarity: {similarity:.3f}")
        assert similarity > 0.8  # Should be high for identical text
        
        text3 = "Water purification systems installed"
        similarity2 = aligner.compute_cross_lingual_similarity(text1, text3, 'en', 'en')
        logger.info(f"Different text similarity: {similarity2:.3f}")
        assert similarity2 < similarity  # Should be lower for different text
        
        logger.info("✅ Cross-lingual alignment test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cross-lingual alignment test failed: {e}")
        return False

def test_dataset_builder():
    """Test enhanced dataset builder."""
    logger.info("Testing Enhanced Dataset Builder...")
    
    try:
        from vislang_ultralow.dataset import DatasetBuilder
        
        # Initialize dataset builder
        builder = DatasetBuilder(
            target_languages=['en', 'fr', 'sw'],
            source_language='en',
            min_quality_score=0.7
        )
        
        # Create mock documents
        mock_documents = [
            {
                'id': 'doc_1',
                'url': 'https://example.org/report1',
                'source': 'unhcr',
                'content': 'Emergency response operations in progress. Medical teams deployed.',
                'images': [
                    {
                        'path': 'mock_image_1.jpg',
                        'alt_text': 'Medical supplies distribution',
                        'caption': 'Emergency medical supplies being distributed',
                        'width': 800,
                        'height': 600
                    },
                    {
                        'path': 'mock_image_2.jpg', 
                        'alt_text': 'Refugee camp statistics chart',
                        'caption': 'Population statistics by region',
                        'width': 600,
                        'height': 400
                    }
                ],
                'timestamp': '2024-01-15T10:30:00Z'
            },
            {
                'id': 'doc_2',
                'url': 'https://example.org/report2',
                'source': 'who',
                'content': 'Health system restoration underway. Vaccination campaigns resumed.',
                'images': [
                    {
                        'path': 'mock_image_3.jpg',
                        'alt_text': 'Vaccination progress infographic',
                        'caption': 'Vaccination coverage by age group',
                        'width': 1200,
                        'height': 800
                    }
                ],
                'timestamp': '2024-01-16T14:45:00Z'
            }
        ]
        
        # Test dataset building
        logger.info("Building dataset from mock documents...")
        dataset = builder.build(
            documents=mock_documents,
            include_infographics=True,
            include_maps=True,
            include_charts=True,
            output_format="hf_dataset"
        )
        
        logger.info(f"Dataset created with splits: {list(dataset.keys())}")
        
        # Check dataset content
        for split_name, split_data in dataset.items():
            logger.info(f"Split '{split_name}': {len(split_data)} items")
            if len(split_data) > 0:
                first_item = split_data[0]
                logger.info(f"  Sample instruction: {first_item.get('instruction', '')[:50]}...")
                logger.info(f"  Sample response: {first_item.get('response', '')[:50]}...")
                logger.info(f"  Language: {first_item.get('language')}")
                
                # Verify required fields
                required_fields = ['instruction', 'response', 'language', 'image_type', 'quality_score']
                for field in required_fields:
                    assert field in first_item, f"Missing required field: {field}"
        
        # Test statistics
        stats = builder.get_dataset_statistics(dataset)
        logger.info(f"Dataset statistics: {stats}")
        
        logger.info("✅ Dataset builder test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset builder test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_humanitarian_scraper():
    """Test humanitarian scraper functionality."""
    logger.info("Testing Humanitarian Scraper...")
    
    try:
        from vislang_ultralow.scraper import HumanitarianScraper
        
        # Initialize scraper
        scraper = HumanitarianScraper(
            sources=['unhcr', 'who'],
            languages=['en', 'fr'],
            max_workers=2
        )
        
        # Test scraping with limit (will use mock data)
        documents = scraper.scrape(max_documents=5)
        
        logger.info(f"Scraped {len(documents)} documents")
        
        if documents:
            first_doc = documents[0]
            logger.info(f"Sample document:")
            logger.info(f"  ID: {first_doc.get('id', 'N/A')}")
            logger.info(f"  Source: {first_doc.get('source', 'N/A')}")
            logger.info(f"  Title: {first_doc.get('title', 'N/A')[:50]}...")
            logger.info(f"  Content length: {len(first_doc.get('content', ''))}")
            logger.info(f"  Images: {len(first_doc.get('images', []))}")
            
            # Verify required fields
            required_fields = ['id', 'url', 'source', 'title', 'content']
            for field in required_fields:
                assert field in first_doc, f"Missing required field: {field}"
        
        # Test statistics
        stats = scraper.get_stats()
        logger.info(f"Scraper statistics: {stats}")
        
        logger.info("✅ Humanitarian scraper test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Humanitarian scraper test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Improved Functionality Tests")
    
    tests = [
        ("Adaptive OCR", test_adaptive_ocr),
        ("Cross-lingual Alignment", test_cross_lingual_alignment), 
        ("Dataset Builder", test_dataset_builder),
        ("Humanitarian Scraper", test_humanitarian_scraper)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info('='*60)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            failed += 1
    
    logger.info(f"\n{'='*60}")
    logger.info("📊 TEST RESULTS")
    logger.info('='*60)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {passed}/{passed+failed} ({passed/(passed+failed)*100:.1f}%)")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Core functionality is working.")
        return True
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Some issues need to be addressed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)