#!/usr/bin/env python3
"""Test basic functionality of the VisLang-UltraLow-Resource framework."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from vislang_ultralow import DatasetBuilder, HumanitarianScraper
        print("✓ Core imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dataset_builder():
    """Test DatasetBuilder basic functionality."""
    try:
        from vislang_ultralow import DatasetBuilder
        
        # Initialize dataset builder
        builder = DatasetBuilder(
            target_languages=["sw", "am"],
            source_language="en",
            min_quality_score=0.5
        )
        
        # Test with mock documents
        mock_documents = [
            {
                'id': 'test-doc-1',
                'url': 'http://example.com/doc1',
                'title': 'Test Humanitarian Document',
                'content': 'This is a test document about humanitarian aid and crisis response.',
                'images': [
                    {
                        'path': 'test_image.jpg',
                        'alt_text': 'Test infographic about aid distribution',
                        'caption': 'Distribution of humanitarian aid'
                    }
                ],
                'source': 'test',
                'timestamp': '2024-01-01T00:00:00'
            }
        ]
        
        # Build dataset
        dataset = builder.build(
            documents=mock_documents,
            include_infographics=True,
            output_format="raw"
        )
        
        print(f"✓ DatasetBuilder created {len(dataset.get('train', []))} training samples")
        return True
        
    except Exception as e:
        print(f"✗ DatasetBuilder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_humanitarian_scraper():
    """Test HumanitarianScraper basic functionality."""
    try:
        from vislang_ultralow import HumanitarianScraper
        
        # Initialize scraper
        scraper = HumanitarianScraper(
            sources=["unhcr"],
            languages=["en", "sw"],
            max_workers=1
        )
        
        # Test with mock scraping (won't make real requests)
        documents = scraper.scrape(max_documents=2)
        
        print(f"✓ HumanitarianScraper created {len(documents)} mock documents")
        return True
        
    except Exception as e:
        print(f"✗ HumanitarianScraper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vision_language_trainer():
    """Test VisionLanguageTrainer basic functionality."""
    try:
        from vislang_ultralow import VisionLanguageTrainer
        from vislang_ultralow.research.placeholder_imports import MockProcessor, MockModel
        
        # Initialize with mock model and processor
        model = MockModel()
        processor = MockProcessor()
        
        trainer = VisionLanguageTrainer(
            model=model,
            processor=processor,
            languages=["en", "sw"],
            use_wandb=False
        )
        
        print("✓ VisionLanguageTrainer initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ VisionLanguageTrainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_research_modules():
    """Test research modules can be imported."""
    try:
        from vislang_ultralow.research.adaptive_ocr import OCRConsensusAlgorithm
        from vislang_ultralow.research.cross_lingual_alignment import ZeroShotCrossLingual
        
        # Test OCR algorithm
        ocr = OCRConsensusAlgorithm()
        result = ocr.consensus_ocr("mock_image_data", ["en", "sw"])
        
        # Test cross-lingual alignment
        aligner = ZeroShotCrossLingual()
        aligned = aligner.align_cross_lingual("Test text", "sw")
        
        print("✓ Research modules working")
        return True
        
    except Exception as e:
        print(f"✗ Research modules test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing VisLang-UltraLow-Resource Basic Functionality")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_imports),
        ("DatasetBuilder", test_dataset_builder),
        ("HumanitarianScraper", test_humanitarian_scraper),
        ("VisionLanguageTrainer", test_vision_language_trainer),
        ("Research Modules", test_research_modules)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Generation 1 implementation is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())