#!/usr/bin/env python3
"""Simple test for basic functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports work."""
    try:
        # Test individual component imports
        from vislang_ultralow.dataset import DatasetBuilder
        from vislang_ultralow.scraper import HumanitarianScraper
        from vislang_ultralow.trainer import VisionLanguageTrainer, VisionLanguageDataset
        
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_builder():
    """Test DatasetBuilder basic functionality."""
    try:
        from vislang_ultralow.dataset import DatasetBuilder
        
        builder = DatasetBuilder(
            target_languages=["sw"],
            source_language="en",
            min_quality_score=0.5
        )
        
        # Test with simple mock document
        mock_docs = [{
            'id': 'test1',
            'title': 'Test Doc',
            'content': 'This is test content',
            'images': [{
                'path': 'test.jpg',
                'alt_text': 'test image'
            }],
            'source': 'test',
            'timestamp': '2024-01-01'
        }]
        
        result = builder.build(mock_docs, output_format="raw")
        print(f"✓ DatasetBuilder created {len(result.get('train', []))} items")
        return True
        
    except Exception as e:
        print(f"✗ DatasetBuilder test failed: {e}")
        return False

def test_scraper():
    """Test HumanitarianScraper basic functionality.""" 
    try:
        from vislang_ultralow.scraper import HumanitarianScraper
        
        scraper = HumanitarianScraper(
            sources=["unhcr"],
            languages=["en"]
        )
        
        # Test stats
        stats = scraper.get_stats()
        print(f"✓ HumanitarianScraper initialized with stats: {stats}")
        return True
        
    except Exception as e:
        print(f"✗ Scraper test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple Functionality Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("DatasetBuilder", test_dataset_builder), 
        ("Scraper", test_scraper)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if test_func():
            passed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())