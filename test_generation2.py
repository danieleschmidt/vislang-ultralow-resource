#!/usr/bin/env python3
"""Test Generation 2 robust functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_robust_dataset_builder():
    """Test Generation 2 enhanced DatasetBuilder with security and error handling."""
    try:
        from vislang_ultralow import DatasetBuilder
        
        # Initialize with enhanced monitoring
        builder = DatasetBuilder(
            target_languages=["sw", "am"],
            source_language="en",
            min_quality_score=0.6
        )
        
        print("✓ DatasetBuilder initialized with Generation 2 enhancements")
        
        # Test health status
        health = builder.get_health_status()
        print(f"✓ Health Status: {health['status']}")
        
        # Test performance metrics
        metrics = builder.get_performance_metrics()
        print(f"✓ Performance Metrics: {metrics['documents_processed']} processed")
        
        # Test with various document types including edge cases
        test_documents = [
            {
                'id': 'secure-doc-1',
                'url': 'https://example.org/humanitarian-report.pdf',
                'title': 'Safe Humanitarian Report',
                'content': 'This is a safe test document with proper content.',
                'images': [{
                    'path': 'safe_image.jpg',
                    'alt_text': 'Safe test image'
                }],
                'source': 'test',
                'timestamp': '2024-01-01'
            },
            {
                'id': 'invalid-doc',
                'url': 'http://localhost/malicious.pdf',  # Should be blocked
                'title': 'Malicious Document',
                'content': 'This document should fail security validation.',
                'images': [{
                    'path': '../../../etc/passwd',  # Directory traversal attempt
                    'alt_text': 'malicious path'
                }],
                'source': 'test',
                'timestamp': '2024-01-01'
            },
            {
                'id': 'empty-doc',
                'title': 'Empty Document',
                'content': '',
                'images': [],
                'source': 'test',
                'timestamp': '2024-01-01'
            }
        ]
        
        # Test secure dataset building
        result = builder.build(test_documents, output_format="raw")
        
        print(f"✓ Secure processing completed")
        print(f"  - Total documents: {len(test_documents)}")
        print(f"  - Training items: {len(result.get('train', []))}")
        
        # Test final health check
        final_health = builder.get_health_status()
        final_metrics = builder.get_performance_metrics()
        
        print(f"✓ Final Health: {final_health['status']} (Error rate: {final_health.get('error_rate', 0):.2%})")
        print(f"✓ Final Metrics: {final_metrics['documents_processed']} processed, {final_metrics['errors_encountered']} errors")
        
        return True
        
    except Exception as e:
        print(f"✗ Generation 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_features():
    """Test security validation features."""
    try:
        from vislang_ultralow import DatasetBuilder
        
        builder = DatasetBuilder(
            target_languages=["en"],
            source_language="en"
        )
        
        # Test security validation methods directly
        safe_doc = {
            'id': 'safe-123',
            'url': 'https://example.com/report.pdf',
            'content': 'Safe content',
            'images': [{'path': 'image.jpg'}]
        }
        
        unsafe_doc = {
            'id': '../../../malicious',
            'url': 'file:///etc/passwd',
            'content': 'x' * (1024 * 1024 + 1),  # Exceeds size limit
            'images': [{'path': '../malicious.exe'}]
        }
        
        safe_result = builder._validate_document_security(safe_doc)
        unsafe_result = builder._validate_document_security(unsafe_doc)
        
        print(f"✓ Safe document validation: {safe_result}")
        print(f"✓ Unsafe document blocked: {not unsafe_result}")
        
        return safe_result and not unsafe_result
        
    except Exception as e:
        print(f"✗ Security test failed: {e}")
        return False

def main():
    """Run Generation 2 tests."""
    print("Generation 2: Make It Robust - Testing")
    print("=" * 50)
    
    tests = [
        ("Robust DatasetBuilder", test_robust_dataset_builder),
        ("Security Features", test_security_features)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if test_func():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Generation 2 Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 Generation 2: Make It Robust - COMPLETE!")
        return 0
    else:
        print("❌ Generation 2 needs fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())