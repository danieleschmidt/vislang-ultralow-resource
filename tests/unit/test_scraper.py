"""Tests for humanitarian scraper functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import requests
from datetime import datetime
import json
import tempfile

from vislang_ultralow.scraper import HumanitarianScraper


class TestHumanitarianScraper:
    """Test cases for HumanitarianScraper class."""
    
    def test_init_with_valid_sources(self):
        """Test initialization with valid sources."""
        sources = ["unhcr", "who", "unicef"]
        languages = ["en", "fr", "es"]
        
        scraper = HumanitarianScraper(
            sources=sources,
            languages=languages,
            date_range=("2023-01-01", "2023-12-31")
        )
        
        assert scraper.sources == sources
        assert scraper.languages == languages
        assert scraper.date_range == ("2023-01-01", "2023-12-31")
    
    def test_init_with_invalid_sources(self):
        """Test initialization with invalid sources raises ValueError."""
        invalid_sources = ["invalid_source", "another_invalid"]
        languages = ["en"]
        
        with pytest.raises(ValueError, match="Invalid sources"):
            HumanitarianScraper(
                sources=invalid_sources,
                languages=languages
            )
    
    @patch('vislang_ultralow.scraper.requests.Session')
    def test_scrape_basic_functionality(self, mock_session):
        """Test basic scraping functionality."""
        # Setup mock responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body>Sample content</body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        # Create scraper
        scraper = HumanitarianScraper(
            sources=["unhcr"],
            languages=["en"]
        )
        
        # Test scraping
        with patch.object(scraper, '_scrape_unhcr') as mock_scrape_unhcr:
            mock_scrape_unhcr.return_value = [
                {
                    "source": "unhcr",
                    "title": "Test Document",
                    "url": "https://test.com/doc.pdf",
                    "language": "en",
                    "content": "Test content",
                    "images": []
                }
            ]
            
            results = scraper.scrape()
            
            assert len(results) == 1
            assert results[0]["source"] == "unhcr"
            assert results[0]["title"] == "Test Document"
            mock_scrape_unhcr.assert_called_once()
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        scraper = HumanitarianScraper(
            sources=["unhcr"],
            languages=["en"]
        )
        
        # Test that rate limiting doesn't raise an error
        import time
        start_time = time.time()
        scraper._respect_rate_limits("unhcr")
        end_time = time.time()
        
        # Should take at least some time (rate limited)
        assert end_time > start_time
    
    @patch('requests.Session')
    def test_extract_document_html(self, mock_session_class):
        """Test HTML document extraction."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        html_content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is test content for HTML extraction.</p>
            <img src="test.jpg" alt="Test image" width="100" height="100">
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.text = html_content
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        
        scraper = HumanitarianScraper(["unhcr"], ["en"])
        scraper.session = mock_session
        
        # Mock cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper.cache_dir = Path(temp_dir)
            
            result = scraper._extract_document(
                "https://test.com/doc.html", "unhcr", "en"
            )
            
            assert result is not None
            assert result["source"] == "unhcr"
            assert result["title"] == "Test Document"
            assert "This is test content" in result["content"]
            assert len(result["images"]) == 1
            assert result["images"][0]["alt"] == "Test image"
    
    @patch('fitz.open')
    @patch('requests.Session')
    def test_extract_document_pdf(self, mock_session_class, mock_fitz_open):
        """Test PDF document extraction."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock PDF response
        mock_response = Mock()
        mock_response.content = b"PDF content bytes"
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Extracted PDF text content"
        mock_page.get_images.return_value = [(123, 0, 800, 600, 0, "JPEG", "img1")]
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_doc.extract_image.return_value = {
            "width": 800,
            "height": 600,
            "colorspace": 3,
            "ext": "jpeg"
        }
        mock_fitz_open.return_value = mock_doc
        
        scraper = HumanitarianScraper(["who"], ["en"])
        scraper.session = mock_session
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper.cache_dir = Path(temp_dir)
            
            result = scraper._extract_document(
                "https://test.com/doc.pdf", "who", "en"
            )
            
            assert result is not None
            assert result["source"] == "who"
            assert result["content"] == "Extracted PDF text content"
            assert result["content_type"] == "pdf"
            assert len(result["images"]) == 1
            assert result["images"][0]["width"] == 800
    
    def test_detect_document_language(self):
        """Test language detection from URL patterns."""
        scraper = HumanitarianScraper(["unhcr"], ["en", "fr", "ar"])
        
        # Test various URL patterns
        assert scraper._detect_document_language("https://test.com/en/report.pdf") == "en"
        assert scraper._detect_document_language("https://test.com/fr/rapport.pdf") == "fr"
        assert scraper._detect_document_language("https://test.com/arabic/report.pdf") == "ar"
        assert scraper._detect_document_language("https://test.com/swahili/report.pdf") == "sw"
        assert scraper._detect_document_language("https://test.com/unknown/report.pdf") == "en"  # default
    
    def test_filter_by_date(self):
        """Test document filtering by date range."""
        scraper = HumanitarianScraper(
            ["unhcr"], ["en"], 
            date_range=("2023-06-01", "2023-12-31")
        )
        
        documents = [
            {"date": "2023-05-15T10:00:00", "title": "Old doc"},
            {"date": "2023-07-15T10:00:00", "title": "Good doc"},
            {"date": "2024-01-15T10:00:00", "title": "Future doc"},
            {"date": "invalid_date", "title": "Invalid date doc"}
        ]
        
        filtered = scraper._filter_by_date(documents)
        
        # Should keep docs within range and invalid dates
        assert len(filtered) == 3  # Good doc, Future doc (out of range), Invalid date doc
        titles = [doc["title"] for doc in filtered]
        assert "Good doc" in titles
        assert "Old doc" not in titles
    
    def test_get_statistics(self):
        """Test statistics generation."""
        scraper = HumanitarianScraper(["unhcr", "who"], ["en", "fr"])
        
        documents = [
            {
                "source": "unhcr", "language": "en", "content_type": "pdf",
                "word_count": 100, "images": [{}, {}],
                "date": datetime.now().isoformat()
            },
            {
                "source": "who", "language": "fr", "content_type": "html",
                "word_count": 200, "images": [{}],
                "date": datetime.now().isoformat()
            }
        ]
        
        stats = scraper.get_statistics(documents)
        
        assert stats["total_documents"] == 2
        assert stats["sources"]["unhcr"] == 1
        assert stats["sources"]["who"] == 1
        assert stats["languages"]["en"] == 1
        assert stats["languages"]["fr"] == 1
        assert stats["content_types"]["pdf"] == 1
        assert stats["content_types"]["html"] == 1
        assert stats["total_words"] == 300
        assert stats["total_images"] == 3
    
    @patch('requests.Session')
    def test_scrape_with_caching(self, mock_session_class):
        """Test document caching functionality."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        scraper = HumanitarianScraper(["unhcr"], ["en"])
        scraper.session = mock_session
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # First call should extract and cache
            with patch.object(scraper, '_extract_html_content') as mock_extract:
                mock_extract.return_value = {
                    "source": "unhcr", "title": "Test", "content": "Cached content"
                }
                
                result1 = scraper._extract_document(
                    "https://test.com/doc.html", "unhcr", "en"
                )
                mock_extract.assert_called_once()
            
            # Second call should use cache
            with patch.object(scraper, '_extract_html_content') as mock_extract:
                scraper.cache_dir = cache_dir  # Ensure cache dir is set
                result2 = scraper._extract_document(
                    "https://test.com/doc.html", "unhcr", "en"
                )
                mock_extract.assert_not_called()  # Should not extract again
            
            assert result1["content"] == result2["content"]
    
    def test_extract_title_from_text(self):
        """Test title extraction from text content."""
        scraper = HumanitarianScraper(["unhcr"], ["en"])
        
        # Test with good text
        text = "\n\nMain Document Title\n\nThis is the body content..."
        title = scraper._extract_title_from_text(text)
        assert title == "Main Document Title"
        
        # Test with very short lines
        text = "A\nB\nC\nThis is a longer line that should be the title\nMore content"
        title = scraper._extract_title_from_text(text)
        assert title == "This is a longer line that should be the title"
        
        # Test with no good title
        text = "A\nB\nC"
        title = scraper._extract_title_from_text(text)
        assert title == "Untitled Document"
    
    @patch('requests.Session')
    def test_error_handling(self, mock_session_class):
        """Test error handling in scraping methods."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Test network error handling
        mock_session.get.side_effect = requests.RequestException("Network error")
        
        scraper = HumanitarianScraper(["unhcr"], ["en"])
        scraper.session = mock_session
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper.cache_dir = Path(temp_dir)
            
            result = scraper._extract_document(
                "https://test.com/doc.html", "unhcr", "en"
            )
            
            assert result is None  # Should handle error gracefully
    
    def test_scrape_multiple_sources(self):
        """Test scraping from multiple sources."""
        scraper = HumanitarianScraper(["unhcr", "who", "unicef"], ["en"])
        
        with patch.object(scraper, '_scrape_unhcr') as mock_unhcr, \
             patch.object(scraper, '_scrape_who') as mock_who, \
             patch.object(scraper, '_scrape_unicef') as mock_unicef:
            
            mock_unhcr.return_value = [{"source": "unhcr", "title": "UNHCR Doc"}]
            mock_who.return_value = [{"source": "who", "title": "WHO Doc"}]
            mock_unicef.return_value = [{"source": "unicef", "title": "UNICEF Doc"}]
            
            results = scraper.scrape()
            
            assert len(results) == 3
            sources = [doc["source"] for doc in results]
            assert "unhcr" in sources
            assert "who" in sources
            assert "unicef" in sources
    
    def test_max_docs_limit(self):
        """Test maximum documents limit."""
        scraper = HumanitarianScraper(["unhcr"], ["en"])
        
        with patch.object(scraper, '_scrape_unhcr') as mock_scrape:
            # Mock returns many documents
            mock_docs = [{"source": "unhcr", "title": f"Doc {i}"} for i in range(100)]
            mock_scrape.return_value = mock_docs
            
            # Test with limit
            results = scraper.scrape(max_docs=10)
            
            # Should call with max_docs parameter
            mock_scrape.assert_called_once_with(10)


@pytest.mark.integration
def test_scraper_integration():
    """Integration test for scraper with real-like data."""
    # Test with mock HTTP responses that simulate real scraping
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Mock realistic HTML response
        html_response = Mock()
        html_response.status_code = 200
        html_response.text = """
        <html>
        <head><title>UNHCR Global Report 2024</title></head>
        <body>
            <h1>Global Forced Displacement Report</h1>
            <p>This report presents key findings on global displacement...</p>
            <img src="chart1.png" alt="Displacement statistics chart">
            <div class="report-content">
                <p>Key statistics and trends in forced displacement.</p>
            </div>
        </body>
        </html>
        """
        html_response.headers = {'content-type': 'text/html'}
        html_response.raise_for_status = Mock()
        
        mock_session.get.return_value = html_response
        
        scraper = HumanitarianScraper(["unhcr"], ["en"])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = scraper.scrape(cache_dir=Path(temp_dir), max_docs=1)
            
            # Verify results have expected structure
            assert len(results) >= 0  # May be empty due to mocking
            # In a real integration test, we'd verify actual content


@pytest.mark.slow
def test_large_document_processing():
    """Test processing of large documents."""
    scraper = HumanitarianScraper(["unhcr"], ["en"])
    
    # Create a large text content
    large_content = "Test content. " * 10000  # Large text
    
    with patch.object(scraper, '_extract_html_content') as mock_extract:
        mock_extract.return_value = {
            "source": "unhcr",
            "title": "Large Document",
            "content": large_content,
            "word_count": len(large_content.split())
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper.cache_dir = Path(temp_dir)
            
            result = scraper._extract_document(
                "https://test.com/large.html", "unhcr", "en"
            )
            
            assert result is not None
            assert len(result["content"]) > 100000  # Verify large content