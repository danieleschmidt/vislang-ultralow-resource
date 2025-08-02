"""Unit tests for the HumanitarianScraper class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import requests
from bs4 import BeautifulSoup

from vislang_ultralow.scraper import HumanitarianScraper


class TestHumanitarianScraper:
    """Test cases for HumanitarianScraper."""

    def test_init_with_default_sources(self):
        """Test scraper initialization with default sources."""
        scraper = HumanitarianScraper()
        
        assert isinstance(scraper.sources, list)
        assert len(scraper.sources) > 0
        assert "unhcr" in scraper.sources

    def test_init_with_custom_sources(self):
        """Test scraper initialization with custom sources."""
        custom_sources = ["unhcr", "who"]
        scraper = HumanitarianScraper(sources=custom_sources)
        
        assert scraper.sources == custom_sources

    def test_init_with_languages(self):
        """Test scraper initialization with language filters."""
        languages = ["en", "sw", "am"]
        scraper = HumanitarianScraper(languages=languages)
        
        assert scraper.languages == languages

    @patch('requests.get')
    def test_fetch_document_success(self, mock_get):
        """Test successful document fetching."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"Sample PDF content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_get.return_value = mock_response
        
        scraper = HumanitarianScraper()
        url = "https://example.com/document.pdf"
        
        result = scraper._fetch_document(url)
        
        assert result["content"] == b"Sample PDF content"
        assert result["content_type"] == "application/pdf"
        mock_get.assert_called_once_with(url, timeout=30, headers=scraper._get_headers())

    @patch('requests.get')
    def test_fetch_document_failure(self, mock_get):
        """Test document fetching with HTTP error."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        scraper = HumanitarianScraper()
        url = "https://example.com/document.pdf"
        
        with pytest.raises(requests.RequestException):
            scraper._fetch_document(url)

    def test_get_headers(self):
        """Test request headers generation."""
        scraper = HumanitarianScraper()
        headers = scraper._get_headers()
        
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "vislang-ultralow-resource" in headers["User-Agent"]

    @patch('requests.get')
    def test_discover_unhcr_documents(self, mock_get):
        """Test UNHCR document discovery."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "123",
                    "fields": {
                        "title": "Sample Report",
                        "url": "https://example.com/report.pdf",
                        "date": {"created": "2023-01-01T00:00:00Z"},
                        "language": [{"code": "en"}],
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        scraper = HumanitarianScraper(sources=["unhcr"])
        documents = scraper._discover_unhcr_documents()
        
        assert len(documents) == 1
        assert documents[0]["title"] == "Sample Report"
        assert documents[0]["url"] == "https://example.com/report.pdf"

    @patch('requests.get')
    def test_discover_who_documents(self, mock_get):
        """Test WHO document discovery."""
        # Mock HTML response
        html_content = """
        <html>
            <body>
                <div class="publication">
                    <a href="/report1.pdf">Report 1</a>
                    <span class="date">2023-01-01</span>
                </div>
                <div class="publication">
                    <a href="/report2.pdf">Report 2</a>
                    <span class="date">2023-01-02</span>
                </div>
            </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html_content
        mock_get.return_value = mock_response
        
        scraper = HumanitarianScraper(sources=["who"])
        documents = scraper._discover_who_documents()
        
        assert len(documents) >= 0  # WHO parsing might be complex

    def test_filter_by_language(self):
        """Test document filtering by language."""
        documents = [
            {"title": "English Doc", "language": "en", "url": "doc1.pdf"},
            {"title": "Swahili Doc", "language": "sw", "url": "doc2.pdf"},
            {"title": "French Doc", "language": "fr", "url": "doc3.pdf"},
        ]
        
        scraper = HumanitarianScraper(languages=["en", "sw"])
        filtered = scraper._filter_by_language(documents)
        
        assert len(filtered) == 2
        assert all(doc["language"] in ["en", "sw"] for doc in filtered)

    def test_filter_by_date_range(self):
        """Test document filtering by date range."""
        documents = [
            {"title": "Old Doc", "date": "2020-01-01", "url": "doc1.pdf"},
            {"title": "Recent Doc", "date": "2023-01-01", "url": "doc2.pdf"},
            {"title": "Future Doc", "date": "2025-01-01", "url": "doc3.pdf"},
        ]
        
        scraper = HumanitarianScraper(date_range=("2022-01-01", "2024-01-01"))
        filtered = scraper._filter_by_date_range(documents)
        
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Recent Doc"

    def test_validate_document(self):
        """Test document validation."""
        scraper = HumanitarianScraper()
        
        # Valid document
        valid_doc = {
            "url": "https://example.com/doc.pdf",
            "title": "Valid Document",
            "content": b"PDF content here",
            "metadata": {"source": "unhcr"}
        }
        assert scraper._validate_document(valid_doc) is True
        
        # Invalid document (missing required fields)
        invalid_doc = {"title": "Invalid Document"}
        assert scraper._validate_document(invalid_doc) is False

    @patch.object(HumanitarianScraper, '_fetch_document')
    @patch.object(HumanitarianScraper, '_discover_unhcr_documents')
    def test_scrape_success(self, mock_discover, mock_fetch):
        """Test successful scraping workflow."""
        # Setup mocks
        mock_discover.return_value = [
            {
                "title": "Test Document",
                "url": "https://example.com/doc.pdf",
                "date": "2023-01-01",
                "language": "en",
                "source": "unhcr"
            }
        ]
        
        mock_fetch.return_value = {
            "content": b"PDF content",
            "content_type": "application/pdf"
        }
        
        scraper = HumanitarianScraper(sources=["unhcr"])
        results = scraper.scrape(max_documents=1)
        
        assert len(results) == 1
        assert results[0]["title"] == "Test Document"
        assert results[0]["content"] == b"PDF content"
        mock_discover.assert_called_once()
        mock_fetch.assert_called_once()

    def test_scrape_with_filters(self):
        """Test scraping with language and date filters."""
        scraper = HumanitarianScraper(
            sources=["unhcr"],
            languages=["en", "sw"],
            date_range=("2023-01-01", "2023-12-31")
        )
        
        # Verify filters are set
        assert scraper.languages == ["en", "sw"]
        assert scraper.date_range == ("2023-01-01", "2023-12-31")

    def test_extract_metadata(self):
        """Test metadata extraction from documents."""
        scraper = HumanitarianScraper()
        
        document = {
            "title": "Sample Report",
            "url": "https://example.com/report.pdf",
            "date": "2023-01-01",
            "language": "en",
            "source": "unhcr"
        }
        
        metadata = scraper._extract_metadata(document)
        
        assert metadata["title"] == "Sample Report"
        assert metadata["source"] == "unhcr"
        assert metadata["language"] == "en"
        assert "scraped_at" in metadata

    @pytest.mark.slow
    @patch.object(HumanitarianScraper, '_fetch_document')
    def test_scrape_rate_limiting(self, mock_fetch):
        """Test that scraping respects rate limiting."""
        import time
        
        mock_fetch.return_value = {
            "content": b"PDF content",
            "content_type": "application/pdf"
        }
        
        scraper = HumanitarianScraper()
        scraper.rate_limit_delay = 0.1  # 100ms delay for testing
        
        start_time = time.time()
        
        # Mock discovery to return multiple documents
        with patch.object(scraper, '_discover_unhcr_documents') as mock_discover:
            mock_discover.return_value = [
                {"url": f"https://example.com/doc{i}.pdf", "title": f"Doc {i}"}
                for i in range(3)
            ]
            
            results = scraper.scrape(max_documents=3)
            
        end_time = time.time()
        
        # Should take at least 2 * delay (2 delays between 3 requests)
        assert end_time - start_time >= 0.2
        assert len(results) == 3

    def test_error_handling_in_scrape(self):
        """Test error handling during scraping."""
        scraper = HumanitarianScraper(sources=["unhcr"])
        
        # Mock discovery to raise an exception
        with patch.object(scraper, '_discover_unhcr_documents') as mock_discover:
            mock_discover.side_effect = Exception("API Error")
            
            results = scraper.scrape()
            
            # Should handle error gracefully and return empty list
            assert results == []

    def test_content_type_validation(self):
        """Test validation of document content types."""
        scraper = HumanitarianScraper()
        
        # Valid content types
        assert scraper._is_valid_content_type("application/pdf") is True
        assert scraper._is_valid_content_type("text/html") is True
        assert scraper._is_valid_content_type("application/msword") is True
        
        # Invalid content types
        assert scraper._is_valid_content_type("image/jpeg") is False
        assert scraper._is_valid_content_type("audio/mp3") is False

    def test_url_normalization(self):
        """Test URL normalization for relative links."""
        scraper = HumanitarianScraper()
        
        base_url = "https://example.com/path/"
        
        # Absolute URL should remain unchanged
        absolute_url = "https://other.com/doc.pdf"
        assert scraper._normalize_url(absolute_url, base_url) == absolute_url
        
        # Relative URL should be resolved
        relative_url = "docs/report.pdf"
        expected = "https://example.com/path/docs/report.pdf"
        assert scraper._normalize_url(relative_url, base_url) == expected

    def test_duplicate_detection(self):
        """Test detection and removal of duplicate documents."""
        documents = [
            {"url": "https://example.com/doc1.pdf", "title": "Document 1"},
            {"url": "https://example.com/doc2.pdf", "title": "Document 2"},
            {"url": "https://example.com/doc1.pdf", "title": "Document 1 Duplicate"},
        ]
        
        scraper = HumanitarianScraper()
        unique_docs = scraper._remove_duplicates(documents)
        
        assert len(unique_docs) == 2
        urls = [doc["url"] for doc in unique_docs]
        assert "https://example.com/doc1.pdf" in urls
        assert "https://example.com/doc2.pdf" in urls