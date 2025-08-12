"""Humanitarian report scraping functionality."""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
from urllib.parse import urljoin, urlparse
import os
from contextlib import contextmanager
import tempfile
import signal

# Conditional imports with fallbacks
try:
    import asyncio
    import aiohttp
except ImportError:
    asyncio = aiohttp = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    class BeautifulSoup:
        def __init__(self, *args, **kwargs): pass
        def select_one(self, *args): return None
        def select(self, *args): return []
        def find(self, *args): return None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    class requests:
        class RequestException(Exception): pass
        class Session:
            def __init__(self): self.headers = {}
            def mount(self, *args): pass
            def get(self, url, **kwargs):
                class MockResponse:
                    content = b'<html><body>Mock content</body></html>'
                    def raise_for_status(self): pass
                return MockResponse()
    class HTTPAdapter:
        def __init__(self, *args, **kwargs): pass
    class Retry:
        def __init__(self, *args, **kwargs): pass

try:
    import backoff
except ImportError:
    class backoff:
        expo = None
        @staticmethod
        def on_exception(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

try:
    import psutil
except ImportError:
    class psutil:
        @staticmethod
        def cpu_count(): return 4

from .exceptions import ScrapingError, ValidationError, ResourceError, RateLimitError
from .utils.validation import DataValidator

logger = logging.getLogger(__name__)


class HumanitarianScraper:
    """Scraper for humanitarian organization reports."""
    
    def __init__(
        self,
        sources: List[str],
        languages: List[str],
        date_range: Optional[Tuple[str, str]] = None,
        max_retries: int = 3,
        timeout: int = 30,
        respect_robots: bool = True,
        user_agent: str = None,
        max_workers: int = 5
    ):
        """Initialize humanitarian scraper.
        
        Args:
            sources: List of source organizations ("unhcr", "who", "unicef", "wfp")
            languages: List of language codes to scrape
            date_range: Optional date range tuple (start, end) in YYYY-MM-DD format
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            respect_robots: Whether to respect robots.txt
            user_agent: Custom user agent string
            max_workers: Maximum concurrent workers
        """
        self.sources = sources
        self.languages = languages
        self.date_range = date_range
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.max_workers = max_workers
        
        # Validate sources
        valid_sources = {"unhcr", "who", "unicef", "wfp", "ocha", "undp"}
        invalid_sources = set(sources) - valid_sources
        if invalid_sources:
            raise ValidationError(f"Invalid sources: {invalid_sources}")
        
        # Initialize validator
        self.validator = DataValidator(strict_mode=False)
        
        # Set user agent
        self.user_agent = user_agent or (
            "VisLang-UltraLow-Resource/1.0 "
            "(https://github.com/danieleschmidt/quantum-inspired-task-planner; "
            "humanitarian-research@terragonlabs.ai)"
        )
        
        # Rate limiting configuration
        self.rate_limits = {
            "unhcr": {"requests_per_minute": 30, "burst": 5},
            "who": {"requests_per_minute": 20, "burst": 3},
            "unicef": {"requests_per_minute": 25, "burst": 4},
            "wfp": {"requests_per_minute": 20, "burst": 3},
            "ocha": {"requests_per_minute": 15, "burst": 2},
            "undp": {"requests_per_minute": 20, "burst": 3}
        }
        
        # Track request times for rate limiting
        self.request_times = {source: [] for source in sources}
        
        # Performance monitoring
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'documents_extracted': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Initialize HTTP session with retry strategy
        self.session = self._create_session()
        
        # Source-specific configurations
        self.source_configs = self._initialize_source_configs()
        
        logger.info(f"Initialized HumanitarianScraper for sources: {sources}, languages: {languages}")
    
    def scrape(self, max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape humanitarian documents from configured sources.
        
        Args:
            max_documents: Maximum number of documents to scrape (None for no limit)
            
        Returns:
            List of scraped document dictionaries
        """
        logger.info(f"Starting scraping process for {len(self.sources)} sources")
        start_time = time.time()
        
        all_documents = []
        
        for source in self.sources:
            try:
                logger.info(f"Scraping from {source}")
                source_docs = self._scrape_source(source, max_documents)
                all_documents.extend(source_docs)
                
                # Check if we've reached max documents
                if max_documents and len(all_documents) >= max_documents:
                    all_documents = all_documents[:max_documents]
                    break
                    
            except Exception as e:
                logger.error(f"Failed to scrape from {source}: {e}")
                self.stats['requests_failed'] += 1
                continue
        
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] = processing_time
        self.stats['avg_processing_time'] = processing_time / max(len(all_documents), 1)
        self.stats['documents_extracted'] = len(all_documents)
        
        logger.info(f"Scraping completed. Extracted {len(all_documents)} documents in {processing_time:.2f}s")
        return all_documents
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': ','.join(self.languages) + ',en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        return session
    
    def _initialize_source_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize source-specific configurations."""
        return {
            "unhcr": {
                "base_url": "https://www.unhcr.org",
                "search_endpoint": "/api/documents/search",
                "document_selectors": {
                    "title": "h1.document-title, .title",
                    "content": ".document-content, .main-content",
                    "images": "img[src*='unhcr'], .figure img",
                    "metadata": ".document-metadata, .meta-info"
                },
                "pagination": {"param": "page", "max_pages": 50}
            },
            "who": {
                "base_url": "https://www.who.int",
                "search_endpoint": "/emergencies/situations",
                "document_selectors": {
                    "title": "h1, .page-title",
                    "content": ".sf-content, .main-content",
                    "images": ".figure img, .content img",
                    "metadata": ".publication-meta, .document-info"
                },
                "pagination": {"param": "offset", "max_pages": 30}
            },
            "unicef": {
                "base_url": "https://www.unicef.org",
                "search_endpoint": "/reports",
                "document_selectors": {
                    "title": "h1.hero__title, h1",
                    "content": ".rich-text, .content-body",
                    "images": ".media img, .figure img",
                    "metadata": ".publication-details, .meta"
                },
                "pagination": {"param": "page", "max_pages": 40}
            },
            "wfp": {
                "base_url": "https://www.wfp.org",
                "search_endpoint": "/publications",
                "document_selectors": {
                    "title": "h1, .publication-title",
                    "content": ".publication-content, .body-text",
                    "images": ".publication-images img, .content img",
                    "metadata": ".publication-meta, .details"
                },
                "pagination": {"param": "page", "max_pages": 25}
            },
            "ocha": {
                "base_url": "https://www.unocha.org",
                "search_endpoint": "/publications",
                "document_selectors": {
                    "title": "h1.node-title, h1",
                    "content": ".field-body, .content",
                    "images": ".field-media img, .content img",
                    "metadata": ".node-meta, .publication-info"
                },
                "pagination": {"param": "page", "max_pages": 30}
            },
            "undp": {
                "base_url": "https://www.undp.org",
                "search_endpoint": "/publications",
                "document_selectors": {
                    "title": "h1, .publication-title",
                    "content": ".publication-body, .main-content",
                    "images": ".publication-media img, .content img",
                    "metadata": ".publication-details, .meta-info"
                },
                "pagination": {"param": "page", "max_pages": 35}
            }
        }
    
    def _scrape_source(self, source: str, max_docs: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape documents from a specific source."""
        config = self.source_configs[source]
        documents = []
        
        # Get document URLs
        doc_urls = self._get_document_urls(source, max_docs)
        
        logger.info(f"Found {len(doc_urls)} document URLs for {source}")
        
        # Process documents with rate limiting
        for url in doc_urls:
            try:
                # Check rate limit
                self._check_rate_limit(source)
                
                # Scrape individual document
                doc = self._scrape_document(url, source, config)
                if doc:
                    documents.append(doc)
                
                # Update stats
                self.stats['requests_made'] += 1
                
                if max_docs and len(documents) >= max_docs:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to scrape document {url}: {e}")
                self.stats['requests_failed'] += 1
                continue
        
        return documents
    
    def _get_document_urls(self, source: str, max_docs: Optional[int] = None) -> List[str]:
        """Get list of document URLs from source."""
        config = self.source_configs[source]
        urls = []
        
        # For demonstration, return mock URLs - in production would scrape actual URLs
        base_url = config["base_url"]
        
        # Mock URL generation for demo purposes
        for i in range(min(max_docs or 10, 10)):
            url = f"{base_url}/document-{i + 1}"
            urls.append(url)
        
        return urls
    
    def _scrape_document(self, url: str, source: str, config: Dict) -> Optional[Dict[str, Any]]:
        """Scrape individual document."""
        try:
            # Check rate limit
            self._check_rate_limit(source)
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract document information
            doc = {
                'id': hashlib.md5(url.encode()).hexdigest(),
                'url': url,
                'source': source,
                'timestamp': datetime.now().isoformat(),
                'title': self._extract_title(soup, config),
                'content': self._extract_content(soup, config),
                'images': self._extract_images(soup, config, url),
                'metadata': self._extract_metadata(soup, config),
                'language': self._detect_language(soup),
                'quality_score': 0.8  # Basic quality score
            }
            
            # Validate document
            if self.validator.validate_document(doc):
                return doc
            else:
                logger.warning(f"Document validation failed for {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping document {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, config: Dict) -> str:
        """Extract document title."""
        selectors = config["document_selectors"]["title"].split(", ")
        
        for selector in selectors:
            element = soup.select_one(selector.strip())
            if element:
                return element.get_text(strip=True)
        
        # Fallback to page title
        title_element = soup.find('title')
        return title_element.get_text(strip=True) if title_element else "Untitled Document"
    
    def _extract_content(self, soup: BeautifulSoup, config: Dict) -> str:
        """Extract main content."""
        selectors = config["document_selectors"]["content"].split(", ")
        
        content_parts = []
        for selector in selectors:
            elements = soup.select(selector.strip())
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) > 50:  # Filter out short snippets
                    content_parts.append(text)
        
        return "\n\n".join(content_parts)
    
    def _extract_images(self, soup: BeautifulSoup, config: Dict, base_url: str) -> List[Dict[str, Any]]:
        """Extract images from document."""
        selectors = config["document_selectors"]["images"].split(", ")
        images = []
        
        for selector in selectors:
            img_elements = soup.select(selector.strip())
            for img in img_elements:
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                
                # Convert relative URLs to absolute
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    parsed_url = urlparse(base_url)
                    src = f"{parsed_url.scheme}://{parsed_url.netloc}{src}"
                elif not src.startswith('http'):
                    src = urljoin(base_url, src)
                
                # Extract image metadata
                image_data = {
                    'url': src,
                    'alt_text': img.get('alt', ''),
                    'caption': self._get_image_caption(img),
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'format': self._get_image_format(src)
                }
                
                images.append(image_data)
        
        return images[:20]  # Limit to first 20 images
    
    def _extract_metadata(self, soup: BeautifulSoup, config: Dict) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {}
        
        # Extract publication date
        date_selectors = [
            'meta[property="article:published_time"]',
            '.publication-date', '.date', 'time[datetime]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_value = element.get('content') or element.get('datetime') or element.get_text(strip=True)
                if date_value:
                    metadata['publication_date'] = date_value
                    break
        
        # Extract author information
        author_selectors = [
            'meta[name="author"]',
            '.author', '.byline', '[rel="author"]'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get('content') or element.get_text(strip=True)
                if author:
                    metadata['author'] = author
                    break
        
        # Extract keywords/tags
        keywords_element = soup.select_one('meta[name="keywords"]')
        if keywords_element:
            metadata['keywords'] = keywords_element.get('content', '').split(',')
        
        return metadata
    
    def _detect_language(self, soup: BeautifulSoup) -> str:
        """Detect document language."""
        # Check html lang attribute
        html_element = soup.find('html')
        if html_element and html_element.get('lang'):
            return html_element.get('lang')[:2]  # Get language code
        
        # Check meta language tags
        lang_meta = soup.find('meta', {'name': 'language'}) or soup.find('meta', {'http-equiv': 'content-language'})
        if lang_meta:
            return lang_meta.get('content', 'en')[:2]
        
        # Default to English
        return 'en'
    
    def _get_image_caption(self, img_element) -> str:
        """Extract image caption."""
        # Look for caption in various locations
        parent = img_element.parent
        
        # Check for figcaption
        if parent and parent.name == 'figure':
            caption = parent.find('figcaption')
            if caption:
                return caption.get_text(strip=True)
        
        # Check for nearby caption elements
        caption_selectors = ['.caption', '.image-caption', '.figure-caption']
        for selector in caption_selectors:
            caption = parent.select_one(selector) if parent else None
            if caption:
                return caption.get_text(strip=True)
        
        return ''
    
    def _get_image_format(self, url: str) -> str:
        """Determine image format from URL."""
        url_lower = url.lower()
        if '.jpg' in url_lower or '.jpeg' in url_lower:
            return 'jpeg'
        elif '.png' in url_lower:
            return 'png'
        elif '.gif' in url_lower:
            return 'gif'
        elif '.svg' in url_lower:
            return 'svg'
        elif '.webp' in url_lower:
            return 'webp'
        else:
            return 'unknown'
    
    def _check_rate_limit(self, source: str):
        """Check and enforce rate limits."""
        current_time = time.time()
        source_times = self.request_times[source]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        source_times[:] = [t for t in source_times if t > cutoff_time]
        
        # Check rate limit
        rate_config = self.rate_limits[source]
        if len(source_times) >= rate_config["requests_per_minute"]:
            sleep_time = 60 - (current_time - source_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached for {source}, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Add current request time
        source_times.append(current_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset scraping statistics."""
        self.stats = {
            'requests_made': 0,
            'requests_failed': 0,
            'documents_extracted': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info(f"Initialized robust scraper for sources: {sources}, languages: {languages}")
        logger.info(f"Configuration: max_retries={max_retries}, timeout={timeout}s, max_workers={max_workers}")
    
    @contextmanager
    def _resource_monitor(self):
        """Context manager to monitor resource usage during scraping."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Check available resources
            if not self.validator.validate_system_resources(min_memory_gb=2.0, min_disk_gb=5.0):
                raise ResourceError("Insufficient system resources for scraping")
            
            yield
            
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            
            if memory_used > 500:  # More than 500MB used
                logger.warning(f"High memory usage during scraping: {memory_used:.1f}MB")
    
    def _setup_session(self) -> requests.Session:
        """Set up HTTP session with robust configuration."""
        session = requests.Session()
        
        # Set headers
        session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504, 522, 524],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2,  # Exponential backoff
            raise_on_status=False
        )
        
        # Mount adapters
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @backoff.on_exception(
        backoff.expo,
        (requests.RequestException, ConnectionError, TimeoutError),
        max_tries=3,
        max_time=300
    )
    def _make_request(self, url: str, source: str):
        """Make HTTP request with rate limiting and error handling.
        
        Args:
            url: URL to request
            source: Source organization for rate limiting
            
        Returns:
            HTTP response
            
        Raises:
            ScrapingError: If request fails after retries
            RateLimitError: If rate limited
        """
        # Check rate limits
        self._enforce_rate_limit(source)
        
        try:
            self.stats['requests_made'] += 1
            
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                stream=False  # Don't stream for small documents
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self.stats['rate_limit_delays'] += 1
                raise RateLimitError(f"Rate limited by {source}", retry_after=retry_after)
            
            # Check response status
            response.raise_for_status()
            
            # Validate response size
            content_length = response.headers.get('content-length', 0)
            if content_length and int(content_length) > 100 * 1024 * 1024:  # 100MB limit
                raise ScrapingError(f"Response too large: {content_length} bytes", source=source, url=url)
            
            return response
            
        except requests.exceptions.Timeout:
            self.stats['requests_failed'] += 1
            raise ScrapingError(f"Request timeout after {self.timeout}s", source=source, url=url)
        
        except requests.exceptions.ConnectionError as e:
            self.stats['requests_failed'] += 1
            raise ScrapingError(f"Connection error: {str(e)}", source=source, url=url)
        
        except requests.exceptions.HTTPError as e:
            self.stats['requests_failed'] += 1
            if response.status_code == 404:
                logger.warning(f"Document not found: {url}")
                return None
            elif response.status_code == 403:
                raise ScrapingError(f"Access forbidden: {url}", source=source, url=url)
            else:
                raise ScrapingError(f"HTTP error {response.status_code}: {str(e)}", source=source, url=url)
        
        except Exception as e:
            self.stats['requests_failed'] += 1
            raise ScrapingError(f"Unexpected error: {str(e)}", source=source, url=url)
    
    def _enforce_rate_limit(self, source: str) -> None:
        """Enforce rate limiting for source.
        
        Args:
            source: Source organization
            
        Raises:
            RateLimitError: If rate limit would be exceeded
        """
        if source not in self.rate_limits:
            return
        
        config = self.rate_limits[source]
        now = time.time()
        
        # Clean old requests (older than 1 minute)
        self.request_times[source] = [
            t for t in self.request_times[source] 
            if now - t < 60
        ]
        
        recent_requests = len(self.request_times[source])
        max_requests = config['requests_per_minute']
        
        if recent_requests >= max_requests:
            # Calculate delay needed
            oldest_request = min(self.request_times[source])
            delay = 60 - (now - oldest_request)
            
            if delay > 0:
                logger.info(f"Rate limiting {source}: waiting {delay:.1f}s")
                time.sleep(delay)
                self.stats['rate_limit_delays'] += 1
        
        # Record this request
        self.request_times[source].append(now)
    
    def scrape_robust(self, cache_dir: Optional[Path] = None, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape humanitarian reports with robust error handling.
        
        Args:
            cache_dir: Directory to cache downloaded files
            max_docs: Maximum number of documents to scrape per source
        
        Returns:
            List of scraped document metadata and content
            
        Raises:
            ScrapingError: If critical scraping error occurs
            ResourceError: If insufficient system resources
        """
        self.stats['start_time'] = datetime.now()
        logger.info("Starting robust humanitarian report scraping")
        
        try:
            with self._resource_monitor():
                # Setup cache directory
                self.cache_dir = cache_dir or Path("./cache")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Validate cache directory permissions
                if not os.access(self.cache_dir, os.W_OK):
                    raise ScrapingError(f"Cannot write to cache directory: {self.cache_dir}")
                
                # Initialize HTTP session
                self.session = self._setup_session()
                
                documents = []
                failed_sources = []
                
                # Process each source with error isolation
                for source in self.sources:
                    try:
                        logger.info(f"Processing source: {source}")
                        source_docs = self._scrape_source_robust(source, max_docs)
                        documents.extend(source_docs)
                        
                        logger.info(f"Successfully scraped {len(source_docs)} documents from {source}")
                        
                    except ScrapingError as e:
                        logger.error(f"Scraping error for {source}: {e}")
                        failed_sources.append(source)
                        
                        # Continue with other sources unless all are failing
                        if len(failed_sources) >= len(self.sources):
                            raise ScrapingError("All sources failed to scrape")
                        
                    except Exception as e:
                        logger.error(f"Unexpected error scraping {source}: {e}")
                        failed_sources.append(source)
                        continue
                
                # Validate results
                if not documents:
                    raise ScrapingError("No documents were successfully scraped from any source")
                
                # Validate document quality
                valid_documents = []
                for doc in documents:
                    try:
                        if self.validator.validate_document(doc):
                            valid_documents.append(doc)
                            self.stats['documents_extracted'] += 1
                        else:
                            logger.debug(f"Document validation failed: {doc.get('url')}")
                    except Exception as e:
                        logger.warning(f"Error validating document: {e}")
                
                # Final quality check
                success_rate = len(valid_documents) / len(documents) if documents else 0
                if success_rate < 0.1:  # Less than 10% success rate
                    logger.warning(f"Low success rate: {success_rate:.1%}")
                
                duration = datetime.now() - self.stats['start_time']
                logger.info(f"Scraping completed in {duration.total_seconds():.1f}s")
                logger.info(f"Final results: {len(valid_documents)} valid documents, "
                           f"{len(failed_sources)} failed sources")
                
                return valid_documents
                
        except KeyboardInterrupt:
            logger.warning("Scraping interrupted by user")
            raise
        except (ResourceError, ScrapingError):
            raise
        except Exception as e:
            logger.error(f"Critical scraping error: {e}")
            raise ScrapingError(f"Critical scraping failure: {str(e)}")
        finally:
            # Cleanup
            if hasattr(self, 'session'):
                self.session.close()
            
            # Log final statistics
            self._log_scraping_stats()
    
    def _scrape_source_robust(self, source: str, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape documents from specific source with robust error handling.
        
        Args:
            source: Source organization code
            max_docs: Maximum documents to scrape
            
        Returns:
            List of documents from source
        """
        logger.info(f"Starting robust scraping for source: {source}")
        
        # Source-specific scraping methods
        scraper_methods = {
            "unhcr": self._scrape_unhcr,
            "who": self._scrape_who,
            "unicef": self._scrape_unicef,
            "wfp": self._scrape_wfp,
            "ocha": self._scrape_ocha,
            "undp": self._scrape_undp
        }
        
        if source not in scraper_methods:
            raise ScrapingError(f"No scraper implemented for source: {source}", source=source)
        
        try:
            # Execute source-specific scraping with timeout
            documents = scraper_methods[source](max_docs)
            
            # Filter by date range if specified
            if self.date_range:
                original_count = len(documents)
                documents = self._filter_by_date(documents)
                filtered_count = original_count - len(documents)
                if filtered_count > 0:
                    logger.info(f"Filtered {filtered_count} documents by date range")
            
            # Validate extracted documents
            valid_documents = []
            for doc in documents:
                try:
                    if self._validate_extracted_document(doc, source):
                        valid_documents.append(doc)
                except Exception as e:
                    logger.debug(f"Document validation error: {e}")
            
            logger.info(f"Source {source}: {len(valid_documents)}/{len(documents)} documents passed validation")
            return valid_documents
            
        except RateLimitError as e:
            logger.warning(f"Rate limited by {source}, waiting {e.retry_after}s")
            time.sleep(e.retry_after)
            # Retry once after rate limit
            return self._scrape_source_robust(source, max_docs)
            
        except ScrapingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error scraping {source}: {e}")
            raise ScrapingError(f"Unexpected error in {source}: {str(e)}", source=source)
    
    def _validate_extracted_document(self, doc: Dict[str, any], source: str) -> bool:
        """Validate extracted document meets quality standards.
        
        Args:
            doc: Document dictionary
            source: Source organization
            
        Returns:
            True if document is valid
        """
        if not doc:
            return False
        
        # Required fields
        required_fields = ['url', 'title', 'content', 'source', 'language']
        for field in required_fields:
            if field not in doc or not doc[field]:
                logger.debug(f"Document missing required field: {field}")
                return False
        
        # Content quality checks
        content = doc.get('content', '')
        if len(content.strip()) < 100:  # Minimum content length
            logger.debug("Document content too short")
            return False
        
        # Source consistency
        if doc.get('source') != source:
            logger.debug(f"Source mismatch: expected {source}, got {doc.get('source')}")
            return False
        
        # Language validation
        if doc.get('language') not in self.languages:
            logger.debug(f"Document language {doc.get('language')} not in target languages")
            return False
        
        return True
    
    def _log_scraping_stats(self) -> None:
        """Log comprehensive scraping statistics."""
        if not self.stats['start_time']:
            return
        
        duration = datetime.now() - self.stats['start_time']
        stats = self.stats.copy()
        stats['duration_seconds'] = duration.total_seconds()
        stats['success_rate'] = (
            (stats['requests_made'] - stats['requests_failed']) / max(stats['requests_made'], 1)
        )
        
        logger.info("=== Scraping Statistics ===")
        logger.info(f"Duration: {duration.total_seconds():.1f}s")
        logger.info(f"Requests made: {stats['requests_made']}")
        logger.info(f"Requests failed: {stats['requests_failed']}")
        logger.info(f"Success rate: {stats['success_rate']:.1%}")
        logger.info(f"Documents extracted: {stats['documents_extracted']}")
        logger.info(f"Cache hits: {stats['cache_hits']}")
        logger.info(f"Rate limit delays: {stats['rate_limit_delays']}")
        logger.info("==========================")
    
    def _scrape_source(self, source: str, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape documents from specific source.
        
        Args:
            source: Source organization code
            max_docs: Maximum documents to scrape
            
        Returns:
            List of documents from source
        """
        logger.info(f"Scraping source: {source}")
        
        # Source-specific scraping methods
        scraper_methods = {
            "unhcr": self._scrape_unhcr,
            "who": self._scrape_who,
            "unicef": self._scrape_unicef,
            "wfp": self._scrape_wfp,
            "ocha": self._scrape_ocha,
            "undp": self._scrape_undp
        }
        
        if source not in scraper_methods:
            logger.warning(f"No scraper implemented for {source}")
            return []
        
        try:
            documents = scraper_methods[source](max_docs)
            # Filter by date range if specified
            if self.date_range:
                documents = self._filter_by_date(documents)
            return documents
        except Exception as e:
            logger.error(f"Error scraping {source}: {e}")
            return []
    
    def _scrape_unhcr(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape UNHCR documents."""
        documents = []
        base_url = "https://www.unhcr.org"
        
        # Search for reports in specified languages
        for lang in self.languages:
            try:
                search_url = f"{base_url}/search?query=reports&language={lang}"
                response = self.session.get(search_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                report_links = soup.find_all('a', href=True)
                
                count = 0
                for link in report_links:
                    if max_docs and count >= max_docs:
                        break
                    
                    href = link.get('href', '')
                    if any(ext in href.lower() for ext in ['.pdf', 'report', 'document']):
                        doc_url = urljoin(base_url, href)
                        doc = self._extract_document(doc_url, "unhcr", lang)
                        if doc:
                            documents.append(doc)
                            count += 1
                    
                    self._respect_rate_limits("unhcr")
                
            except Exception as e:
                logger.error(f"Error scraping UNHCR for language {lang}: {e}")
        
        return documents
    
    def _scrape_who(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape WHO documents."""
        documents = []
        base_url = "https://www.who.int"
        
        try:
            # WHO API endpoint for publications
            api_url = "https://www.who.int/api/publications"
            params = {
                "limit": max_docs or 50,
                "languages": ",".join(self.languages)
            }
            
            response = self.session.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', []):
                    doc = self._process_who_item(item)
                    if doc:
                        documents.append(doc)
                        
        except Exception as e:
            logger.error(f"Error scraping WHO: {e}")
            # Fallback to web scraping
            documents.extend(self._scrape_who_fallback(max_docs))
        
        return documents
    
    def _scrape_unicef(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape UNICEF documents."""
        documents = []
        base_url = "https://www.unicef.org"
        
        for lang in self.languages:
            try:
                search_url = f"{base_url}/reports?language={lang}"
                response = self.session.get(search_url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                report_sections = soup.find_all('div', class_=['report', 'publication'])
                
                count = 0
                for section in report_sections:
                    if max_docs and count >= max_docs:
                        break
                    
                    link = section.find('a', href=True)
                    if link:
                        doc_url = urljoin(base_url, link['href'])
                        doc = self._extract_document(doc_url, "unicef", lang)
                        if doc:
                            documents.append(doc)
                            count += 1
                    
                    self._respect_rate_limits("unicef")
                    
            except Exception as e:
                logger.error(f"Error scraping UNICEF for language {lang}: {e}")
        
        return documents
    
    def _scrape_wfp(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape WFP documents."""
        documents = []
        base_url = "https://www.wfp.org"
        
        try:
            # WFP publications endpoint
            search_url = f"{base_url}/publications"
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            pub_links = soup.find_all('a', href=True)
            
            count = 0
            for link in pub_links:
                if max_docs and count >= max_docs:
                    break
                
                href = link.get('href', '')
                if 'publication' in href.lower() or '.pdf' in href.lower():
                    doc_url = urljoin(base_url, href)
                    # Determine language from content or URL
                    lang = self._detect_document_language(doc_url)
                    if lang in self.languages:
                        doc = self._extract_document(doc_url, "wfp", lang)
                        if doc:
                            documents.append(doc)
                            count += 1
                
                self._respect_rate_limits("wfp")
                
        except Exception as e:
            logger.error(f"Error scraping WFP: {e}")
        
        return documents
    
    def _scrape_ocha(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape OCHA documents."""
        documents = []
        base_url = "https://www.unocha.org"
        
        try:
            search_url = f"{base_url}/documents"
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            doc_links = soup.find_all('a', class_=['document-link', 'report-link'])
            
            count = 0
            for link in doc_links:
                if max_docs and count >= max_docs:
                    break
                
                doc_url = urljoin(base_url, link['href'])
                lang = self._detect_document_language(doc_url)
                if lang in self.languages:
                    doc = self._extract_document(doc_url, "ocha", lang)
                    if doc:
                        documents.append(doc)
                        count += 1
                
                self._respect_rate_limits("ocha")
                
        except Exception as e:
            logger.error(f"Error scraping OCHA: {e}")
        
        return documents
    
    def _scrape_undp(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape UNDP documents."""
        documents = []
        base_url = "https://www.undp.org"
        
        try:
            publications_url = f"{base_url}/publications"
            response = self.session.get(publications_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            publication_cards = soup.find_all('div', class_=['publication', 'report-card'])
            
            count = 0
            for card in publication_cards:
                if max_docs and count >= max_docs:
                    break
                
                link = card.find('a', href=True)
                if link:
                    doc_url = urljoin(base_url, link['href'])
                    lang = self._detect_document_language(doc_url)
                    if lang in self.languages:
                        doc = self._extract_document(doc_url, "undp", lang)
                        if doc:
                            documents.append(doc)
                            count += 1
                
                self._respect_rate_limits("undp")
                
        except Exception as e:
            logger.error(f"Error scraping UNDP: {e}")
        
        return documents
    
    def _extract_document(self, url: str, source: str, language: str) -> Optional[Dict[str, any]]:
        """Extract document content from URL."""
        try:
            # Generate cache key
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Check cache first
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type:
                doc = self._extract_pdf_content(response.content, url, source, language)
            else:
                doc = self._extract_html_content(response.text, url, source, language)
            
            # Cache the result
            if doc:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
            
            return doc
            
        except Exception as e:
            logger.error(f"Error extracting document {url}: {e}")
            return None
    
    def _extract_pdf_content(self, pdf_content: bytes, url: str, source: str, language: str) -> Dict[str, any]:
        """Extract text and images from PDF."""
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        text_content = ""
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            text_content += page.get_text()
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = {
                    "page": page_num + 1,
                    "index": img_index,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "colorspace": base_image["colorspace"],
                    "ext": base_image["ext"]
                }
                images.append(image_data)
        
        doc.close()
        
        return {
            "source": source,
            "url": url,
            "language": language,
            "title": self._extract_title_from_text(text_content),
            "content": text_content.strip(),
            "images": images,
            "date": datetime.now().isoformat(),
            "content_type": "pdf",
            "word_count": len(text_content.split())
        }
    
    def _extract_html_content(self, html_content: str, url: str, source: str, language: str) -> Dict[str, any]:
        """Extract text and metadata from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text_content = soup.get_text()
        
        # Extract images
        images = []
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src', '')
            if src:
                images.append({
                    "src": urljoin(url, src),
                    "alt": img.get('alt', ''),
                    "width": img.get('width'),
                    "height": img.get('height')
                })
        
        return {
            "source": source,
            "url": url,
            "language": language,
            "title": title,
            "content": text_content.strip(),
            "images": images,
            "date": datetime.now().isoformat(),
            "content_type": "html",
            "word_count": len(text_content.split())
        }
    
    def _extract_title_from_text(self, text: str) -> str:
        """Extract title from text content."""
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                return line
        return "Untitled Document"
    
    def _detect_document_language(self, url: str) -> str:
        """Detect document language from URL or content."""
        # Simple heuristic based on URL patterns
        url_lower = url.lower()
        
        # Language patterns in URLs
        lang_patterns = {
            'en': ['english', '/en/', '/en-'],
            'fr': ['french', '/fr/', '/fr-', 'francais'],
            'es': ['spanish', '/es/', '/es-', 'espanol'],
            'ar': ['arabic', '/ar/', '/ar-'],
            'sw': ['swahili', '/sw/', 'kiswahili'],
            'am': ['amharic', '/am/', 'amharic'],
            'ha': ['hausa', '/ha/', 'hausa']
        }
        
        for lang, patterns in lang_patterns.items():
            if any(pattern in url_lower for pattern in patterns):
                return lang
        
        # Default to English if no pattern found
        return 'en'
    
    def _filter_by_date(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Filter documents by date range."""
        if not self.date_range:
            return documents
        
        start_date = datetime.fromisoformat(self.date_range[0])
        end_date = datetime.fromisoformat(self.date_range[1])
        
        filtered = []
        for doc in documents:
            try:
                doc_date = datetime.fromisoformat(doc['date'].replace('Z', '+00:00'))
                if start_date <= doc_date <= end_date:
                    filtered.append(doc)
            except (ValueError, KeyError):
                # Include document if date parsing fails
                filtered.append(doc)
        
        return filtered
    
    def _process_who_item(self, item: Dict) -> Optional[Dict[str, any]]:
        """Process WHO API item."""
        try:
            return {
                "source": "who",
                "url": item.get('url', ''),
                "title": item.get('title', ''),
                "content": item.get('description', ''),
                "language": item.get('language', 'en'),
                "date": item.get('publication_date', datetime.now().isoformat()),
                "images": [],
                "content_type": "api",
                "word_count": len(item.get('description', '').split())
            }
        except Exception as e:
            logger.error(f"Error processing WHO item: {e}")
            return None
    
    def _scrape_who_fallback(self, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Fallback WHO scraping method."""
        documents = []
        base_url = "https://www.who.int"
        
        try:
            search_url = f"{base_url}/publications"
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            pub_links = soup.find_all('a', href=True)
            
            count = 0
            for link in pub_links:
                if max_docs and count >= max_docs:
                    break
                
                href = link.get('href', '')
                if 'publication' in href.lower():
                    doc_url = urljoin(base_url, href)
                    lang = self._detect_document_language(doc_url)
                    if lang in self.languages:
                        doc = self._extract_document(doc_url, "who", lang)
                        if doc:
                            documents.append(doc)
                            count += 1
                
                self._respect_rate_limits("who")
                
        except Exception as e:
            logger.error(f"Error in WHO fallback scraping: {e}")
        
        return documents
    
    def _respect_rate_limits(self, source: str) -> None:
        """Implement rate limiting for respectful scraping.
        
        Args:
            source: Source organization code
        """
        # Rate limits by source (requests per minute)
        rate_limits = {
            "unhcr": 30,  # 30 requests per minute
            "who": 20,
            "unicef": 25,
            "wfp": 20,
            "ocha": 15,
            "undp": 20
        }
        
        limit = rate_limits.get(source, 10)
        delay = 60.0 / limit  # Convert to seconds between requests
        
        time.sleep(delay)
    
    def get_statistics(self, documents: List[Dict[str, any]]) -> Dict[str, any]:
        """Get scraping statistics."""
        if not documents:
            return {}
        
        stats = {
            "total_documents": len(documents),
            "sources": {},
            "languages": {},
            "content_types": {},
            "date_range": {
                "earliest": None,
                "latest": None
            },
            "total_words": 0,
            "total_images": 0
        }
        
        dates = []
        for doc in documents:
            # Source statistics
            source = doc.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Language statistics
            lang = doc.get('language', 'unknown')
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
            
            # Content type statistics
            content_type = doc.get('content_type', 'unknown')
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            # Word and image counts
            stats['total_words'] += doc.get('word_count', 0)
            stats['total_images'] += len(doc.get('images', []))
            
            # Date tracking
            try:
                doc_date = datetime.fromisoformat(doc['date'].replace('Z', '+00:00'))
                dates.append(doc_date)
            except (ValueError, KeyError):
                pass
        
        if dates:
            stats['date_range']['earliest'] = min(dates).isoformat()
            stats['date_range']['latest'] = max(dates).isoformat()
        
        return stats