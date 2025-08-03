"""Humanitarian report scraping functionality."""

from typing import Dict, List, Optional, Tuple, Union
import logging
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
from urllib.parse import urljoin, urlparse
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class HumanitarianScraper:
    """Scraper for humanitarian organization reports."""
    
    def __init__(
        self,
        sources: List[str],
        languages: List[str],
        date_range: Optional[Tuple[str, str]] = None
    ):
        """Initialize humanitarian scraper.
        
        Args:
            sources: List of source organizations ("unhcr", "who", "unicef", "wfp")
            languages: List of language codes to scrape
            date_range: Optional date range tuple (start, end) in YYYY-MM-DD format
        """
        self.sources = sources
        self.languages = languages
        self.date_range = date_range
        
        # Validate sources
        valid_sources = {"unhcr", "who", "unicef", "wfp", "ocha", "undp"}
        invalid_sources = set(sources) - valid_sources
        if invalid_sources:
            raise ValueError(f"Invalid sources: {invalid_sources}")
        
        logger.info(f"Initialized scraper for sources: {sources}, languages: {languages}")
    
    def scrape(self, cache_dir: Optional[Path] = None, max_docs: Optional[int] = None) -> List[Dict[str, any]]:
        """Scrape humanitarian reports.
        
        Args:
            cache_dir: Directory to cache downloaded files
            max_docs: Maximum number of documents to scrape per source
        
        Returns:
            List of scraped document metadata and content
        """
        logger.info("Starting humanitarian report scraping")
        
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        documents = []
        for source in self.sources:
            try:
                source_docs = self._scrape_source(source, max_docs)
                documents.extend(source_docs)
                logger.info(f"Scraped {len(source_docs)} documents from {source}")
            except Exception as e:
                logger.error(f"Failed to scrape {source}: {e}")
                continue
        
        logger.info(f"Total scraped {len(documents)} documents")
        return documents
    
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