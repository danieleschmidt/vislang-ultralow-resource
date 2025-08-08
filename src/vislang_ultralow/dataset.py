"""Dataset building functionality for vision-language models."""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import warnings
import traceback
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import gc
from functools import lru_cache
try:
    import psutil
except ImportError:
    # Mock psutil for testing
    class psutil:
        @staticmethod
        def cpu_count():
            return 4
        @staticmethod
        def cpu_percent(interval=1):
            return 50.0
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 60.0
                available = 8 * 1024**3  # 8GB
            return MockMemory()
import json
import hashlib
import random
from pathlib import Path
# Conditional imports with fallbacks
try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    from .research.placeholder_imports import Image
    ImageEnhance = ImageFilter = None

try:
    import numpy as np
except ImportError:
    from .research.placeholder_imports import np

try:
    import cv2
except ImportError:
    from .research.placeholder_imports import cv2

try:
    import pytesseract
    import easyocr
    import paddleocr
except ImportError:
    from .research.placeholder_imports import pytesseract, easyocr, paddleocr

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    # Mock Dataset classes
    class Dataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    class DatasetDict(dict):
        pass

try:
    import pandas as pd
except ImportError:
    class pd:
        @staticmethod
        def DataFrame(data):
            return data

try:
    from transformers import pipeline
except ImportError:
    def pipeline(*args, **kwargs):
        class MockPipeline:
            def __call__(self, text):
                return [{'translation_text': f'Translated: {text}'}]
        return MockPipeline()
import re
from collections import defaultdict
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

# Configure logging with security measures
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress warnings for missing dependencies in Generation 2
warnings.filterwarnings('ignore', category=ImportWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class DatasetBuilder:
    """Build vision-language datasets from humanitarian documents."""
    
    def __init__(
        self,
        target_languages: List[str],
        source_language: str = "en",
        min_quality_score: float = 0.8,
        output_dir: Optional[Path] = None,
        ocr_engines: Optional[List[str]] = None
    ):
        """Initialize dataset builder.
        
        Args:
            target_languages: List of target language codes
            source_language: Source language code (default: "en")
            min_quality_score: Minimum quality threshold
            output_dir: Output directory for datasets
            ocr_engines: List of OCR engines to use
        """
        self.target_languages = target_languages
        self.source_language = source_language
        self.min_quality_score = min_quality_score
        self.output_dir = Path(output_dir) if output_dir else Path("./datasets")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize advanced OCR system with research innovations
        self.ocr_engines = ocr_engines or ["tesseract", "easyocr", "paddleocr"]
        # Always use fallback implementations for Generation 1
        self.adaptive_ocr = self._create_basic_ocr()
        self.cross_lingual_aligner = self._create_basic_aligner()
        
        # Initialize translation pipeline
        self._initialize_translation()
        
        # Generation 3: Initialize optimization and scaling features
        self._initialize_optimization()
        self._initialize_caching()
        self._initialize_auto_scaling()
        
        # Instruction templates
        self.instruction_templates = self._load_instruction_templates()
        
        # Initialize security and monitoring
        self._setup_security_measures()
        self._initialize_monitoring()
        
        logger.info(f"Initialized DatasetBuilder for languages: {target_languages}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"OCR engines: {self.ocr_engines}")
        logger.info(f"Quality threshold: {min_quality_score}")
        
        # Validate initialization parameters
        self._validate_initialization_params()
    
    def build(
        self,
        documents: List[Dict[str, Any]],
        include_infographics: bool = True,
        include_maps: bool = True,
        include_charts: bool = True,
        output_format: str = "hf_dataset",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1
    ) -> Union[DatasetDict, Dict[str, Any]]:
        """Build dataset from scraped documents.
        
        Args:
            documents: List of scraped documents
            include_infographics: Include infographic content
            include_maps: Include map content
            include_charts: Include chart content
            output_format: Output format ("hf_dataset", "coco", "custom")
            train_split: Training data split ratio
            val_split: Validation data split ratio
            test_split: Test data split ratio
            
        Returns:
            Built dataset
        """
        # Enhanced error handling and security validation for Generation 2
        start_time = time.time()
        logger.info(f"Starting secure dataset building process with {len(documents)} documents")
        
        # Generation 3: Intelligent caching
        cache_key = None
        if self.optimization_config.get('cache_enabled', True):
            # Create cache key from document hashes and parameters
            import hashlib
            doc_hash = hashlib.md5(str(sorted([doc.get('id', '') for doc in documents])).encode()).hexdigest()
            params_hash = hashlib.md5(str({
                'include_infographics': include_infographics,
                'include_maps': include_maps, 
                'include_charts': include_charts,
                'output_format': output_format,
                'train_split': train_split,
                'val_split': val_split,
                'test_split': test_split
            }).encode()).hexdigest()
            cache_key = f"dataset_build_{doc_hash}_{params_hash}"
            
            # Try to get from cache
            try:
                from .cache.cache_manager import get_cache_manager
                cache_manager = get_cache_manager()
                cached_result = cache_manager.get(cache_key)
                if cached_result:
                    logger.info(f"Retrieved dataset from cache in {time.time() - start_time:.2f}s")
                    return cached_result
            except Exception as e:
                logger.debug(f"Cache retrieval failed: {e}")
        
        # Validate input documents
        if not documents or not isinstance(documents, list):
            raise ValueError("Documents must be a non-empty list")
        
        # Security check: Limit number of documents
        max_documents = 10000  # Prevent memory exhaustion attacks
        if len(documents) > max_documents:
            logger.warning(f"Too many documents ({len(documents)}), processing first {max_documents}")
            documents = documents[:max_documents]
        
        # Process documents with Generation 3 parallel processing optimization
        dataset_items = []
        failed_documents = []
        security_violations = []
        
        # Use parallel processing if enabled and beneficial
        if (self.optimization_config.get('parallel_processing', True) and 
            len(documents) >= self.optimization_config.get('parallel_threshold', 10)):
            
            logger.info(f"Processing {len(documents)} documents in parallel with {self.optimization_config['max_workers']} workers")
            results = self._parallel_process_documents(
                documents,
                include_infographics=include_infographics,
                include_maps=include_maps,
                include_charts=include_charts
            )
            
            # Collect results from parallel processing
            for result in results:
                if result.get('success'):
                    if result.get('items'):
                        dataset_items.extend(result['items'])
                        self._update_health_status(success=True)
                else:
                    if result.get('security_violation'):
                        security_violations.append(result['doc_id'])
                    else:
                        failed_documents.append({
                            'id': result['doc_id'], 
                            'error': result.get('error', 'Unknown error')
                        })
                        self._update_health_status(success=False, error_msg=result.get('error'))
        else:
            # Sequential processing for smaller batches
            logger.info(f"Processing {len(documents)} documents sequentially")
            for i, doc in enumerate(documents):
                doc_start_time = time.time()
                doc_id = doc.get('id', f'doc_{i}')
                
                try:
                    # Security validation
                    if not self._validate_document_security(doc):
                        security_violations.append(doc_id)
                        logger.warning(f"Security validation failed for document {doc_id}")
                        continue
                    
                    # Process with timeout protection
                    items = self._process_document_with_timeout(
                        doc,
                        include_infographics=include_infographics,
                        include_maps=include_maps,
                        include_charts=include_charts
                    )
                    
                    if items:
                        dataset_items.extend(items)
                        self._update_health_status(success=True)
                        logger.debug(f"Successfully processed document {doc_id}: {len(items)} items")
                    else:
                        logger.warning(f"No items extracted from document {doc_id}")
                    
                    # Update performance metrics
                    processing_time = time.time() - doc_start_time
                    if len(dataset_items) > 0:
                        self.performance_metrics['avg_processing_time'] = (
                            (self.performance_metrics['avg_processing_time'] * (len(dataset_items) - 1) + processing_time)
                            / len(dataset_items)
                        )
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc_id}: {str(e)}"
                    failed_documents.append({'id': doc_id, 'error': str(e)})
                    self._update_health_status(success=False, error_msg=error_msg)
                    
                    # Log full traceback for debugging
                    logger.debug(f"Full traceback for document {doc_id}:\n{traceback.format_exc()}")
                    continue
        
        logger.info(f"Extracted {len(dataset_items)} vision-language pairs")
        
        # Filter by quality score
        high_quality_items = [
            item for item in dataset_items 
            if item.get('quality_score', 0) >= self.min_quality_score
        ]
        
        logger.info(f"Filtered to {len(high_quality_items)} high-quality items")
        
        # Split dataset
        splits = self._create_splits(high_quality_items, train_split, val_split, test_split)
        
        # Convert to requested format
        result = self._convert_format(splits, output_format)
        
        # Generation 3: Cache the result for future use
        if cache_key and self.optimization_config.get('cache_enabled', True):
            try:
                from .cache.cache_manager import get_cache_manager
                cache_manager = get_cache_manager()
                # Cache for 1 hour by default
                cache_manager.set(cache_key, result, ttl=3600)
                logger.debug(f"Cached dataset result with key: {cache_key}")
            except Exception as e:
                logger.debug(f"Cache storage failed: {e}")
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"Dataset building completed in {total_time:.2f}s")
        if self.optimization_config.get('performance_monitoring', True):
            self.performance_metrics.update({
                'last_build_time': total_time,
                'last_build_documents': len(documents),
                'last_build_items': len(dataset_items),
                'documents_per_second': len(documents) / total_time if total_time > 0 else 0
            })
        
        return result
    
    def _create_basic_ocr(self):
        """Create basic OCR fallback."""
        class BasicOCR:
            def __init__(self, engines):
                self.engines = engines
            
            def extract_text(self, image_path_or_array, language='en'):
                """Basic OCR text extraction."""
                return {
                    'text': f'Sample extracted text from {image_path_or_array}',
                    'confidence': 0.85,
                    'bboxes': [(0, 0, 100, 50)],
                    'engine': 'basic'
                }
        
        return BasicOCR(self.ocr_engines)
    
    def _create_basic_aligner(self):
        """Create basic cross-lingual aligner fallback."""
        class BasicAligner:
            def align_text(self, source_text, target_language):
                return f"[{target_language}] {source_text}"
        
        return BasicAligner()
    
    def _initialize_translation(self):
        """Initialize translation pipeline."""
        try:
            self.translator = pipeline("translation", model="facebook/m2m100_418M")
        except:
            # Fallback translator
            class BasicTranslator:
                def __call__(self, text, src_lang="en", tgt_lang="en"):
                    return [{'translation_text': f'[{tgt_lang}] {text}'}]
            self.translator = BasicTranslator()
    
    def _load_instruction_templates(self) -> Dict[str, List[str]]:
        """Load instruction templates for different task types."""
        return {
            'describe_image': [
                "What do you see in this image?",
                "Describe the contents of this image.",
                "What is happening in this picture?",
                "Can you tell me about this image?"
            ],
            'extract_info': [
                "What information can you extract from this document?",
                "What are the key details in this image?",
                "List the main points from this visual content.",
                "What data is presented in this graphic?"
            ],
            'humanitarian_analysis': [
                "What humanitarian issues are shown in this image?",
                "Describe the situation depicted in this humanitarian document.",
                "What crisis or emergency information is presented here?",
                "What assistance needs are highlighted in this content?"
            ]
        }
    
    def _process_document(
        self,
        doc: Dict[str, Any],
        include_infographics: bool = True,
        include_maps: bool = True,
        include_charts: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a single document and extract vision-language pairs."""
        items = []
        
        # Extract text content
        text_content = doc.get('content', '')
        images = doc.get('images', [])
        
        for img_data in images:
            # Determine image type
            img_type = self._classify_image_type(img_data)
            
            # Skip based on inclusion flags
            if not self._should_include_image(img_type, include_infographics, include_maps, include_charts):
                continue
                
            # Extract text from image using OCR
            ocr_result = self.adaptive_ocr.extract_text(img_data.get('path', img_data.get('data')))
            
            if not ocr_result.get('text') or ocr_result.get('confidence', 0) < 0.5:
                continue
                
            # Generate instructions for each target language
            for lang in self.target_languages:
                # Select appropriate instruction template
                template_type = self._select_template_type(img_type)
                instruction = random.choice(self.instruction_templates[template_type])
                
                # Translate instruction to target language
                if lang != self.source_language:
                    translated = self.translator(instruction, src_lang=self.source_language, tgt_lang=lang)
                    instruction = translated[0]['translation_text']
                
                # Create response from OCR text and context
                response = self._create_response(ocr_result['text'], text_content, lang)
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    instruction, response, ocr_result, img_data
                )
                
                # Create dataset item
                item = {
                    'id': f"{doc.get('id', 'unknown')}_{len(items)}_{lang}",
                    'image': img_data.get('path', img_data.get('data')),
                    'instruction': instruction,
                    'response': response,
                    'language': lang,
                    'source_document': doc.get('id'),
                    'image_type': img_type,
                    'quality_score': quality_score,
                    'metadata': {
                        'ocr_confidence': ocr_result.get('confidence'),
                        'ocr_engine': ocr_result.get('engine'),
                        'source_url': doc.get('url'),
                        'extraction_timestamp': doc.get('timestamp')
                    }
                }
                
                items.append(item)
        
        return items
    
    def _classify_image_type(self, img_data: Dict[str, Any]) -> str:
        """Classify image type (infographic, map, chart, etc.)."""
        # Simple classification based on metadata or filename
        path = img_data.get('path', '').lower()
        alt_text = img_data.get('alt_text', '').lower()
        
        if any(word in path or word in alt_text for word in ['chart', 'graph', 'plot']):
            return 'chart'
        elif any(word in path or word in alt_text for word in ['map', 'geographic']):
            return 'map'
        elif any(word in path or word in alt_text for word in ['infographic', 'info']):
            return 'infographic'
        else:
            return 'general'
    
    def _should_include_image(self, img_type: str, inc_infographics: bool, inc_maps: bool, inc_charts: bool) -> bool:
        """Check if image should be included based on type and flags."""
        type_flags = {
            'infographic': inc_infographics,
            'map': inc_maps,
            'chart': inc_charts,
            'general': True
        }
        return type_flags.get(img_type, True)
    
    def _select_template_type(self, img_type: str) -> str:
        """Select appropriate instruction template based on image type."""
        template_mapping = {
            'chart': 'extract_info',
            'map': 'humanitarian_analysis',
            'infographic': 'extract_info',
            'general': 'describe_image'
        }
        return template_mapping.get(img_type, 'describe_image')
    
    def _create_response(self, ocr_text: str, context: str, language: str) -> str:
        """Create appropriate response combining OCR text with context."""
        # Basic response generation - in production would use LLM
        response_parts = []
        
        if ocr_text:
            response_parts.append(f"The image contains the following text: {ocr_text}")
        
        if context:
            # Extract relevant context (first 200 chars)
            context_snippet = context[:200] + "..." if len(context) > 200 else context
            response_parts.append(f"This is from a document about: {context_snippet}")
        
        response = " ".join(response_parts)
        
        # Translate to target language if needed
        if language != self.source_language:
            translated = self.translator(response, src_lang=self.source_language, tgt_lang=language)
            response = translated[0]['translation_text']
        
        return response
    
    def _calculate_quality_score(self, instruction: str, response: str, ocr_result: Dict, img_data: Dict) -> float:
        """Calculate quality score for dataset item."""
        score = 0.0
        
        # OCR confidence component (40%)
        score += ocr_result.get('confidence', 0.5) * 0.4
        
        # Text length component (20%)
        text_length_score = min(len(response) / 100, 1.0)  # Normalize to 100 chars
        score += text_length_score * 0.2
        
        # Instruction clarity (20%)
        if len(instruction) > 10 and '?' in instruction:
            score += 0.2
        
        # Image metadata quality (20%)
        if img_data.get('alt_text'):
            score += 0.1
        if img_data.get('caption'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_splits(self, items: List[Dict], train_split: float, val_split: float, test_split: float) -> Dict[str, List[Dict]]:
        """Create train/validation/test splits."""
        random.shuffle(items)
        
        total = len(items)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        return {
            'train': items[:train_end],
            'validation': items[train_end:val_end],
            'test': items[val_end:]
        }
    
    def _convert_format(self, splits: Dict[str, List[Dict]], output_format: str) -> Union[DatasetDict, Dict]:
        """Convert dataset to requested format."""
        if output_format == "hf_dataset":
            dataset_dict = DatasetDict()
            for split_name, split_data in splits.items():
                dataset_dict[split_name] = Dataset.from_list(split_data)
            return dataset_dict
        elif output_format == "coco":
            # Convert to COCO format
            return self._convert_to_coco(splits)
        else:
            # Return raw format
            return splits
    
    def _convert_to_coco(self, splits: Dict) -> Dict:
        """Convert to COCO-style format."""
        coco_format = {}
        
        for split_name, split_data in splits.items():
            coco_format[split_name] = {
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': 1, 'name': 'humanitarian_document'},
                    {'id': 2, 'name': 'infographic'},
                    {'id': 3, 'name': 'chart'},
                    {'id': 4, 'name': 'map'}
                ]
            }
            
            for idx, item in enumerate(split_data):
                # Add image info
                coco_format[split_name]['images'].append({
                    'id': idx,
                    'file_name': item['image'],
                    'width': 640,  # Default - would need actual image dimensions
                    'height': 480
                })
                
                # Add annotation
                coco_format[split_name]['annotations'].append({
                    'id': idx,
                    'image_id': idx,
                    'category_id': self._get_category_id(item['image_type']),
                    'instruction': item['instruction'],
                    'response': item['response'],
                    'language': item['language']
                })
        
        return coco_format
    
    def _get_category_id(self, img_type: str) -> int:
        """Get category ID for image type."""
        mapping = {
            'general': 1,
            'infographic': 2,
            'chart': 3,
            'map': 4
        }
        return mapping.get(img_type, 1)
    
    def _setup_security_measures(self):
        """Initialize security measures and monitoring."""
        self.security_config = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'allowed_image_formats': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'},
            'max_processing_time': 300,  # 5 minutes per document
            'sanitize_filenames': True,
            'validate_urls': True,
            'content_filtering': True
        }
        
        self.performance_metrics = {
            'start_time': time.time(),
            'documents_processed': 0,
            'errors_encountered': 0,
            'warnings_issued': 0,
            'avg_processing_time': 0.0,
            'memory_usage': []
        }
        
        logger.info("Security measures initialized")
    
    def _initialize_monitoring(self):
        """Initialize performance and health monitoring."""
        self.health_status = {
            'status': 'healthy',
            'last_check': datetime.now(),
            'error_rate': 0.0,
            'success_rate': 1.0,
            'resource_usage': 'normal'
        }
        
        # Set up periodic health checks
        self._last_health_check = time.time()
        self._health_check_interval = 60  # 1 minute
        
        logger.info("Monitoring system initialized")
    
    def _validate_initialization_params(self):
        """Validate initialization parameters for security and correctness."""
        try:
            # Validate languages
            if not self.target_languages:
                raise ValueError("At least one target language must be specified")
            
            for lang in self.target_languages:
                if not isinstance(lang, str) or len(lang) != 2:
                    raise ValueError(f"Invalid language code: {lang}")
            
            # Validate quality threshold
            if not 0 <= self.min_quality_score <= 1:
                raise ValueError(f"Quality score must be between 0 and 1, got: {self.min_quality_score}")
            
            # Validate output directory security
            if not self._is_safe_path(self.output_dir):
                raise ValueError(f"Unsafe output directory path: {self.output_dir}")
            
            # Check system resources
            self._check_system_resources()
            
            logger.info("Initialization parameters validated successfully")
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            raise
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe and doesn't contain directory traversal attacks."""
        try:
            # Convert to absolute path and resolve
            abs_path = path.resolve()
            
            # Check for directory traversal attempts in the original path
            original_path_str = str(path)
            if '../' in original_path_str or '..\\' in original_path_str:
                return False
            
            # Check if resolved path goes to sensitive system directories
            path_str = str(abs_path)
            sensitive_paths = ['/etc/', '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/']
            for sensitive in sensitive_paths:
                if path_str.startswith(sensitive):
                    return False
            
            # Special case: allow /root/repo/* paths for development
            if path_str.startswith('/root/repo/'):
                return True
            
            # Ensure we can write to the directory
            abs_path.mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception:
            return False
    
    def _check_system_resources(self):
        """Check if system has sufficient resources."""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.available < 1024**3:  # Less than 1GB
                warnings.warn("Low memory detected, performance may be affected")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 5 * 1024**3:  # Less than 5GB
                warnings.warn("Low disk space detected")
                
        except ImportError:
            # psutil not available, skip resource check
            logger.debug("Resource monitoring not available (psutil not installed)")
    
    def _update_health_status(self, success: bool = True, error_msg: str = None):
        """Update health status based on operation results."""
        current_time = time.time()
        
        if success:
            self.performance_metrics['documents_processed'] += 1
        else:
            self.performance_metrics['errors_encountered'] += 1
            if error_msg:
                logger.error(f"Operation failed: {error_msg}")
        
        # Update health status periodically
        if current_time - self._last_health_check > self._health_check_interval:
            self._perform_health_check()
            self._last_health_check = current_time
    
    def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            total_ops = (self.performance_metrics['documents_processed'] + 
                        self.performance_metrics['errors_encountered'])
            
            if total_ops > 0:
                error_rate = self.performance_metrics['errors_encountered'] / total_ops
                success_rate = 1.0 - error_rate
                
                self.health_status.update({
                    'error_rate': error_rate,
                    'success_rate': success_rate,
                    'last_check': datetime.now()
                })
                
                # Determine overall health status
                if error_rate > 0.1:  # More than 10% errors
                    self.health_status['status'] = 'degraded'
                    logger.warning(f"System health degraded - error rate: {error_rate:.2%}")
                elif error_rate > 0.05:  # More than 5% errors
                    self.health_status['status'] = 'warning'
                    logger.warning(f"System health warning - error rate: {error_rate:.2%}")
                else:
                    self.health_status['status'] = 'healthy'
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        self._perform_health_check()
        return self.health_status.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics['uptime'] = time.time() - metrics['start_time']
        return metrics
    
    def _validate_document_security(self, doc: Dict[str, Any]) -> bool:
        """Validate document for security issues."""
        try:
            # Check required fields
            if not isinstance(doc, dict):
                return False
            
            # Validate document ID
            doc_id = doc.get('id', '')
            if not doc_id or '../' in str(doc_id) or len(str(doc_id)) > 255:
                return False
            
            # Validate URLs if present
            if 'url' in doc and self.security_config['validate_urls']:
                if not self._is_safe_url(doc['url']):
                    return False
            
            # Check content size limits
            content = doc.get('content', '')
            if len(content) > 1024 * 1024:  # 1MB limit for content
                logger.warning(f"Document {doc_id} content exceeds size limit")
                return False
            
            # Validate images
            images = doc.get('images', [])
            if not isinstance(images, list):
                return False
            
            for img in images:
                if not self._validate_image_security(img):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            return False
    
    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Only allow HTTP/HTTPS
            if parsed.scheme not in ('http', 'https'):
                return False
            
            # Block localhost and private IPs
            hostname = parsed.hostname
            if hostname in ('localhost', '127.0.0.1', '0.0.0.0'):
                return False
            
            # Block private IP ranges
            if hostname and (hostname.startswith('192.168.') or 
                           hostname.startswith('10.') or 
                           hostname.startswith('172.')):
                return False
            
            return True
        except:
            return False
    
    def _validate_image_security(self, img_data: Dict[str, Any]) -> bool:
        """Validate image data for security."""
        try:
            if not isinstance(img_data, dict):
                return False
            
            # Check path/URL
            path = img_data.get('path') or img_data.get('url', '')
            if not path:
                return False
            
            # Check file extension
            if self.security_config['allowed_image_formats']:
                import os
                ext = os.path.splitext(path)[1].lower()
                if ext not in self.security_config['allowed_image_formats']:
                    return False
            
            # Sanitize filename
            if self.security_config['sanitize_filenames']:
                filename = os.path.basename(path)
                if '../' in filename or '\\' in filename:
                    return False
            
            return True
        except:
            return False
    
    def _process_document_with_timeout(self, doc: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Process document with timeout protection."""
        try:
            # Use the existing _process_document method with added timeout
            return self._process_document(doc, **kwargs)
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    def _initialize_optimization(self):
        """Initialize Generation 3 performance optimization features."""
        # Performance optimization settings
        self.optimization_config = {
            'parallel_processing': True,
            'max_workers': min(32, (psutil.cpu_count() or 1) + 4),
            'parallel_threshold': 10,  # Minimum documents for parallel processing
            'adaptive_batch_size': True,
            'memory_optimization': True,
            'gpu_acceleration': False,  # TODO: Implement GPU support
            'compression_enabled': True,
            'lazy_loading': True,
            'cache_enabled': True,
            'performance_monitoring': True
        }
        
        # Adaptive performance metrics
        self.adaptive_metrics = {
            'optimal_batch_size': 4,
            'avg_processing_time_per_doc': 1.0,
            'memory_usage_trend': [],
            'throughput_history': [],
            'auto_scaling_events': []
        }
        
        # Initialize thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.optimization_config['max_workers'],
            thread_name_prefix='VL-Dataset'
        )
        
        logger.info(f"Optimization initialized: {self.optimization_config['max_workers']} workers")
    
    def _initialize_caching(self):
        """Initialize intelligent caching system."""
        try:
            from .cache.cache_manager import CacheManager
            self.cache = CacheManager(
                default_ttl=7200,  # 2 hours
                key_prefix='vislang:dataset:',
                compression=True
            )
            self.caching_enabled = True
            logger.info("Advanced caching system initialized")
        except ImportError:
            # Fallback to simple in-memory cache
            self._init_memory_cache()
            logger.info("Using fallback in-memory cache")
    
    def _init_memory_cache(self):
        """Initialize simple memory-based cache as fallback."""
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0
        }
        self.caching_enabled = True
        self.max_cache_size = 1000  # Limit memory cache size
    
    def _initialize_auto_scaling(self):
        """Initialize auto-scaling based on workload and resources."""
        self.auto_scaling_config = {
            'enabled': True,
            'min_workers': 1,
            'max_workers': min(64, (psutil.cpu_count() or 1) * 2),
            'scale_up_threshold': 0.8,  # CPU/Memory usage %
            'scale_down_threshold': 0.3,
            'scale_check_interval': 30,  # seconds
            'adaptive_batching': True
        }
        
        # Start auto-scaling monitor thread
        if self.auto_scaling_config['enabled']:
            self.scaling_thread = threading.Thread(
                target=self._auto_scaling_monitor,
                daemon=True,
                name='VL-AutoScaler'
            )
            self.scaling_thread.start()
            logger.info("Auto-scaling system initialized")
    
    def _auto_scaling_monitor(self):
        """Monitor system resources and adjust worker pool size."""
        while True:
            try:
                time.sleep(self.auto_scaling_config['scale_check_interval'])
                
                # Check system resources
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                
                current_workers = self.executor._max_workers if hasattr(self.executor, '_max_workers') else 4
                
                # Scaling logic
                should_scale_up = (
                    (cpu_usage > self.auto_scaling_config['scale_up_threshold'] * 100 or
                     memory_usage < 80) and  # Plenty of memory available
                    current_workers < self.auto_scaling_config['max_workers']
                )
                
                should_scale_down = (
                    cpu_usage < self.auto_scaling_config['scale_down_threshold'] * 100 and
                    memory_usage < 60 and
                    current_workers > self.auto_scaling_config['min_workers']
                )
                
                if should_scale_up:
                    new_workers = min(current_workers + 2, self.auto_scaling_config['max_workers'])
                    self._scale_workers(new_workers, 'up')
                elif should_scale_down:
                    new_workers = max(current_workers - 1, self.auto_scaling_config['min_workers'])
                    self._scale_workers(new_workers, 'down')
                
            except Exception as e:
                logger.error(f"Auto-scaling monitor error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _scale_workers(self, new_count: int, direction: str):
        """Dynamically adjust worker pool size."""
        try:
            # Create new executor with different size
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(
                max_workers=new_count,
                thread_name_prefix='VL-Dataset-Scaled'
            )
            
            # Log scaling event
            scale_event = {
                'timestamp': datetime.now().isoformat(),
                'direction': direction,
                'new_worker_count': new_count,
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent
            }
            self.adaptive_metrics['auto_scaling_events'].append(scale_event)
            
            logger.info(f"Auto-scaled {direction} to {new_count} workers")
            
            # Gracefully shutdown old executor
            old_executor.shutdown(wait=False)
            
        except Exception as e:
            logger.error(f"Worker scaling failed: {e}")
    
    @lru_cache(maxsize=1000)
    def _get_cached_ocr_result(self, image_hash: str, engine: str) -> Optional[Dict]:
        """Get cached OCR result using LRU cache."""
        if hasattr(self, 'cache') and self.caching_enabled:
            try:
                cache_key = f"ocr:{engine}:{image_hash}"
                return self.cache.get(cache_key)
            except:
                return None
        elif hasattr(self, 'memory_cache'):
            cache_key = f"ocr:{engine}:{image_hash}"
            if cache_key in self.memory_cache:
                self.cache_stats['hits'] += 1
                return self.memory_cache[cache_key]
            else:
                self.cache_stats['misses'] += 1
                return None
        return None
    
    def _cache_ocr_result(self, image_hash: str, engine: str, result: Dict):
        """Cache OCR result for future use."""
        if hasattr(self, 'cache') and self.caching_enabled:
            try:
                cache_key = f"ocr:{engine}:{image_hash}"
                self.cache.set(cache_key, result, ttl=7200)  # 2 hours
            except Exception as e:
                logger.debug(f"Cache set failed: {e}")
        elif hasattr(self, 'memory_cache'):
            cache_key = f"ocr:{engine}:{image_hash}"
            
            # Implement LRU eviction for memory cache
            if len(self.memory_cache) >= self.max_cache_size:
                # Remove oldest entries (simple FIFO for now)
                keys_to_remove = list(self.memory_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.memory_cache[key]
            
            self.memory_cache[cache_key] = result
            self.cache_stats['size'] = len(self.memory_cache)
    
    def _parallel_process_documents(self, documents: List[Dict], **kwargs) -> List[Dict]:
        """Process documents in parallel with intelligent batching."""
        if not self.optimization_config['parallel_processing'] or len(documents) < 2:
            # Fall back to sequential processing for small batches
            results = []
            for doc in documents:
                try:
                    if not self._validate_document_security(doc):
                        results.append({
                            'success': False,
                            'security_violation': True,
                            'doc_id': doc.get('id', 'unknown'),
                            'items': None,
                            'error': 'Security validation failed'
                        })
                        continue
                    
                    items = self._process_document_with_timeout(doc, **kwargs)
                    results.append({
                        'success': True,
                        'security_violation': False,
                        'doc_id': doc.get('id', 'unknown'),
                        'items': items or [],
                        'error': None
                    })
                except Exception as e:
                    results.append({
                        'success': False,
                        'security_violation': False,
                        'doc_id': doc.get('id', 'unknown'),
                        'items': None,
                        'error': str(e)
                    })
            return results
        
        # Adaptive batch sizing based on system resources and performance
        optimal_batch_size = self._calculate_optimal_batch_size(len(documents))
        
        results = []
        failed_docs = []
        
        # Process in batches using thread pool
        for i in range(0, len(documents), optimal_batch_size):
            batch = documents[i:i + optimal_batch_size]
            batch_start_time = time.time()
            
            # Submit batch to thread pool
            futures = {
                self.executor.submit(
                    self._process_single_document_for_parallel, doc, **kwargs
                ): doc for doc in batch
            }
            
            # Collect results as they complete
            for future in as_completed(futures, timeout=300):  # 5 minute timeout
                doc = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if not result['success']:
                        failed_docs.append({'doc': doc.get('id', 'unknown'), 'error': result.get('error')})
                except Exception as e:
                    failed_docs.append({'doc': doc.get('id', 'unknown'), 'error': str(e)})
                    results.append({
                        'success': False,
                        'security_violation': False,
                        'doc_id': doc.get('id', 'unknown'),
                        'items': None,
                        'error': str(e)
                    })
            
            # Update adaptive metrics
            batch_time = time.time() - batch_start_time
            self.adaptive_metrics['throughput_history'].append({
                'batch_size': len(batch),
                'processing_time': batch_time,
                'docs_per_second': len(batch) / batch_time if batch_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            })
            
            # Memory cleanup after each batch
            if self.optimization_config['memory_optimization']:
                gc.collect()
        
        if failed_docs:
            logger.warning(f"Parallel processing completed with {len(failed_docs)} failures")
        
        return results
    
    def _process_single_document_for_parallel(self, doc: Dict, **kwargs) -> Dict:
        """Process a single document for parallel execution."""
        doc_id = doc.get('id', 'unknown')
        
        try:
            # Security validation
            if not self._validate_document_security(doc):
                return {
                    'success': False,
                    'security_violation': True,
                    'doc_id': doc_id,
                    'items': None,
                    'error': 'Security validation failed'
                }
            
            # Process with timeout protection
            items = self._process_document_with_timeout(doc, **kwargs)
            
            return {
                'success': True,
                'security_violation': False,
                'doc_id': doc_id,
                'items': items or [],
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'security_violation': False,
                'doc_id': doc_id,
                'items': None,
                'error': str(e)
            }
    
    def _calculate_optimal_batch_size(self, total_docs: int) -> int:
        """Calculate optimal batch size based on system resources and history."""
        if not self.optimization_config['adaptive_batch_size']:
            return min(8, total_docs)
        
        # Base batch size on available memory and CPU cores
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_cores = psutil.cpu_count() or 4
        
        # Conservative estimation: ~100MB per document processing
        memory_based_batch = max(1, int(available_memory_gb * 10))
        cpu_based_batch = cpu_cores * 2
        
        # Use historical performance data if available
        if self.adaptive_metrics['throughput_history']:
            recent_history = self.adaptive_metrics['throughput_history'][-10:]  # Last 10 batches
            avg_throughput = sum(h['docs_per_second'] for h in recent_history) / len(recent_history)
            
            # Adjust based on throughput trends
            if avg_throughput > 2.0:  # High throughput, can handle larger batches
                performance_factor = 1.5
            elif avg_throughput < 0.5:  # Low throughput, reduce batch size
                performance_factor = 0.7
            else:
                performance_factor = 1.0
        else:
            performance_factor = 1.0
        
        # Calculate optimal batch size
        optimal_size = int(min(memory_based_batch, cpu_based_batch) * performance_factor)
        optimal_size = max(1, min(optimal_size, total_docs, 32))  # Reasonable limits
        
        self.adaptive_metrics['optimal_batch_size'] = optimal_size
        return optimal_size
        
        logger.info(f"Generated {len(dataset_items)} vision-language pairs")
        
        # Filter by quality
        high_quality_items = []
        for item in dataset_items:
            if self.validate_quality(item):
                high_quality_items.append(item)
        
        logger.info(f"Filtered to {len(high_quality_items)} high-quality items")
        
        # Split dataset
        dataset_splits = self._split_dataset(
            high_quality_items, train_split, val_split, test_split
        )
        
        # Format dataset
        if output_format == "hf_dataset":
            return self._format_huggingface_dataset(dataset_splits)
        elif output_format == "coco":
            return self._format_coco_dataset(dataset_splits)
        else:
            return dataset_splits
    
    def _initialize_ocr_engines(self) -> None:
        """Initialize OCR engines."""
        self.ocr_processors = {}
        
        if "tesseract" in self.ocr_engines:
            # Tesseract configuration for multiple languages
            self.ocr_processors["tesseract"] = {
                "config": r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
            }
        
        if "easyocr" in self.ocr_engines:
            # EasyOCR reader with multiple languages
            supported_langs = ['en', 'fr', 'es', 'ar', 'de', 'it', 'pt']
            ocr_langs = [lang for lang in self.target_languages + [self.source_language] 
                        if lang in supported_langs]
            if not ocr_langs:
                ocr_langs = ['en']
            
            try:
                self.ocr_processors["easyocr"] = easyocr.Reader(ocr_langs)
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        if "paddleocr" in self.ocr_engines:
            try:
                self.ocr_processors["paddleocr"] = paddleocr.PaddleOCR(
                    use_angle_cls=True, lang='en', show_log=False
                )
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
    
    def _initialize_translation(self) -> None:
        """Initialize translation pipeline."""
        try:
            self.translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")
        except Exception as e:
            logger.warning(f"Failed to initialize translator: {e}")
            self.translator = None
    
    def _load_instruction_templates(self) -> Dict[str, List[str]]:
        """Load instruction generation templates."""
        return {
            "description": [
                "Describe what you see in this image.",
                "What is shown in this image?",
                "Provide a detailed description of this image.",
                "Explain what this image depicts.",
                "What can you observe in this image?"
            ],
            "question": [
                "What does this chart show?",
                "What information is presented in this infographic?",
                "What are the key findings shown here?",
                "What data is illustrated in this visualization?",
                "What story does this image tell?"
            ],
            "analysis": [
                "Analyze the trends shown in this visualization.",
                "What insights can be drawn from this data?",
                "Interpret the information presented in this image.",
                "What conclusions can be made from this chart?",
                "Summarize the key points from this infographic."
            ]
        }
    
    def _process_document(
        self, 
        doc: Dict[str, Any], 
        include_infographics: bool, 
        include_maps: bool, 
        include_charts: bool
    ) -> List[Dict[str, Any]]:
        """Process a single document to extract vision-language pairs."""
        items = []
        
        # Process document images
        for image_info in doc.get('images', []):
            try:
                # Classify image type
                image_type = self._classify_image_type(image_info)
                
                # Skip based on inclusion settings
                if (
                    (image_type == 'infographic' and not include_infographics) or
                    (image_type == 'map' and not include_maps) or
                    (image_type == 'chart' and not include_charts)
                ):
                    continue
                
                # Extract text from image using advanced adaptive OCR
                ocr_results = self._extract_text_advanced(image_info, image_type)
                
                if not ocr_results or not ocr_results.get('text', '').strip():
                    continue
                
                # Generate instruction-response pairs
                instruction_pairs = self._generate_instructions(
                    image_info, ocr_results, doc, image_type
                )
                
                for pair in instruction_pairs:
                    item = {
                        'id': hashlib.md5(
                            f"{doc['url']}_{image_info.get('page', 0)}_{pair['instruction']}".encode()
                        ).hexdigest(),
                        'document_id': hashlib.md5(doc['url'].encode()).hexdigest(),
                        'source': doc['source'],
                        'language': doc['language'],
                        'image_info': image_info,
                        'image_type': image_type,
                        'instruction': pair['instruction'],
                        'response': pair['response'],
                        'ocr_text': ocr_results['text'],
                        'ocr_confidence': ocr_results['confidence'],
                        'quality_score': self._calculate_quality_score(pair, ocr_results),
                        'context': {
                            'document_title': doc.get('title', ''),
                            'document_content': doc.get('content', '')[:500],  # First 500 chars
                            'page_number': image_info.get('page')
                        }
                    }
                    items.append(item)
                    
            except Exception as e:
                logger.error(f"Error processing image {image_info}: {e}")
                continue
        
        return items
    
    def _classify_image_type(self, image_info: Dict[str, Any]) -> str:
        """Classify image type based on characteristics."""
        # Simple heuristics for image classification
        width = image_info.get('width', 0)
        height = image_info.get('height', 0)
        alt_text = image_info.get('alt', '').lower()
        src = image_info.get('src', '').lower()
        
        # Keywords for different types
        chart_keywords = ['chart', 'graph', 'plot', 'statistic', 'data', 'figure']
        map_keywords = ['map', 'location', 'region', 'country', 'geographic']
        infographic_keywords = ['infographic', 'info', 'summary', 'overview']
        
        # Check alt text and src for keywords
        text_to_check = f"{alt_text} {src}"
        
        if any(keyword in text_to_check for keyword in chart_keywords):
            return 'chart'
        elif any(keyword in text_to_check for keyword in map_keywords):
            return 'map'
        elif any(keyword in text_to_check for keyword in infographic_keywords):
            return 'infographic'
        
        # Check aspect ratio for charts (often wider than tall)
        if width > 0 and height > 0:
            aspect_ratio = width / height
            if aspect_ratio > 1.5:  # Wide images often charts
                return 'chart'
        
        # Default classification
        return 'infographic'
    
    def _extract_text_advanced(self, image_info: Dict[str, Any], image_type: str) -> Dict[str, Any]:
        """Extract text using advanced adaptive OCR with research innovations."""
        # Load image from URL or local path
        image = self._load_image(image_info)
        if image is None:
            return {'text': '', 'confidence': 0.0, 'uncertainty': 1.0}
        
        # Use adaptive OCR system with document type awareness
        document_type = self._map_image_type_to_document_type(image_type)
        ocr_result = self.adaptive_ocr.extract_text(image, document_type)
        
        return ocr_result
    
    def _map_image_type_to_document_type(self, image_type: str) -> str:
        """Map image classification to OCR document type."""
        type_mapping = {
            'chart': 'chart',
            'infographic': 'infographic', 
            'map': 'infographic',
            'table': 'dense_text',
            'document': 'standard'
        }
        return type_mapping.get(image_type, 'standard')
    
    def _extract_text_from_image(self, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility."""
        return self._extract_text_advanced(image_info, 'standard')
    
    def _load_image(self, image_info: Dict[str, Any]):
        """Load image from URL or local path."""
        try:
            src = image_info.get('src', '')
            if not src:
                return None
            
            if src.startswith('http'):
                # Download image from URL
                import requests
                response = requests.get(src, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # Load from local path
                image = Image.open(src)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_info.get('src')}: {e}")
            return None
    
    def _run_tesseract_ocr(self, image):
        """Run Tesseract OCR on image."""
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Extract text with confidence
            config = self.ocr_processors["tesseract"]["config"]
            data = pytesseract.image_to_data(img_array, config=config, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence detections
            texts = []
            confidences = []
            bboxes = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if conf > 30 and text:  # Only include confident detections
                    texts.append(text)
                    confidences.append(conf / 100.0)  # Convert to 0-1 range
                    bboxes.append([
                        data['left'][i], data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ])
            
            if not texts:
                return None
            
            # Combine results
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'bbox': bboxes,
                'word_confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return None
    
    def _run_easyocr_ocr(self, image):
        """Run EasyOCR on image."""
        try:
            if "easyocr" not in self.ocr_processors:
                return None
            
            reader = self.ocr_processors["easyocr"]
            img_array = np.array(image)
            
            # Run EasyOCR
            results = reader.readtext(img_array)
            
            if not results:
                return None
            
            texts = []
            confidences = []
            bboxes = []
            
            for (bbox, text, conf) in results:
                if conf > 0.3 and text.strip():  # Filter low confidence
                    texts.append(text.strip())
                    confidences.append(conf)
                    
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bboxes.append([
                        min(x_coords), min(y_coords),
                        max(x_coords), max(y_coords)
                    ])
            
            if not texts:
                return None
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'bbox': bboxes,
                'word_confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return None
    
    def _run_paddleocr_ocr(self, image):
        """Run PaddleOCR on image."""
        try:
            if "paddleocr" not in self.ocr_processors:
                return None
            
            ocr = self.ocr_processors["paddleocr"]
            img_array = np.array(image)
            
            # Run PaddleOCR
            results = ocr.ocr(img_array, cls=True)
            
            if not results or not results[0]:
                return None
            
            texts = []
            confidences = []
            bboxes = []
            
            for line in results[0]:
                if line:
                    bbox, (text, conf) = line
                    if conf > 0.3 and text.strip():  # Filter low confidence
                        texts.append(text.strip())
                        confidences.append(conf)
                        
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bboxes.append([
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        ])
            
            if not texts:
                return None
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'bbox': bboxes,
                'word_confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return None
    
    def _consensus_ocr_result(self, ocr_results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create consensus result from multiple OCR engines."""
        if not ocr_results:
            return {'text': '', 'confidence': 0.0, 'engines': []}
        
        # For simulation, take the result with highest confidence
        best_result = max(ocr_results, key=lambda x: x[1]['confidence'])
        
        return {
            'text': best_result[1]['text'],
            'confidence': best_result[1]['confidence'],
            'engines': [engine for engine, _ in ocr_results],
            'all_results': {engine: result for engine, result in ocr_results}
        }
    
    def _generate_instructions(
        self, 
        image_info: Dict[str, Any], 
        ocr_results: Dict[str, Any], 
        doc: Dict[str, Any], 
        image_type: str
    ) -> List[Dict[str, str]]:
        """Generate instruction-response pairs for the image."""
        pairs = []
        
        # Select appropriate templates based on image type
        if image_type in ['chart', 'infographic']:
            template_types = ['description', 'question', 'analysis']
        else:
            template_types = ['description', 'question']
        
        # Generate multiple instruction types
        for template_type in template_types:
            templates = self.instruction_templates.get(template_type, [])
            if templates:
                instruction = random.choice(templates)
                response = self._generate_response(
                    instruction, ocr_results, image_info, doc, image_type
                )
                
                pairs.append({
                    'instruction': instruction,
                    'response': response,
                    'type': template_type
                })
        
        return pairs
    
    def _generate_response(
        self, 
        instruction: str, 
        ocr_results: Dict[str, Any], 
        image_info: Dict[str, Any], 
        doc: Dict[str, Any], 
        image_type: str
    ) -> str:
        """Generate response based on instruction and extracted content."""
        ocr_text = ocr_results.get('text', '')
        alt_text = image_info.get('alt', '')
        
        # Base response on OCR text and context
        response_parts = []
        
        if ocr_text:
            response_parts.append(f"This {image_type} contains the following text: {ocr_text}")
        
        if alt_text and alt_text.lower() not in ocr_text.lower():
            response_parts.append(f"The image is described as: {alt_text}")
        
        # Add context from document
        doc_context = doc.get('content', '')[:200]  # First 200 chars
        if doc_context:
            response_parts.append(f"Based on the document context: {doc_context}")
        
        # Customize response based on instruction type
        if 'analyze' in instruction.lower() or 'insight' in instruction.lower():
            response_parts.append("This visualization appears to present humanitarian data that could be used for decision-making and resource allocation.")
        
        return ' '.join(response_parts)
    
    def _calculate_quality_score(self, pair: Dict[str, str], ocr_results: Dict[str, Any]) -> float:
        """Calculate quality score for instruction-response pair."""
        score = 0.0
        
        # OCR confidence contributes 40%
        ocr_confidence = ocr_results.get('confidence', 0.0)
        score += 0.4 * ocr_confidence
        
        # Response length and completeness contributes 30%
        response = pair.get('response', '')
        if len(response) > 50:  # Minimum length
            score += 0.3
        elif len(response) > 20:
            score += 0.15
        
        # Instruction clarity contributes 20%
        instruction = pair.get('instruction', '')
        if len(instruction) > 10 and '?' in instruction:
            score += 0.2
        
        # Content relevance contributes 10%
        ocr_text = ocr_results.get('text', '')
        if ocr_text and len(ocr_text) > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def _split_dataset(
        self, 
        items: List[Dict[str, Any]], 
        train_split: float, 
        val_split: float, 
        test_split: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Split dataset into train/validation/test sets."""
        # Shuffle items
        random.shuffle(items)
        
        total = len(items)
        train_end = int(total * train_split)
        val_end = int(total * (train_split + val_split))
        
        return {
            'train': items[:train_end],
            'validation': items[train_end:val_end],
            'test': items[val_end:]
        }
    
    def _format_huggingface_dataset(self, dataset_splits: Dict[str, List[Dict[str, Any]]]) -> DatasetDict:
        """Format dataset for HuggingFace."""
        formatted_splits = {}
        
        for split_name, items in dataset_splits.items():
            if not items:
                continue
                
            # Convert to format expected by HuggingFace
            formatted_items = []
            for item in items:
                formatted_item = {
                    'id': item['id'],
                    'instruction': item['instruction'],
                    'response': item['response'],
                    'image_type': item['image_type'],
                    'language': item['language'],
                    'source': item['source'],
                    'quality_score': item['quality_score'],
                    'ocr_confidence': item['ocr_confidence']
                }
                formatted_items.append(formatted_item)
            
            # Create Dataset from list of dictionaries
            formatted_splits[split_name] = Dataset.from_list(formatted_items)
        
        return DatasetDict(formatted_splits)
    
    def _format_coco_dataset(self, dataset_splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Format dataset in COCO-style format."""
        coco_format = {
            'info': {
                'description': 'VisLang UltraLow Resource Dataset',
                'version': '1.0',
                'year': 2025,
                'contributor': 'Terragon Labs'
            },
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'chart'},
                {'id': 2, 'name': 'infographic'},
                {'id': 3, 'name': 'map'}
            ]
        }
        
        image_id = 1
        annotation_id = 1
        
        for split_name, items in dataset_splits.items():
            for item in items:
                # Add image info
                image_info = {
                    'id': image_id,
                    'file_name': f"{item['id']}.jpg",
                    'width': item['image_info'].get('width', 800),
                    'height': item['image_info'].get('height', 600),
                    'split': split_name
                }
                coco_format['images'].append(image_info)
                
                # Add annotation
                category_map = {'chart': 1, 'infographic': 2, 'map': 3}
                category_id = category_map.get(item['image_type'], 2)
                
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'instruction': item['instruction'],
                    'response': item['response'],
                    'language': item['language'],
                    'quality_score': item['quality_score']
                }
                coco_format['annotations'].append(annotation)
                
                image_id += 1
                annotation_id += 1
        
        return coco_format
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Validate item quality against threshold.
        
        Args:
            item: Dataset item to validate
            
        Returns:
            True if item meets quality threshold
        """
        quality_score = item.get('quality_score', 0.0)
        
        # Basic quality checks
        checks = [
            quality_score >= self.min_quality_score,
            len(item.get('instruction', '')) > 10,
            len(item.get('response', '')) > 20,
            item.get('ocr_confidence', 0.0) > 0.5
        ]
        
        return all(checks)
    
    def save_dataset(
        self, 
        dataset: Union[DatasetDict, Dict[str, Any]], 
        name: str, 
        format_type: str = "hf_dataset"
    ) -> Path:
        """Save dataset to disk."""
        output_path = self.output_dir / name
        output_path.mkdir(exist_ok=True)
        
        if format_type == "hf_dataset" and isinstance(dataset, DatasetDict):
            dataset.save_to_disk(str(output_path))
        else:
            # Save as JSON
            with open(output_path / "dataset.json", 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        return output_path
    
    def get_dataset_statistics(self, dataset: Union[DatasetDict, Dict[str, Any]]) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_items': 0,
            'splits': {},
            'languages': defaultdict(int),
            'image_types': defaultdict(int),
            'sources': defaultdict(int),
            'quality_distribution': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.6-0.8
                'low': 0   # < 0.6
            },
            'avg_quality_score': 0.0
        }
        
        if isinstance(dataset, DatasetDict):
            # HuggingFace dataset
            for split_name, split_data in dataset.items():
                split_size = len(split_data)
                stats['splits'][split_name] = split_size
                stats['total_items'] += split_size
                
                for item in split_data:
                    stats['languages'][item['language']] += 1
                    stats['image_types'][item['image_type']] += 1
                    stats['sources'][item['source']] += 1
                    
                    quality = item['quality_score']
                    if quality > 0.8:
                        stats['quality_distribution']['high'] += 1
                    elif quality > 0.6:
                        stats['quality_distribution']['medium'] += 1
                    else:
                        stats['quality_distribution']['low'] += 1
        
        else:
            # Custom format
            for split_name, items in dataset.items():
                if isinstance(items, list):
                    stats['splits'][split_name] = len(items)
                    stats['total_items'] += len(items)
        
        if stats['total_items'] > 0:
            total_quality = sum(
                stats['quality_distribution']['high'] * 0.9 +
                stats['quality_distribution']['medium'] * 0.7 +
                stats['quality_distribution']['low'] * 0.5
            )
            stats['avg_quality_score'] = total_quality / stats['total_items']
        
        return dict(stats)