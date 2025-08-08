"""Data validation utilities."""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import os

# Conditional imports with fallbacks
try:
    from PIL import Image
except ImportError:
    class Image:
        @staticmethod
        def open(path):
            class MockImg:
                width = 640
                height = 480
                mode = 'RGB'
                format = 'JPEG'
                def verify(self): pass
            return MockImg()

try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0

try:
    import psutil
except ImportError:
    class psutil:
        @staticmethod
        def virtual_memory():
            class Memory:
                available = 8 * 1024**3  # 8GB mock
            return Memory()
        @staticmethod
        def disk_usage(path):
            class Disk:
                free = 50 * 1024**3  # 50GB mock
            return Disk()
        @staticmethod
        def cpu_count(): return 4

from ..exceptions import ValidationError, QualityError, ResourceError

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation utilities."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: If True, raise exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.validation_errors = []
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate document structure and content.
        
        Args:
            document: Document dictionary to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If strict_mode=True and validation fails
        """
        errors = []
        
        # Required fields
        required_fields = ['url', 'title', 'source', 'language', 'content']
        for field in required_fields:
            if field not in document or not document[field]:
                errors.append(f"Missing required field: {field}")
        
        # URL validation
        if 'url' in document:
            if not self._validate_url(document['url']):
                errors.append(f"Invalid URL format: {document['url']}")
        
        # Source validation
        if 'source' in document:
            valid_sources = ['unhcr', 'who', 'unicef', 'wfp', 'ocha', 'undp']
            if document['source'] not in valid_sources:
                errors.append(f"Invalid source: {document['source']}")
        
        # Language validation
        if 'language' in document:
            if not self._validate_language_code(document['language']):
                errors.append(f"Invalid language code: {document['language']}")
        
        # Content validation
        if 'content' in document:
            if not self._validate_text_content(document['content']):
                errors.append("Content quality too low")
        
        # Word count validation
        if 'word_count' in document:
            if not isinstance(document['word_count'], int) or document['word_count'] < 0:
                errors.append("Invalid word count")
        
        # Images validation
        if 'images' in document:
            if not isinstance(document['images'], list):
                errors.append("Images must be a list")
            else:
                for i, img in enumerate(document['images']):
                    if not self.validate_image_metadata(img):
                        errors.append(f"Invalid image metadata at index {i}")
        
        self.validation_errors.extend(errors)
        
        if errors and self.strict_mode:
            raise ValidationError(f"Document validation failed: {'; '.join(errors)}")
        
        return len(errors) == 0
    
    def validate_image_metadata(self, image_info: Dict[str, Any]) -> bool:
        """Validate image metadata structure.
        
        Args:
            image_info: Image metadata dictionary
            
        Returns:
            True if valid, False otherwise
        """
        errors = []
        
        # Check for src field
        if 'src' not in image_info:
            errors.append("Image missing 'src' field")
        elif image_info['src']:
            # Validate URL or file path
            src = image_info['src']
            if src.startswith('http'):
                if not self._validate_url(src):
                    errors.append(f"Invalid image URL: {src}")
            else:
                # Check if file exists for local paths
                if not Path(src).is_absolute() and not Path(src).exists():
                    logger.warning(f"Image file may not exist: {src}")
        
        # Validate dimensions if present
        for dim in ['width', 'height']:
            if dim in image_info:
                if not isinstance(image_info[dim], (int, float)) or image_info[dim] <= 0:
                    errors.append(f"Invalid {dim}: {image_info[dim]}")
        
        # Validate alt text
        if 'alt' in image_info and image_info['alt']:
            if not isinstance(image_info['alt'], str):
                errors.append("Alt text must be string")
            elif len(image_info['alt']) > 500:  # Reasonable limit
                errors.append("Alt text too long")
        
        self.validation_errors.extend(errors)
        return len(errors) == 0
    
    def validate_dataset_item(self, item: Dict[str, Any]) -> bool:
        """Validate dataset item structure and quality.
        
        Args:
            item: Dataset item to validate
            
        Returns:
            True if valid, False otherwise
        """
        errors = []
        
        # Required fields
        required_fields = ['instruction', 'response', 'quality_score']
        for field in required_fields:
            if field not in item or item[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Instruction validation
        if 'instruction' in item:
            instruction = item['instruction']
            if not isinstance(instruction, str):
                errors.append("Instruction must be string")
            elif len(instruction.strip()) < 10:
                errors.append("Instruction too short")
            elif len(instruction) > 1000:
                errors.append("Instruction too long")
        
        # Response validation
        if 'response' in item:
            response = item['response']
            if not isinstance(response, str):
                errors.append("Response must be string")
            elif len(response.strip()) < 10:
                errors.append("Response too short")
            elif len(response) > 2000:
                errors.append("Response too long")
        
        # Quality score validation
        if 'quality_score' in item:
            score = item['quality_score']
            if not isinstance(score, (int, float)):
                errors.append("Quality score must be numeric")
            elif not 0 <= score <= 1:
                errors.append("Quality score must be between 0 and 1")
        
        # OCR confidence validation
        if 'ocr_confidence' in item:
            conf = item['ocr_confidence']
            if not isinstance(conf, (int, float)):
                errors.append("OCR confidence must be numeric")
            elif not 0 <= conf <= 1:
                errors.append("OCR confidence must be between 0 and 1")
        
        # Language validation
        if 'language' in item:
            if not self._validate_language_code(item['language']):
                errors.append(f"Invalid language code: {item['language']}")
        
        self.validation_errors.extend(errors)
        return len(errors) == 0
    
    def validate_training_config(self, config: Dict[str, Any]) -> bool:
        """Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        errors = []
        
        # Required fields
        required_fields = ['model_name', 'languages', 'batch_size', 'learning_rate']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Batch size validation
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append("Batch size must be positive integer")
            elif batch_size > 128:  # Reasonable upper limit
                logger.warning(f"Large batch size: {batch_size}")
        
        # Learning rate validation
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("Learning rate must be positive number")
            elif lr > 0.1:
                logger.warning(f"High learning rate: {lr}")
        
        # Epochs validation
        if 'num_epochs' in config:
            epochs = config['num_epochs']
            if not isinstance(epochs, int) or epochs < 1:
                errors.append("Number of epochs must be positive integer")
            elif epochs > 100:
                logger.warning(f"Large number of epochs: {epochs}")
        
        # Languages validation
        if 'languages' in config:
            languages = config['languages']
            if not isinstance(languages, list):
                errors.append("Languages must be a list")
            elif not languages:
                errors.append("At least one language required")
            else:
                for lang in languages:
                    if not self._validate_language_code(lang):
                        errors.append(f"Invalid language code: {lang}")
        
        self.validation_errors.extend(errors)
        return len(errors) == 0
    
    def validate_image_file(self, image_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate image file and extract metadata.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, metadata_dict)
        """
        metadata = {}
        errors = []
        
        try:
            if not image_path.exists():
                errors.append(f"Image file not found: {image_path}")
                return False, metadata
            
            # Check file size
            file_size = image_path.stat().st_size
            metadata['file_size'] = file_size
            
            if file_size == 0:
                errors.append("Image file is empty")
            elif file_size > 50 * 1024 * 1024:  # 50MB limit
                errors.append("Image file too large")
            
            # Try to open and validate image
            with Image.open(image_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['mode'] = img.mode
                metadata['format'] = img.format
                
                # Validate dimensions
                if img.width < 50 or img.height < 50:
                    errors.append("Image too small")
                elif img.width > 5000 or img.height > 5000:
                    logger.warning(f"Very large image: {img.width}x{img.height}")
                
                # Check for corruption
                img.verify()
                
        except Exception as e:
            errors.append(f"Failed to validate image: {str(e)}")
        
        self.validation_errors.extend(errors)
        return len(errors) == 0, metadata
    
    def validate_system_resources(self, min_memory_gb: float = 4.0, min_disk_gb: float = 10.0) -> bool:
        """Validate system has sufficient resources.
        
        Args:
            min_memory_gb: Minimum available memory in GB
            min_disk_gb: Minimum available disk space in GB
            
        Returns:
            True if sufficient resources available
            
        Raises:
            ResourceError: If insufficient resources and strict_mode=True
        """
        errors = []
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        if available_memory_gb < min_memory_gb:
            errors.append(f"Insufficient memory: {available_memory_gb:.1f}GB < {min_memory_gb}GB")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        available_disk_gb = disk.free / (1024**3)
        
        if available_disk_gb < min_disk_gb:
            errors.append(f"Insufficient disk space: {available_disk_gb:.1f}GB < {min_disk_gb}GB")
        
        # Check CPU availability
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            logger.warning(f"Limited CPU cores: {cpu_count}")
        
        if errors and self.strict_mode:
            raise ResourceError(f"Insufficient system resources: {'; '.join(errors)}")
        
        return len(errors) == 0
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _validate_language_code(self, lang_code: str) -> bool:
        """Validate ISO 639-1 language code."""
        if not isinstance(lang_code, str) or len(lang_code) != 2:
            return False
        
        # Common language codes for humanitarian work
        valid_codes = {
            'en', 'fr', 'es', 'ar', 'pt', 'de', 'it', 'ru', 'zh',
            'sw', 'am', 'ha', 'yo', 'ig', 'zu', 'so', 'ti', 'om',
            'hi', 'ur', 'bn', 'ta', 'te', 'ml', 'kn', 'gu', 'pa',
            'fa', 'ps', 'ku', 'tr', 'az', 'uz', 'kk', 'ky', 'tg',
            'my', 'th', 'vi', 'km', 'lo', 'si', 'ne', 'dz', 'bo'
        }
        
        return lang_code.lower() in valid_codes
    
    def _validate_text_content(self, content: str) -> bool:
        """Validate text content quality."""
        if not isinstance(content, str):
            return False
        
        content = content.strip()
        
        # Minimum length
        if len(content) < 50:
            return False
        
        # Check for reasonable word count
        words = content.split()
        if len(words) < 10:
            return False
        
        # Check for too many special characters (might indicate OCR errors)
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content)
        if special_char_ratio > 0.3:
            return False
        
        # Check for repeated characters (OCR artifacts)
        if re.search(r'(.)\1{10,}', content):  # Same character repeated 10+ times
            return False
        
        return True
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results.
        
        Returns:
            Dictionary with validation statistics
        """
        return {
            'total_errors': len(self.validation_errors),
            'errors': self.validation_errors,
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_errors(self):
        """Clear accumulated validation errors."""
        self.validation_errors.clear()


class QualityAssessment:
    """Quality assessment utilities for datasets and models."""
    
    def __init__(self, quality_thresholds: Dict[str, float] = None):
        """Initialize quality assessment.
        
        Args:
            quality_thresholds: Custom quality thresholds
        """
        self.thresholds = quality_thresholds or {
            'ocr_confidence': 0.7,
            'text_quality': 0.6,
            'image_quality': 0.5,
            'overall_quality': 0.8
        }
    
    def assess_ocr_quality(self, ocr_results: Dict[str, Any]) -> float:
        """Assess OCR result quality.
        
        Args:
            ocr_results: OCR results dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        factors = []
        
        # Confidence from OCR engine
        if 'confidence' in ocr_results:
            conf = ocr_results['confidence']
            factors.append(conf)
            score += 0.4 * conf
        
        # Text length factor
        text = ocr_results.get('text', '')
        if text:
            # Longer text generally indicates better extraction
            length_factor = min(len(text.split()) / 50.0, 1.0)
            factors.append(length_factor)
            score += 0.2 * length_factor
            
            # Character variety (not just repeated chars)
            unique_chars = len(set(text.lower().replace(' ', '')))
            variety_factor = min(unique_chars / 20.0, 1.0)
            factors.append(variety_factor)
            score += 0.2 * variety_factor
            
            # Word formation quality
            words = text.split()
            if words:
                avg_word_length = np.mean([len(w) for w in words])
                # Average word length between 3-8 characters is ideal
                if 3 <= avg_word_length <= 8:
                    word_quality = 1.0
                else:
                    word_quality = max(0.0, 1.0 - abs(avg_word_length - 5.5) / 10)
                factors.append(word_quality)
                score += 0.2 * word_quality
        
        return min(score, 1.0)
    
    def assess_text_quality(self, text: str) -> float:
        """Assess text content quality.
        
        Args:
            text: Text content to assess
            
        Returns:
            Quality score between 0 and 1
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        text = text.strip()
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length factor (reasonable length gets higher score)
        length_score = min(len(text) / 500.0, 1.0)  # Optimal around 500 chars
        score += 0.2 * length_score
        
        # Word count factor
        words = text.split()
        word_count_score = min(len(words) / 50.0, 1.0)  # Optimal around 50 words
        score += 0.2 * word_count_score
        
        # Sentence structure (presence of punctuation)
        sentence_endings = sum(1 for c in text if c in '.!?')
        if len(words) > 0:
            sentence_score = min(sentence_endings / (len(words) / 10), 1.0)
            score += 0.2 * sentence_score
        
        # Character quality (not too many special chars)
        alphanum_ratio = sum(1 for c in text if c.isalnum()) / len(text)
        score += 0.2 * alphanum_ratio
        
        # Language consistency (basic check)
        if re.search(r'[a-zA-Z]', text):  # Contains Latin characters
            consistency_score = 0.8
        else:
            consistency_score = 0.6  # Non-Latin scripts
        score += 0.2 * consistency_score
        
        return min(score, 1.0)
    
    def assess_dataset_item_quality(self, item: Dict[str, Any]) -> float:
        """Assess overall quality of dataset item.
        
        Args:
            item: Dataset item dictionary
            
        Returns:
            Overall quality score between 0 and 1
        """
        scores = []
        weights = []
        
        # OCR quality
        if 'ocr_confidence' in item:
            ocr_score = item['ocr_confidence']
            scores.append(ocr_score)
            weights.append(0.3)
        
        # Instruction quality
        if 'instruction' in item:
            inst_score = self.assess_text_quality(item['instruction'])
            scores.append(inst_score)
            weights.append(0.3)
        
        # Response quality
        if 'response' in item:
            resp_score = self.assess_text_quality(item['response'])
            scores.append(resp_score)
            weights.append(0.4)
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Apply quality thresholds
        if weighted_score < self.thresholds.get('overall_quality', 0.8):
            logger.debug(f"Item quality below threshold: {weighted_score:.3f}")
        
        return weighted_score
    
    def filter_by_quality(self, items: List[Dict[str, Any]], min_quality: float = None) -> List[Dict[str, Any]]:
        """Filter items by quality score.
        
        Args:
            items: List of items to filter
            min_quality: Minimum quality threshold
            
        Returns:
            Filtered list of high-quality items
        """
        if min_quality is None:
            min_quality = self.thresholds.get('overall_quality', 0.8)
        
        filtered_items = []
        quality_scores = []
        
        for item in items:
            # Use existing quality score if available
            if 'quality_score' in item:
                quality = item['quality_score']
            else:
                # Calculate quality score
                quality = self.assess_dataset_item_quality(item)
                item['quality_score'] = quality
            
            quality_scores.append(quality)
            
            if quality >= min_quality:
                filtered_items.append(item)
        
        logger.info(f"Quality filtering: {len(filtered_items)}/{len(items)} items passed "
                   f"(avg quality: {np.mean(quality_scores):.3f})")
        
        return filtered_items


def validate_document_security(document: Dict[str, Any], strict: bool = True) -> bool:
    """Security validation for documents to prevent malicious content.
    
    Args:
        document: Document dictionary to validate
        strict: If True, applies strict security checks
        
    Returns:
        True if document passes security validation
        
    Raises:
        ValidationError: If security validation fails and strict=True
    """
    security_issues = []
    
    # Check for suspicious URLs
    if 'url' in document:
        url = document['url'].lower()
        suspicious_patterns = [
            'javascript:', 'data:', 'vbscript:', 'file:', 'ftp://',
            '<script', '</script>', 'eval(', 'onclick=', 'onerror='
        ]
        
        for pattern in suspicious_patterns:
            if pattern in url:
                security_issues.append(f"Suspicious URL pattern detected: {pattern}")
                break
    
    # Check content for potential XSS/injection
    if 'content' in document:
        content = document['content'].lower()
        dangerous_patterns = [
            '<script', '</script>', 'javascript:', 'eval(',
            '<iframe', '<embed', '<object', 'onclick=', 'onerror=',
            'document.cookie', 'window.location', 'document.write'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in content:
                security_issues.append(f"Potentially dangerous content pattern: {pattern}")
                
    # Check for suspicious file paths
    if 'images' in document:
        for img in document['images']:
            if 'src' in img:
                src = img['src'].lower()
                if any(path in src for path in ['../', '.\\', '/etc/', '/proc/', 'c:\\']):
                    security_issues.append(f"Suspicious file path: {img['src']}")
    
    # Check metadata for suspicious entries
    if 'document_metadata' in document:
        metadata = str(document['document_metadata']).lower()
        if any(pattern in metadata for pattern in ['<script', 'javascript:', 'eval(']):
            security_issues.append("Suspicious patterns in metadata")
    
    # Size limits for DoS prevention
    if 'content' in document and len(document['content']) > 10 * 1024 * 1024:  # 10MB
        security_issues.append("Content exceeds maximum size limit")
    
    if security_issues:
        error_msg = f"Security validation failed: {'; '.join(security_issues)}"
        logger.warning(error_msg)
        if strict:
            raise ValidationError(error_msg)
        return False
    
    return True


def sanitize_input(text: str) -> str:
    """Sanitize text input to prevent injection attacks.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove dangerous HTML/script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)  # Remove all HTML tags
    
    # Remove potential JavaScript
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'data:', '', text, flags=re.IGNORECASE)
    
    # Remove event handlers
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    # Escape special characters
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('"', '&quot;').replace("'", '&#x27;')
    
    return text.strip()