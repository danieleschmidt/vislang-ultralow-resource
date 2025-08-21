"""Advanced validation and error handling system for robust operation."""

import logging
import re
import json
import os
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import functools
import traceback
from contextlib import contextmanager
import threading
from collections import defaultdict, deque

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

try:
    from PIL import Image
    _pil_available = True
except ImportError:
    _pil_available = False

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Validation categories."""
    DATA_INTEGRITY = "data_integrity"
    FORMAT_COMPLIANCE = "format_compliance"
    BUSINESS_LOGIC = "business_logic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    RESOURCE_LIMITS = "resource_limits"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    validator_name: str = ""
    fix_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'severity': self.severity.value,
            'category': self.category.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'validator_name': self.validator_name,
            'fix_suggestions': self.fix_suggestions
        }


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""
    strict_mode: bool = True
    fail_fast: bool = False
    collect_all_errors: bool = True
    enable_auto_fix: bool = False
    max_validation_time: int = 300  # seconds
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        'max_file_size_mb': 100,
        'max_memory_usage_mb': 500,
        'max_processing_time_s': 60
    })


class ValidationError(Exception):
    """Custom validation error with detailed information."""
    
    def __init__(self, message: str, validation_results: List[ValidationResult] = None):
        super().__init__(message)
        self.validation_results = validation_results or []
        self.timestamp = datetime.now()
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get detailed error summary."""
        return {
            'message': str(self),
            'timestamp': self.timestamp.isoformat(),
            'validation_count': len(self.validation_results),
            'error_count': sum(1 for r in self.validation_results if not r.is_valid),
            'critical_count': sum(1 for r in self.validation_results 
                                if not r.is_valid and r.severity == ValidationSeverity.CRITICAL),
            'categories': list(set(r.category.value for r in self.validation_results)),
            'fix_suggestions': list(set(
                suggestion for r in self.validation_results 
                for suggestion in r.fix_suggestions
            ))
        }


class AdvancedValidator:
    """Advanced validation system with comprehensive checks."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.validation_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.custom_validators = {}
        self.validation_cache = {}
        self.lock = threading.RLock()
        
        # Initialize validators
        self._initialize_builtin_validators()
        
        logger.info("Advanced validator initialized", extra={
            'strict_mode': self.config.strict_mode,
            'fail_fast': self.config.fail_fast
        })
    
    def _initialize_builtin_validators(self):
        """Initialize built-in validators."""
        self.register_validator('url', self._validate_url)
        self.register_validator('email', self._validate_email)
        self.register_validator('file_path', self._validate_file_path)
        self.register_validator('image_metadata', self._validate_image_metadata)
        self.register_validator('json_structure', self._validate_json_structure)
        self.register_validator('text_content', self._validate_text_content)
        self.register_validator('language_code', self._validate_language_code)
        self.register_validator('dataset_integrity', self._validate_dataset_integrity)
        self.register_validator('model_config', self._validate_model_config)
        self.register_validator('resource_usage', self._validate_resource_usage)
    
    def register_validator(self, name: str, validator_func: Callable):
        """Register custom validator function."""
        self.custom_validators[name] = validator_func
        logger.debug(f"Registered validator: {name}")
    
    def validate(self, data: Any, validator_names: List[str], 
                context: Dict[str, Any] = None) -> List[ValidationResult]:
        """Run multiple validators on data."""
        start_time = time.time()
        results = []
        context = context or {}
        
        try:
            for validator_name in validator_names:
                if validator_name not in self.custom_validators:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.BUSINESS_LOGIC,
                        message=f"Unknown validator: {validator_name}",
                        validator_name=validator_name
                    ))
                    continue
                
                # Check cache first
                cache_key = self._get_cache_key(data, validator_name, context)
                if cache_key in self.validation_cache:
                    cached_result = self.validation_cache[cache_key]
                    # Check if cache is still valid (5 minutes)
                    if (datetime.now() - cached_result.timestamp).seconds < 300:
                        results.append(cached_result)
                        continue
                
                # Run validator
                try:
                    validator_func = self.custom_validators[validator_name]
                    result = self._run_validator_with_timeout(
                        validator_func, data, context, validator_name
                    )
                    results.append(result)
                    
                    # Cache successful validations
                    if result.is_valid:
                        self.validation_cache[cache_key] = result
                        
                except Exception as e:
                    logger.error(f"Validator {validator_name} failed: {e}")
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        category=ValidationCategory.BUSINESS_LOGIC,
                        message=f"Validator error: {str(e)}",
                        details={'exception': str(e), 'traceback': traceback.format_exc()},
                        validator_name=validator_name
                    ))
                
                # Fail fast if configured
                if self.config.fail_fast and not results[-1].is_valid:
                    break
            
            # Record performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['validation_time'].append(execution_time)
            
            # Store in history
            with self.lock:
                self.validation_history.append({
                    'timestamp': datetime.now(),
                    'validators': validator_names,
                    'results_count': len(results),
                    'success_count': sum(1 for r in results if r.is_valid),
                    'execution_time': execution_time
                })
            
            return results
            
        except Exception as e:
            logger.critical(f"Validation system error: {e}")
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.BUSINESS_LOGIC,
                message=f"Validation system error: {str(e)}",
                details={'exception': str(e)}
            )]
    
    def _run_validator_with_timeout(self, validator_func: Callable, data: Any, 
                                  context: Dict[str, Any], name: str) -> ValidationResult:
        """Run validator with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Validator {name} timed out")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.max_validation_time)
        
        try:
            result = validator_func(data, context)
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _get_cache_key(self, data: Any, validator_name: str, context: Dict[str, Any]) -> str:
        """Generate cache key for validation result."""
        data_hash = hashlib.md5(str(data).encode()).hexdigest()
        context_hash = hashlib.md5(str(sorted(context.items())).encode()).hexdigest()
        return f"{validator_name}:{data_hash}:{context_hash}"
    
    def validate_document(self, document: Dict[str, Any]) -> List[ValidationResult]:
        """Validate humanitarian document structure and content."""
        validators = [
            'json_structure',
            'text_content', 
            'url',
            'dataset_integrity',
            'resource_usage'
        ]
        
        if 'images' in document:
            validators.append('image_metadata')
            
        return self.validate(document, validators, {'document_type': 'humanitarian'})
    
    def validate_training_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate model training configuration."""
        return self.validate(config, ['model_config', 'resource_usage'], 
                           {'config_type': 'training'})
    
    def validate_dataset(self, dataset: Any) -> List[ValidationResult]:
        """Validate dataset for training."""
        return self.validate(dataset, ['dataset_integrity'], {'data_type': 'training'})
    
    # Built-in validator implementations
    
    def _validate_url(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate URL format and accessibility."""
        if isinstance(data, dict):
            urls = []
            for key in ['url', 'source_url', 'image_url']:
                if key in data:
                    urls.append(data[key])
        elif isinstance(data, str):
            urls = [data]
        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="URL validation requires string or dict with URL fields",
                validator_name='url'
            )
        
        for url in urls:
            if not isinstance(url, str):
                continue
                
            # Basic URL format validation
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(url):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.FORMAT_COMPLIANCE,
                    message=f"Invalid URL format: {url}",
                    details={'url': url},
                    validator_name='url',
                    fix_suggestions=['Check URL format', 'Ensure protocol is included']
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.FORMAT_COMPLIANCE,
            message="URL validation passed",
            validator_name='url'
        )
    
    def _validate_email(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate email address format."""
        if not isinstance(data, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="Email must be a string",
                validator_name='email'
            )
        
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not email_pattern.match(data):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message=f"Invalid email format: {data}",
                validator_name='email',
                fix_suggestions=['Check email format', 'Ensure @ symbol is present']
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.FORMAT_COMPLIANCE,
            message="Email validation passed",
            validator_name='email'
        )
    
    def _validate_file_path(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate file path safety and existence."""
        if not isinstance(data, (str, Path)):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="File path must be string or Path object",
                validator_name='file_path'
            )
        
        path = Path(data)
        
        # Check for path traversal
        if '..' in str(path):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.SECURITY,
                message="Path traversal detected in file path",
                details={'path': str(path)},
                validator_name='file_path',
                fix_suggestions=['Remove .. from path', 'Use absolute paths']
            )
        
        # Check file size if it exists
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            max_size = self.config.resource_limits['max_file_size_mb']
            
            if size_mb > max_size:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.RESOURCE_LIMITS,
                    message=f"File too large: {size_mb:.1f}MB > {max_size}MB",
                    details={'size_mb': size_mb, 'max_size_mb': max_size},
                    validator_name='file_path',
                    fix_suggestions=['Compress file', 'Split into smaller files']
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.FORMAT_COMPLIANCE,
            message="File path validation passed",
            validator_name='file_path'
        )
    
    def _validate_image_metadata(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate image metadata structure and values."""
        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="Image metadata must be a dictionary",
                validator_name='image_metadata'
            )
        
        images = data.get('images', [])
        if not isinstance(images, list):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="Images field must be a list",
                validator_name='image_metadata'
            )
        
        for i, image in enumerate(images):
            if not isinstance(image, dict):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.FORMAT_COMPLIANCE,
                    message=f"Image {i} must be a dictionary",
                    validator_name='image_metadata'
                )
            
            # Check required fields
            required_fields = ['src', 'width', 'height']
            missing_fields = [field for field in required_fields if field not in image]
            if missing_fields:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message=f"Image {i} missing fields: {missing_fields}",
                    details={'missing_fields': missing_fields, 'image_index': i},
                    validator_name='image_metadata',
                    fix_suggestions=[f'Add {field} field' for field in missing_fields]
                )
            
            # Validate dimensions
            width = image.get('width')
            height = image.get('height')
            
            if not isinstance(width, (int, float)) or width <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message=f"Image {i} has invalid width: {width}",
                    validator_name='image_metadata',
                    fix_suggestions=['Set valid positive width value']
                )
            
            if not isinstance(height, (int, float)) or height <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    message=f"Image {i} has invalid height: {height}",
                    validator_name='image_metadata',
                    fix_suggestions=['Set valid positive height value']
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.DATA_INTEGRITY,
            message="Image metadata validation passed",
            validator_name='image_metadata'
        )
    
    def _validate_json_structure(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate JSON structure and size."""
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.FORMAT_COMPLIANCE,
                    message=f"Invalid JSON: {str(e)}",
                    validator_name='json_structure',
                    fix_suggestions=['Fix JSON syntax errors']
                )
        else:
            parsed_data = data
        
        # Check nesting depth
        max_depth = 10
        actual_depth = self._get_json_depth(parsed_data)
        
        if actual_depth > max_depth:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.RESOURCE_LIMITS,
                message=f"JSON too deeply nested: {actual_depth} > {max_depth}",
                details={'depth': actual_depth, 'max_depth': max_depth},
                validator_name='json_structure',
                fix_suggestions=['Flatten JSON structure', 'Reduce nesting levels']
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.FORMAT_COMPLIANCE,
            message="JSON structure validation passed",
            validator_name='json_structure'
        )
    
    def _validate_text_content(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate text content quality and safety."""
        if isinstance(data, dict):
            text_fields = ['content', 'text', 'description', 'title']
            texts = [data.get(field, '') for field in text_fields if field in data]
        elif isinstance(data, str):
            texts = [data]
        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="Text content validation requires string or dict",
                validator_name='text_content'
            )
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            # Check text length
            if len(text) > 1000000:  # 1MB
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.RESOURCE_LIMITS,
                    message=f"Text too long: {len(text)} characters",
                    validator_name='text_content',
                    fix_suggestions=['Truncate text', 'Split into smaller chunks']
                )
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>',
                r'javascript:',
                r'eval\s*\(',
                r'document\.cookie'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        category=ValidationCategory.SECURITY,
                        message=f"Suspicious pattern detected: {pattern}",
                        validator_name='text_content',
                        fix_suggestions=['Remove suspicious content', 'Sanitize input']
                    )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.DATA_INTEGRITY,
            message="Text content validation passed",
            validator_name='text_content'
        )
    
    def _validate_language_code(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate language codes."""
        valid_languages = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
            'ar', 'hi', 'sw', 'am', 'ha', 'yo', 'ig', 'zu', 'xh'
        }
        
        if isinstance(data, dict):
            languages = []
            for key in ['language', 'lang', 'target_language', 'source_language']:
                if key in data:
                    lang = data[key]
                    if isinstance(lang, str):
                        languages.append(lang)
                    elif isinstance(lang, list):
                        languages.extend(lang)
        elif isinstance(data, str):
            languages = [data]
        elif isinstance(data, list):
            languages = data
        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="Language validation requires string, list, or dict",
                validator_name='language_code'
            )
        
        invalid_languages = [lang for lang in languages if lang not in valid_languages]
        if invalid_languages:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATA_INTEGRITY,
                message=f"Invalid language codes: {invalid_languages}",
                details={'invalid_languages': invalid_languages},
                validator_name='language_code',
                fix_suggestions=['Use ISO 639-1 language codes', 'Check supported languages']
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.DATA_INTEGRITY,
            message="Language code validation passed",
            validator_name='language_code'
        )
    
    def _validate_dataset_integrity(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate dataset integrity and completeness."""
        if hasattr(data, '__len__'):
            dataset_size = len(data)
        else:
            dataset_size = 1
        
        # Check minimum dataset size
        min_size = context.get('min_dataset_size', 10)
        if dataset_size < min_size:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATA_INTEGRITY,
                message=f"Dataset too small: {dataset_size} < {min_size}",
                validator_name='dataset_integrity',
                fix_suggestions=['Collect more data', 'Use data augmentation']
            )
        
        # Check for data imbalance if it's a collection
        if hasattr(data, '__iter__') and hasattr(data, '__len__'):
            try:
                # Try to check language distribution
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    language_counts = defaultdict(int)
                    for item in data:
                        lang = item.get('language', item.get('lang', 'unknown'))
                        language_counts[lang] += 1
                    
                    if len(language_counts) > 1:
                        min_count = min(language_counts.values())
                        max_count = max(language_counts.values())
                        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                        
                        if imbalance_ratio > 10:  # Significant imbalance
                            return ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.WARNING,
                                category=ValidationCategory.DATA_INTEGRITY,
                                message=f"Dataset imbalanced: ratio {imbalance_ratio:.1f}",
                                details={'language_counts': dict(language_counts)},
                                validator_name='dataset_integrity',
                                fix_suggestions=['Balance dataset', 'Use stratified sampling']
                            )
            except Exception:
                pass  # Skip advanced checks if they fail
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.DATA_INTEGRITY,
            message="Dataset integrity validation passed",
            validator_name='dataset_integrity'
        )
    
    def _validate_model_config(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration."""
        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.FORMAT_COMPLIANCE,
                message="Model config must be a dictionary",
                validator_name='model_config'
            )
        
        # Check required configuration fields
        required_fields = ['model_name', 'learning_rate']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                message=f"Missing config fields: {missing_fields}",
                validator_name='model_config',
                fix_suggestions=[f'Add {field} to config' for field in missing_fields]
            )
        
        # Validate learning rate
        lr = data.get('learning_rate')
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.BUSINESS_LOGIC,
                message=f"Invalid learning rate: {lr}",
                validator_name='model_config',
                fix_suggestions=['Set learning rate between 0 and 1']
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.BUSINESS_LOGIC,
            message="Model config validation passed",
            validator_name='model_config'
        )
    
    def _validate_resource_usage(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
        """Validate resource usage constraints."""
        # Check memory usage
        import psutil
        memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        max_memory = self.config.resource_limits['max_memory_usage_mb']
        
        if memory_usage_mb > max_memory:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.RESOURCE_LIMITS,
                message=f"High memory usage: {memory_usage_mb:.1f}MB > {max_memory}MB",
                details={'memory_usage_mb': memory_usage_mb},
                validator_name='resource_usage',
                fix_suggestions=['Optimize memory usage', 'Process data in batches']
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            category=ValidationCategory.RESOURCE_LIMITS,
            message="Resource usage validation passed",
            validator_name='resource_usage'
        )
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate JSON nesting depth."""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._get_json_depth(value, depth + 1) for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._get_json_depth(item, depth + 1) for item in obj)
        else:
            return depth
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        if not self.validation_history:
            return {'no_data': True}
        
        recent_validations = list(self.validation_history)
        total_validations = len(recent_validations)
        total_success = sum(v['success_count'] for v in recent_validations)
        
        return {
            'total_validations': total_validations,
            'success_rate': total_success / sum(v['results_count'] for v in recent_validations),
            'average_execution_time': np.mean(self.performance_metrics['validation_time']) if _numpy_available else 0,
            'cache_hit_rate': len(self.validation_cache) / total_validations if total_validations > 0 else 0,
            'most_used_validators': self._get_most_used_validators(),
            'performance_trend': self._analyze_performance_trend()
        }
    
    def _get_most_used_validators(self) -> Dict[str, int]:
        """Get most frequently used validators."""
        validator_counts = defaultdict(int)
        for validation in self.validation_history:
            for validator in validation['validators']:
                validator_counts[validator] += 1
        return dict(sorted(validator_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend."""
        if len(self.performance_metrics['validation_time']) < 10:
            return 'insufficient_data'
        
        recent_times = self.performance_metrics['validation_time'][-10:]
        older_times = self.performance_metrics['validation_time'][-20:-10] if len(self.performance_metrics['validation_time']) >= 20 else []
        
        if not older_times:
            return 'stable'
        
        recent_avg = np.mean(recent_times) if _numpy_available else sum(recent_times) / len(recent_times)
        older_avg = np.mean(older_times) if _numpy_available else sum(older_times) / len(older_times)
        
        if recent_avg > older_avg * 1.2:
            return 'degrading'
        elif recent_avg < older_avg * 0.8:
            return 'improving'
        else:
            return 'stable'


@contextmanager
def robust_operation(operation_name: str, 
                    max_retries: int = 3,
                    retry_delay: float = 1.0,
                    fallback_value: Any = None):
    """Context manager for robust operation execution with retries."""
    for attempt in range(max_retries + 1):
        try:
            yield attempt
            break  # Success, exit retry loop
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Operation {operation_name} failed after {max_retries} retries: {e}")
                if fallback_value is not None:
                    logger.info(f"Using fallback value for {operation_name}")
                    yield fallback_value
                    break
                else:
                    raise
            else:
                logger.warning(f"Operation {operation_name} attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff


def validate_with_recovery(validator: AdvancedValidator, 
                         data: Any, 
                         validator_names: List[str],
                         auto_fix: bool = False) -> Tuple[List[ValidationResult], Any]:
    """Validate data with automatic recovery attempts."""
    original_data = data
    results = validator.validate(data, validator_names)
    
    if auto_fix and any(not r.is_valid for r in results):
        logger.info("Attempting automatic fixes for validation errors")
        
        # Attempt basic fixes
        fixed_data = data
        
        for result in results:
            if not result.is_valid and result.fix_suggestions:
                # Apply simple fixes
                if 'Remove suspicious content' in result.fix_suggestions:
                    if isinstance(fixed_data, str):
                        fixed_data = re.sub(r'<script[^>]*>.*?</script>', '', fixed_data, flags=re.IGNORECASE)
                        fixed_data = re.sub(r'javascript:', '', fixed_data, flags=re.IGNORECASE)
                
                elif 'Truncate text' in result.fix_suggestions:
                    if isinstance(fixed_data, str) and len(fixed_data) > 10000:
                        fixed_data = fixed_data[:10000] + '...'
        
        # Re-validate after fixes
        if fixed_data != original_data:
            results = validator.validate(fixed_data, validator_names)
            return results, fixed_data
    
    return results, original_data


# Global validator instance
_global_validator: Optional[AdvancedValidator] = None


def get_validator(config: ValidationConfig = None) -> AdvancedValidator:
    """Get global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = AdvancedValidator(config)
    return _global_validator


def validate_document(document: Dict[str, Any], 
                     strict: bool = True) -> List[ValidationResult]:
    """Validate document with global validator."""
    config = ValidationConfig(strict_mode=strict)
    validator = get_validator(config)
    return validator.validate_document(document)


def validate_safely(data: Any, 
                   validator_names: List[str],
                   fallback_valid: bool = False) -> bool:
    """Safely validate data, returning fallback on errors."""
    try:
        validator = get_validator()
        results = validator.validate(data, validator_names)
        return all(r.is_valid for r in results)
    except Exception as e:
        logger.error(f"Validation failed safely: {e}")
        return fallback_valid