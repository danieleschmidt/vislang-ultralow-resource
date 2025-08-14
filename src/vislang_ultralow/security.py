"""Security utilities and validation for VisLang-UltraLow-Resource."""

import re
import os
import hashlib
import logging
import secrets
import hmac
import base64
import ipaddress
import socket
import ssl
import zlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from pathlib import Path
from urllib.parse import urlparse
import json
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
import jwt
from functools import wraps, lru_cache
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

# Security constants
MAX_PASSWORD_LENGTH = 128
MIN_PASSWORD_LENGTH = 8
MAX_TOKEN_AGE_SECONDS = 3600
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 900  # 15 minutes
RATE_LIMIT_WINDOW_SECONDS = 300  # 5 minutes
DEFAULT_ENCRYPTION_KEY_SIZE = 32
MAX_PAYLOAD_SIZE = 100 * 1024 * 1024  # 100MB
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Authentication methods supported."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"


@dataclass
class SecurityViolation:
    """Security violation details."""
    violation_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    source: str
    ip_address: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    risk_score: float = 0.0
    blocked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'violation_type': self.violation_type,
            'severity': self.severity,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'ip_address': self.ip_address,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'risk_score': self.risk_score,
            'blocked': self.blocked
        }


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    auth_method: Optional[AuthenticationMethod] = None
    permissions: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id': self.request_id,
            'auth_method': self.auth_method.value if self.auth_method else None,
            'permissions': list(self.permissions),
            'security_level': self.security_level.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int
    window_seconds: int
    burst_limit: int = 0
    exponential_backoff: bool = True
    per_ip: bool = True
    per_user: bool = True
    

@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3


class SecurityValidator:
    """Comprehensive security validation system."""
    
    def __init__(self, strict_mode: bool = True, enable_encryption: bool = True):
        """Initialize security validator.
        
        Args:
            strict_mode: If True, raises exceptions for security violations
            enable_encryption: If True, enables data encryption features
        """
        self.strict_mode = strict_mode
        self.enable_encryption = enable_encryption
        self.violations: List[SecurityViolation] = []
        self.blocked_patterns = self._load_blocked_patterns()
        self.allowed_domains = set()
        self.blocked_domains = set()
        self.max_content_length = 50 * 1024 * 1024  # 50MB
        self.rate_limits = defaultdict(lambda: deque())
        self.rate_limit_configs = defaultdict(lambda: RateLimitConfig(100, 300))
        self.failed_attempts = defaultdict(lambda: {'count': 0, 'last_attempt': None})
        self.circuit_breakers = defaultdict(lambda: CircuitBreakerState())
        self.active_sessions = weakref.WeakValueDictionary()
        self.blocked_ips = set()
        self.trusted_ips = set()
        self.honeypot_tokens = set()
        self._lock = threading.RLock()
        
        # Initialize encryption
        if self.enable_encryption:
            self.fernet = Fernet(ENCRYPTION_KEY if isinstance(ENCRYPTION_KEY, bytes) else ENCRYPTION_KEY.encode())
            self.password_hasher = bcrypt
        
        # Threat intelligence
        self.threat_patterns = self._load_threat_patterns()
        self.malware_signatures = self._load_malware_signatures()
        
        # Security metrics
        self.security_metrics = {
            'total_requests': 0,
            'blocked_requests': 0,
            'security_violations': 0,
            'authentication_failures': 0,
            'rate_limit_hits': 0
        }
        
        # Enhanced security configuration
        self.security_config = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'allowed_file_extensions': {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
                '.pdf', '.txt', '.json', '.csv', '.xml', '.tiff', '.svg'
            },
            'blocked_file_extensions': {
                '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
                '.js', '.vbs', '.ps1', '.sh', '.php', '.asp',
                '.jar', '.war', '.ear', '.class', '.dll', '.so'
            },
            'max_url_length': 2048,
            'max_filename_length': 255,
            'scan_for_malware': True,
            'validate_certificates': True,
            'block_private_ips': True,
            'enable_content_scanning': True,
            'max_request_size': MAX_PAYLOAD_SIZE,
            'require_https': True,
            'enable_csrf_protection': True,
            'enable_xss_protection': True,
            'enable_content_type_validation': True,
            'session_timeout_seconds': 3600,
            'max_concurrent_sessions': 10,
            'enable_geo_blocking': False,
            'allowed_countries': set(),
            'blocked_countries': set(),
            'enable_bot_detection': True,
            'min_request_interval': 0.1  # seconds
        }
        
        # Load security configurations from environment
        self._load_security_config_from_env()
        
        # Initialize threat detection
        self._initialize_threat_detection()
        
        logger.info("Advanced security validator initialized", extra={
            'strict_mode': strict_mode,
            'encryption_enabled': enable_encryption,
            'threat_patterns_loaded': len(self.threat_patterns),
            'malware_signatures_loaded': len(self.malware_signatures)
        })
    
    def _load_blocked_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for detecting malicious content."""
        return {
            'xss': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'data:text/html',
                r'on\w+\s*=',
                r'expression\s*\(',
                r'eval\s*\(',
                r'document\.cookie',
                r'window\.location'
            ],
            'sql_injection': [
                r'union\s+select',
                r'drop\s+table',
                r'delete\s+from',
                r'insert\s+into',
                r'update\s+.*set',
                r'exec\s*\(',
                r'sp_\w+',
                r'xp_\w+'
            ],
            'command_injection': [
                r';\s*rm\s',
                r';\s*cat\s',
                r';\s*ls\s',
                r';\s*pwd',
                r'\|\s*nc\s',
                r'&&\s*rm\s',
                r'`.*`',
                r'\$\(.*\)'
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'/etc/passwd',
                r'/etc/shadow',
                r'c:\\windows',
                r'\\\\.*\\c\$'
            ],
            'dangerous_protocols': [
                r'file://',
                r'ftp://',
                r'ldap://',
                r'gopher://',
                r'jar:',
                r'netdoc:'
            ]
        }
    
    def validate_url(self, url: str, context: str = "general") -> bool:
        """Validate URL for security issues.
        
        Args:
            url: URL to validate
            context: Context of URL usage
            
        Returns:
            True if URL is safe, False otherwise
        """
        if not url or not isinstance(url, str):
            self._add_violation(
                violation_type="INVALID_URL",
                severity="MEDIUM",
                message="Empty or invalid URL",
                context={"url": url, "context": context},
                source="url_validation"
            )
            return False
        
        # Length check
        if len(url) > self.security_config['max_url_length']:
            self._add_violation(
                violation_type="URL_TOO_LONG",
                severity="MEDIUM", 
                message=f"URL exceeds maximum length: {len(url)}",
                context={"url": url[:100] + "...", "length": len(url)},
                source="url_validation"
            )
            return False
        
        try:
            parsed = urlparse(url)
            
            # Protocol validation
            if parsed.scheme not in ('http', 'https'):
                self._add_violation(
                    violation_type="UNSAFE_PROTOCOL",
                    severity="HIGH",
                    message=f"Unsafe protocol: {parsed.scheme}",
                    context={"url": url, "scheme": parsed.scheme},
                    source="url_validation"
                )
                return False
            
            # Domain validation
            hostname = parsed.hostname
            if hostname:
                # Block private IPs in production
                if self.security_config['block_private_ips']:
                    if self._is_private_ip(hostname):
                        self._add_violation(
                            violation_type="PRIVATE_IP_ACCESS",
                            severity="HIGH",
                            message=f"Access to private IP blocked: {hostname}",
                            context={"url": url, "hostname": hostname},
                            source="url_validation"
                        )
                        return False
                
                # Check blocked domains
                if hostname in self.blocked_domains:
                    self._add_violation(
                        violation_type="BLOCKED_DOMAIN",
                        severity="HIGH",
                        message=f"Domain is blocked: {hostname}",
                        context={"url": url, "domain": hostname},
                        source="url_validation"
                    )
                    return False
                
                # Check allowed domains (if whitelist is active)
                if self.allowed_domains and hostname not in self.allowed_domains:
                    self._add_violation(
                        violation_type="DOMAIN_NOT_ALLOWED",
                        severity="MEDIUM",
                        message=f"Domain not in whitelist: {hostname}",
                        context={"url": url, "domain": hostname},
                        source="url_validation"
                    )
                    return False
            
            # Check for dangerous patterns in URL
            for pattern_type, patterns in self.blocked_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, url, re.IGNORECASE):
                        self._add_violation(
                            violation_type="DANGEROUS_URL_PATTERN",
                            severity="HIGH",
                            message=f"Dangerous pattern detected: {pattern_type}",
                            context={"url": url, "pattern": pattern, "type": pattern_type},
                            source="url_validation"
                        )
                        return False
            
            return True
            
        except Exception as e:
            self._add_violation(
                violation_type="URL_PARSE_ERROR",
                severity="MEDIUM",
                message=f"Failed to parse URL: {str(e)}",
                context={"url": url, "error": str(e)},
                source="url_validation"
            )
            return False
    
    def validate_content(self, content: str, content_type: str = "text") -> bool:
        """Validate content for security issues.
        
        Args:
            content: Content to validate
            content_type: Type of content (text, html, json, etc.)
            
        Returns:
            True if content is safe, False otherwise
        """
        if not isinstance(content, str):
            return False
        
        # Size check
        if len(content) > self.max_content_length:
            self._add_violation(
                violation_type="CONTENT_TOO_LARGE",
                severity="MEDIUM",
                message=f"Content exceeds maximum size: {len(content)}",
                context={"size": len(content), "type": content_type},
                source="content_validation"
            )
            return False
        
        # Check for malicious patterns
        content_lower = content.lower()
        
        for pattern_type, patterns in self.blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    self._add_violation(
                        violation_type="MALICIOUS_CONTENT",
                        severity="HIGH",
                        message=f"Malicious pattern detected: {pattern_type}",
                        context={
                            "pattern": pattern,
                            "type": pattern_type,
                            "content_type": content_type,
                            "content_preview": content[:100] + "..." if len(content) > 100 else content
                        },
                        source="content_validation"
                    )
                    return False
        
        # Content-specific validation
        if content_type == "json":
            return self._validate_json_content(content)
        elif content_type == "html":
            return self._validate_html_content(content)
        
        return True
    
    def validate_file_path(self, file_path: Path) -> bool:
        """Validate file path for security issues.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if path is safe, False otherwise
        """
        if not isinstance(file_path, (str, Path)):
            return False
        
        path_str = str(file_path)
        
        # Length check
        if len(path_str) > self.security_config['max_filename_length']:
            self._add_violation(
                violation_type="PATH_TOO_LONG",
                severity="MEDIUM",
                message=f"Path exceeds maximum length: {len(path_str)}",
                context={"path": path_str[:100] + "...", "length": len(path_str)},
                source="path_validation"
            )
            return False
        
        # Path traversal check
        if '..' in path_str or path_str.startswith('/'):
            self._add_violation(
                violation_type="PATH_TRAVERSAL",
                severity="HIGH",
                message="Path traversal attempt detected",
                context={"path": path_str},
                source="path_validation"
            )
            return False
        
        # File extension check
        if isinstance(file_path, Path):
            extension = file_path.suffix.lower()
        else:
            extension = Path(path_str).suffix.lower()
        
        if extension in self.security_config['blocked_file_extensions']:
            self._add_violation(
                violation_type="BLOCKED_FILE_TYPE",
                severity="HIGH",
                message=f"Blocked file extension: {extension}",
                context={"path": path_str, "extension": extension},
                source="path_validation"
            )
            return False
        
        # Check for suspicious file names
        suspicious_names = [
            'passwd', 'shadow', 'hosts', 'config', 'secret',
            'key', 'token', 'password', 'credential'
        ]
        
        filename_lower = Path(path_str).name.lower()
        for suspicious in suspicious_names:
            if suspicious in filename_lower:
                self._add_violation(
                    violation_type="SUSPICIOUS_FILENAME",
                    severity="MEDIUM",
                    message=f"Suspicious filename pattern: {suspicious}",
                    context={"path": path_str, "pattern": suspicious},
                    source="path_validation"
                )
                return False
        
        return True
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate document for security issues.
        
        Args:
            document: Document to validate
            
        Returns:
            True if document is safe, False otherwise
        """
        if not isinstance(document, dict):
            return False
        
        # Validate URLs in document
        for url_field in ['url', 'source_url', 'origin']:
            if url_field in document:
                if not self.validate_url(document[url_field], f"document.{url_field}"):
                    return False
        
        # Validate content
        if 'content' in document:
            if not self.validate_content(document['content'], 'text'):
                return False
        
        # Validate images
        if 'images' in document:
            for i, image in enumerate(document.get('images', [])):
                if isinstance(image, dict):
                    # Validate image URLs
                    for url_field in ['src', 'url']:
                        if url_field in image:
                            if not self.validate_url(image[url_field], f"image[{i}].{url_field}"):
                                return False
                    
                    # Validate image metadata
                    if not self._validate_image_metadata(image, i):
                        return False
        
        # Validate metadata
        if 'metadata' in document:
            metadata_str = json.dumps(document['metadata'])
            if not self.validate_content(metadata_str, 'json'):
                return False
        
        return True
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_seconds: int = 300) -> bool:
        """Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier for rate limiting (IP, user ID, etc.)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            True if within limits, False if rate limited
        """
        now = time.time()
        cutoff = now - window_seconds
        
        # Clean old entries
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if timestamp > cutoff
        ]
        
        # Check current count
        current_count = len(self.rate_limits[identifier])
        
        if current_count >= max_requests:
            self._add_violation(
                violation_type="RATE_LIMIT_EXCEEDED",
                severity="MEDIUM",
                message=f"Rate limit exceeded: {current_count}/{max_requests}",
                context={
                    "identifier": identifier,
                    "current_count": current_count,
                    "max_requests": max_requests,
                    "window_seconds": window_seconds
                },
                source="rate_limiting"
            )
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text to prevent injection attacks.
        
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
        text = text.replace('&', '&amp;')
        
        return text.strip()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Secure random token
        """
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash sensitive data with salt.
        
        Args:
            data: Data to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        salted_data = f"{data}{salt}"
        hash_value = hashlib.sha256(salted_data.encode()).hexdigest()
        
        return hash_value, salt
    
    def _validate_json_content(self, content: str) -> bool:
        """Validate JSON content for security issues."""
        try:
            data = json.loads(content)
            
            # Check for excessive nesting (JSON bomb)
            max_depth = 10
            if self._json_depth(data) > max_depth:
                self._add_violation(
                    violation_type="JSON_TOO_DEEP",
                    severity="HIGH",
                    message=f"JSON nesting exceeds maximum depth: {max_depth}",
                    context={"depth": self._json_depth(data)},
                    source="json_validation"
                )
                return False
            
            return True
            
        except json.JSONDecodeError as e:
            self._add_violation(
                violation_type="INVALID_JSON",
                severity="LOW",
                message=f"Invalid JSON: {str(e)}",
                context={"error": str(e)},
                source="json_validation"
            )
            return False
    
    def _validate_html_content(self, content: str) -> bool:
        """Validate HTML content for security issues."""
        # Check for dangerous HTML elements
        dangerous_tags = [
            'script', 'object', 'embed', 'iframe', 'frame',
            'frameset', 'applet', 'link', 'meta', 'style'
        ]
        
        for tag in dangerous_tags:
            pattern = f'<{tag}[^>]*>'
            if re.search(pattern, content, re.IGNORECASE):
                self._add_violation(
                    violation_type="DANGEROUS_HTML_TAG",
                    severity="HIGH",
                    message=f"Dangerous HTML tag detected: {tag}",
                    context={"tag": tag},
                    source="html_validation"
                )
                return False
        
        return True
    
    def _validate_image_metadata(self, image: Dict[str, Any], index: int) -> bool:
        """Validate image metadata for security issues."""
        # Check dimensions for reasonableness
        width = image.get('width')
        height = image.get('height')
        
        if width is not None and isinstance(width, (int, float)):
            if width > 10000 or width <= 0:
                self._add_violation(
                    violation_type="SUSPICIOUS_IMAGE_DIMENSIONS",
                    severity="MEDIUM",
                    message=f"Suspicious image width: {width}",
                    context={"width": width, "image_index": index},
                    source="image_validation"
                )
                return False
        
        if height is not None and isinstance(height, (int, float)):
            if height > 10000 or height <= 0:
                self._add_violation(
                    violation_type="SUSPICIOUS_IMAGE_DIMENSIONS",
                    severity="MEDIUM",
                    message=f"Suspicious image height: {height}",
                    context={"height": height, "image_index": index},
                    source="image_validation"
                )
                return False
        
        # Validate alt text
        alt_text = image.get('alt_text', image.get('alt', ''))
        if alt_text and not self.validate_content(alt_text, 'text'):
            return False
        
        return True
    
    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname is a private IP address."""
        import ipaddress
        
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private
        except ValueError:
            # Not an IP address, hostname is fine
            return False
    
    def _json_depth(self, data: Any, depth: int = 0) -> int:
        """Calculate JSON nesting depth."""
        if isinstance(data, dict):
            if not data:
                return depth
            return max(self._json_depth(value, depth + 1) for value in data.values())
        elif isinstance(data, list):
            if not data:
                return depth
            return max(self._json_depth(item, depth + 1) for item in data)
        else:
            return depth
    
    def _add_violation(
        self,
        violation_type: str,
        severity: str,
        message: str,
        context: Dict[str, Any],
        source: str
    ) -> None:
        """Add security violation to tracking."""
        violation = SecurityViolation(
            violation_type=violation_type,
            severity=severity,
            message=message,
            context=context,
            timestamp=datetime.now(),
            source=source
        )
        
        self.violations.append(violation)
        
        # Log based on severity
        if severity == "CRITICAL":
            logger.critical(f"SECURITY: {message}")
        elif severity == "HIGH":
            logger.error(f"SECURITY: {message}")
        elif severity == "MEDIUM":
            logger.warning(f"SECURITY: {message}")
        else:
            logger.info(f"SECURITY: {message}")
        
        # Raise exception in strict mode for high/critical violations
        if self.strict_mode and severity in ["HIGH", "CRITICAL"]:
            from .exceptions import ValidationError
            raise ValidationError(f"Security violation: {message}")
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of security violations."""
        if not self.violations:
            return {"total": 0, "by_severity": {}, "by_type": {}}
        
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for violation in self.violations:
            by_severity[violation.severity] += 1
            by_type[violation.violation_type] += 1
        
        return {
            "total": len(self.violations),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "latest_violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "message": v.message,
                    "timestamp": v.timestamp.isoformat(),
                    "source": v.source
                }
                for v in self.violations[-10:]  # Last 10 violations
            ]
        }
    
    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violations.clear()
        logger.info("Security violation history cleared")


# Global security validator instance
_security_validator: Optional[SecurityValidator] = None


def get_security_validator(strict_mode: bool = True) -> SecurityValidator:
    """Get global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator(strict_mode=strict_mode)
    return _security_validator


def validate_document_security(document: Dict[str, Any], strict: bool = True) -> bool:
    """Validate document for security issues.
    
    Args:
        document: Document to validate
        strict: Enable strict mode validation
        
    Returns:
        True if document is secure
    """
    validator = get_security_validator(strict_mode=strict)
    return validator.validate_document(document)


def sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent attacks.
    
    Args:
        text: User input to sanitize
        
    Returns:
        Sanitized text
    """
    validator = get_security_validator()
    return validator.sanitize_input(text)


class SecurityManager:
    """Main security management interface."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize security manager.
        
        Args:
            strict_mode: Enable strict mode validation
        """
        self.validator = SecurityValidator(strict_mode=strict_mode)
        self.strict_mode = strict_mode
    
    def validate_url(self, url: str) -> bool:
        """Validate URL for security."""
        return self.validator.validate_url(url)
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input."""
        return self.validator.sanitize_input(text)
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate document for security."""
        return self.validator.validate_document(document)
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100) -> bool:
        """Check rate limiting."""
        return self.validator.check_rate_limit(identifier, max_requests)
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get security violations summary."""
        return self.validator.get_violations_summary()
    
    def clear_violations(self) -> None:
        """Clear violation history."""
        self.validator.clear_violations()