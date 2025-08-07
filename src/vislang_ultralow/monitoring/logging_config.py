"""Advanced logging configuration with structured logging and performance tracking."""

import logging
import logging.handlers
import json
import time
import threading
import os
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback
import uuid
from contextlib import contextmanager


@dataclass
class LogContext:
    """Context information for structured logging."""
    request_id: str = None
    user_id: str = None
    session_id: str = None
    operation: str = None
    component: str = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'thread_name': record.threadName,
            'process_id': record.process,
        }
        
        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            log_entry['context'] = record.context.to_dict()
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'context']:
                log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry['stack_trace'] = record.stack_info
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_times = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to log record."""
        # Add memory usage
        try:
            import psutil
            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except Exception:
            record.memory_mb = 0
            record.cpu_percent = 0
        
        # Add timing information if available
        thread_id = threading.current_thread().ident
        if hasattr(record, 'operation_start'):
            duration = time.time() - record.operation_start
            record.operation_duration = duration
        
        return True


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self):
        super().__init__()
        self.local = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        context = getattr(self.local, 'context', None)
        if context:
            record.context = context
        else:
            record.context = LogContext()
        
        return True
    
    def set_context(self, context: LogContext):
        """Set context for current thread."""
        self.local.context = context
    
    def clear_context(self):
        """Clear context for current thread."""
        if hasattr(self.local, 'context'):
            delattr(self.local, 'context')


class LoggingManager:
    """Advanced logging manager with context and performance tracking."""
    
    def __init__(self):
        self.context_filter = ContextFilter()
        self.performance_filter = PerformanceFilter()
        self.loggers = {}
        self.configured = False
        
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_file: Optional[str] = None,
                     max_file_size: int = 100 * 1024 * 1024,  # 100MB
                     backup_count: int = 5,
                     structured_logging: bool = True,
                     console_output: bool = True) -> logging.Logger:
        """Setup comprehensive logging configuration."""
        
        if self.configured:
            return logging.getLogger('vislang_ultralow')
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if structured_logging:
                console_formatter = StructuredFormatter(include_context=True)
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(self.context_filter)
            console_handler.addFilter(self.performance_filter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            if structured_logging:
                file_formatter = StructuredFormatter(include_context=True)
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(self.context_filter)
            file_handler.addFilter(self.performance_filter)
            root_logger.addHandler(file_handler)
        
        # Application logger
        app_logger = logging.getLogger('vislang_ultralow')
        app_logger.setLevel(getattr(logging, log_level.upper()))
        
        self.configured = True
        app_logger.info("Logging system initialized", extra={'component': 'logging'})
        
        return app_logger
    
    @contextmanager
    def log_context(self, **context_kwargs):
        """Context manager for structured logging context."""
        context = LogContext(**context_kwargs)
        original_context = getattr(self.context_filter.local, 'context', None)
        
        try:
            self.context_filter.set_context(context)
            yield context
        finally:
            if original_context:
                self.context_filter.set_context(original_context)
            else:
                self.context_filter.clear_context()
    
    @contextmanager
    def log_operation(self, operation_name: str, logger: logging.Logger = None, **context_kwargs):
        """Context manager for logging operations with timing."""
        if logger is None:
            logger = logging.getLogger('vislang_ultralow')
        
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        context_kwargs.update({
            'operation': operation_name,
            'operation_id': operation_id
        })
        
        with self.log_context(**context_kwargs) as context:
            logger.info(f"Starting operation: {operation_name}", 
                       extra={'operation_start': start_time})
            
            try:
                yield context
                
                duration = time.time() - start_time
                logger.info(f"Completed operation: {operation_name}", 
                           extra={'operation_duration': duration, 'operation_status': 'success'})
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed operation: {operation_name}: {str(e)}", 
                           extra={'operation_duration': duration, 'operation_status': 'error'},
                           exc_info=True)
                raise
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a configured logger instance."""
        if not self.configured:
            self.setup_logging()
        
        logger_name = name or 'vislang_ultralow'
        
        if logger_name not in self.loggers:
            logger = logging.getLogger(logger_name)
            self.loggers[logger_name] = logger
        
        return self.loggers[logger_name]


class AuditLogger:
    """Specialized logger for audit trails and security events."""
    
    def __init__(self, log_file: str = "audit.log"):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Ensure audit logs are always written
        if not self.logger.handlers:
            audit_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10,
                encoding='utf-8'
            )
            
            audit_formatter = StructuredFormatter(include_context=True)
            audit_handler.setFormatter(audit_formatter)
            
            self.logger.addHandler(audit_handler)
            self.logger.propagate = False  # Don't send to other handlers
    
    def log_user_action(self, user_id: str, action: str, resource: str, 
                       result: str = "success", metadata: Dict[str, Any] = None):
        """Log user action for audit trail."""
        self.logger.info(f"User action: {action}", extra={
            'event_type': 'user_action',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
    
    def log_security_event(self, event_type: str, severity: str, description: str,
                          source_ip: str = None, user_id: str = None,
                          metadata: Dict[str, Any] = None):
        """Log security event."""
        self.logger.warning(f"Security event: {event_type}", extra={
            'event_type': 'security_event',
            'security_event_type': event_type,
            'severity': severity,
            'description': description,
            'source_ip': source_ip,
            'user_id': user_id,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
    
    def log_data_access(self, user_id: str, resource_type: str, resource_id: str,
                       access_type: str, result: str = "success"):
        """Log data access for compliance."""
        self.logger.info(f"Data access: {access_type}", extra={
            'event_type': 'data_access',
            'user_id': user_id,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'access_type': access_type,
            'result': result,
            'timestamp': time.time()
        })


class ErrorTracker:
    """Error tracking and analysis system."""
    
    def __init__(self, max_errors: int = 1000):
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}
        self.lock = threading.Lock()
        
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track an error with context information."""
        error_info = {
            'timestamp': time.time(),
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'thread_id': threading.current_thread().ident,
            'process_id': os.getpid()
        }
        
        with self.lock:
            # Add to errors list
            self.errors.append(error_info)
            
            # Maintain max size
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors:]
            
            # Update error counts
            error_type = type(error).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_error_summary(self, time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Get error summary for specified time range."""
        cutoff_time = time.time() - time_range_seconds
        
        with self.lock:
            recent_errors = [e for e in self.errors if e['timestamp'] >= cutoff_time]
            
            # Error counts by type
            error_types = {}
            for error in recent_errors:
                error_type = error['type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Most recent errors
            recent_errors_sorted = sorted(recent_errors, key=lambda x: x['timestamp'], reverse=True)
            
            return {
                'total_errors': len(recent_errors),
                'error_types': error_types,
                'recent_errors': recent_errors_sorted[:10],  # Last 10 errors
                'time_range_seconds': time_range_seconds,
                'error_rate': len(recent_errors) / time_range_seconds if time_range_seconds > 0 else 0
            }
    
    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top error types by frequency."""
        with self.lock:
            sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
            return [
                {'type': error_type, 'count': count}
                for error_type, count in sorted_errors[:limit]
            ]


# Global instances
logging_manager = LoggingManager()
audit_logger = AuditLogger()
error_tracker = ErrorTracker()


def setup_logging(**kwargs) -> logging.Logger:
    """Setup logging with default configuration."""
    return logging_manager.setup_logging(**kwargs)


def get_logger(name: str = None) -> logging.Logger:
    """Get a configured logger instance."""
    return logging_manager.get_logger(name)