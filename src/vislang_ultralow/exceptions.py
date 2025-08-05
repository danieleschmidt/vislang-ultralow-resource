"""Custom exceptions for VisLang-UltraLow-Resource."""


class VisLangError(Exception):
    """Base exception for VisLang-UltraLow-Resource."""
    pass


class ScrapingError(VisLangError):
    """Exception raised during document scraping."""
    
    def __init__(self, message: str, source: str = None, url: str = None):
        self.source = source
        self.url = url
        super().__init__(message)


class OCRError(VisLangError):
    """Exception raised during OCR processing."""
    
    def __init__(self, message: str, engine: str = None, confidence: float = None):
        self.engine = engine
        self.confidence = confidence
        super().__init__(message)


class DatasetError(VisLangError):
    """Exception raised during dataset building."""
    
    def __init__(self, message: str, items_processed: int = None):
        self.items_processed = items_processed
        super().__init__(message)


class TrainingError(VisLangError):
    """Exception raised during model training."""
    
    def __init__(self, message: str, epoch: int = None, step: int = None):
        self.epoch = epoch
        self.step = step
        super().__init__(message)


class DatabaseError(VisLangError):
    """Exception raised during database operations."""
    pass


class CacheError(VisLangError):
    """Exception raised during cache operations."""
    pass


class ValidationError(VisLangError):
    """Exception raised during data validation."""
    
    def __init__(self, message: str, field: str = None, value = None):
        self.field = field
        self.value = value
        super().__init__(message)


class ConfigurationError(VisLangError):
    """Exception raised for configuration issues."""
    pass


class ModelLoadError(VisLangError):
    """Exception raised when model loading fails."""
    
    def __init__(self, message: str, model_path: str = None):
        self.model_path = model_path
        super().__init__(message)


class InferenceError(VisLangError):
    """Exception raised during model inference."""
    
    def __init__(self, message: str, image_path: str = None):
        self.image_path = image_path
        super().__init__(message)


class QualityError(VisLangError):
    """Exception raised when quality thresholds are not met."""
    
    def __init__(self, message: str, score: float = None, threshold: float = None):
        self.score = score
        self.threshold = threshold
        super().__init__(message)


class ResourceError(VisLangError):
    """Exception raised for resource-related issues (memory, disk, etc.)."""
    
    def __init__(self, message: str, resource_type: str = None):
        self.resource_type = resource_type
        super().__init__(message)


class RateLimitError(VisLangError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: int = None):
        self.retry_after = retry_after
        super().__init__(message)
        
        
class TranslationError(VisLangError):
    """Exception raised during text translation."""
    
    def __init__(self, message: str, source_lang: str = None, target_lang: str = None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        super().__init__(message)