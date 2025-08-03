"""VisLang-UltraLow-Resource: Vision-Language models for humanitarian applications.

A framework for building multilingual vision-language datasets from humanitarian
reports and training models for ultra-low-resource languages.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

from .dataset import DatasetBuilder
from .scraper import HumanitarianScraper
from .trainer import VisionLanguageTrainer, VisionLanguageDataset
from .database import (
    DatabaseManager, get_session, Document, Image, DatasetItem, TrainingRun,
    DocumentRepository, ImageRepository, DatasetRepository
)
from .cache import CacheManager, get_cache_manager, cached, cache_key, invalidate_cache

__all__ = [
    "DatasetBuilder",
    "HumanitarianScraper", 
    "VisionLanguageTrainer",
    "VisionLanguageDataset",
    "DatabaseManager",
    "get_session",
    "Document",
    "Image", 
    "DatasetItem",
    "TrainingRun",
    "DocumentRepository",
    "ImageRepository",
    "DatasetRepository",
    "CacheManager",
    "get_cache_manager",
    "cached",
    "cache_key",
    "invalidate_cache",
    "__version__",
]