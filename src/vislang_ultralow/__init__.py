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
# Database and cache components - available on demand to avoid dependency issues
# from .database import (
#     DatabaseManager, get_session
# )
# from .database.models import Document, Image, DatasetItem, TrainingRun
# from .database.repositories import DocumentRepository, ImageRepository, DatasetRepository, TrainingRepository
# from .cache import CacheManager, get_cache_manager, cached, cache_key, invalidate_cache
# from .cli import cli
# Research modules - import on demand to avoid dependency issues
# from .research import (
#     AdaptiveMultiEngineOCR, OCRConsensusAlgorithm,
#     CrossLingualAlignmentModel, ZeroShotCrossLingual,
#     HumanitarianSceneAnalyzer, CrisisEventDetector,
#     AdaptiveModalFusion, UncertaintyAwareFusion,
#     LowResourceDataAugmentor, SyntheticDataGenerator,
#     HumanitarianVLBenchmark, CrossLingualEvaluator
# )

__all__ = [
    "DatasetBuilder",
    "HumanitarianScraper", 
    "VisionLanguageTrainer",
    "VisionLanguageDataset",
    "__version__"
    # Research modules - available on demand
    # "AdaptiveMultiEngineOCR",
    # "OCRConsensusAlgorithm", 
    # "CrossLingualAlignmentModel",
    # "ZeroShotCrossLingual",
    # "HumanitarianSceneAnalyzer",
    # "CrisisEventDetector",
    # "AdaptiveModalFusion",
    # "UncertaintyAwareFusion",
    # "LowResourceDataAugmentor",
    # "SyntheticDataGenerator",
    # "HumanitarianVLBenchmark",
    # "CrossLingualEvaluator"
]