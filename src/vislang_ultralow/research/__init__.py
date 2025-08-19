"""Research modules for novel algorithmic approaches."""

# Import research modules with fallback for missing dependencies
try:
    from .adaptive_ocr import AdaptiveMultiEngineOCR, OCRConsensusAlgorithm
    _adaptive_ocr_available = True
except ImportError as e:
    import logging
    logging.warning(f"Could not import adaptive_ocr: {e}")
    _adaptive_ocr_available = False
    
    # Create fallback classes
    class AdaptiveMultiEngineOCR:
        def __init__(self, *args, **kwargs):
            pass
        def extract_text(self, *args, **kwargs):
            return {'text': 'fallback text', 'confidence': 0.5}
        def get_engine_performance_stats(self):
            return {}
    
    class OCRConsensusAlgorithm:
        def __init__(self, *args, **kwargs):
            pass
        def compute_consensus(self, *args, **kwargs):
            return {'text': 'consensus text', 'confidence': 0.5}

try:
    from .cross_lingual_alignment import (
        ZeroShotCrossLingual, 
        CrossLingualAlignmentModel,
        HierarchicalCrossLingualAlignment
    )
    _cross_lingual_available = True
except ImportError as e:
    import logging
    logging.warning(f"Could not import cross_lingual_alignment: {e}")
    _cross_lingual_available = False
    
    # Create fallback classes
    class ZeroShotCrossLingual:
        def __init__(self, *args, **kwargs):
            pass
        def align_cross_lingual(self, text, target_lang):
            return f"[{target_lang}] {text}"
        def learn_alignment(self, *args, **kwargs):
            return {'alignment_matrix': [], 'quality_metrics': {}, 'num_samples': 0}
        def compute_cross_lingual_similarity(self, *args, **kwargs):
            return 0.5
        def align_texts(self, texts, source_lang, target_lang):
            return [f"[{target_lang}] {text}" for text in texts]
    
    class CrossLingualAlignmentModel:
        def __init__(self, *args, **kwargs):
            pass
    
    class HierarchicalCrossLingualAlignment:
        def __init__(self, *args, **kwargs):
            pass

__all__ = [
    "AdaptiveMultiEngineOCR",
    "OCRConsensusAlgorithm", 
    "ZeroShotCrossLingual",
    "CrossLingualAlignmentModel",
    "HierarchicalCrossLingualAlignment"
]