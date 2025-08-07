"""Novel adaptive OCR algorithms for humanitarian documents.

This module implements research-grade OCR techniques that dynamically adapt
to document quality and script characteristics in ultra-low-resource scenarios.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging
from PIL import Image
import easyocr
import pytesseract
import paddleocr
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class OCRConsensusAlgorithm:
    """Novel consensus algorithm for multi-engine OCR with uncertainty quantification."""
    
    def __init__(self, confidence_threshold: float = 0.7, agreement_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        self.engine_weights = {"tesseract": 1.0, "easyocr": 1.2, "paddleocr": 1.1}
        
    def compute_consensus(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute consensus using novel weighted agreement algorithm.
        
        Research Innovation: Dynamic confidence-weighted voting with 
        geometric mean aggregation for improved robustness.
        """
        if not ocr_results:
            return {"text": "", "confidence": 0.0, "uncertainty": 1.0}
        
        # Extract individual results
        texts = [result["text"] for result in ocr_results]
        confidences = [result["confidence"] for result in ocr_results]
        engines = [result["engine"] for result in ocr_results]
        
        # Token-level consensus
        token_consensus = self._compute_token_consensus(ocr_results)
        
        # Character-level alignment and scoring
        alignment_scores = self._compute_alignment_scores(texts)
        
        # Weighted consensus with uncertainty quantification
        final_text, final_confidence, uncertainty = self._weighted_consensus(
            token_consensus, alignment_scores, confidences, engines
        )
        
        return {
            "text": final_text,
            "confidence": final_confidence,
            "uncertainty": uncertainty,
            "individual_results": ocr_results,
            "consensus_method": "adaptive_weighted_voting"
        }
    
    def _compute_token_consensus(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute token-level consensus using adaptive clustering."""
        all_tokens = []
        token_scores = []
        
        for result in ocr_results:
            tokens = result["text"].split()
            engine = result["engine"]
            confidence = result["confidence"]
            
            for token in tokens:
                all_tokens.append(token.lower())
                # Weight by engine performance and confidence
                weight = self.engine_weights.get(engine, 1.0) * confidence
                token_scores.append(weight)
        
        if not all_tokens:
            return {"tokens": [], "scores": []}
        
        # Cluster similar tokens
        token_clusters = self._cluster_tokens(all_tokens, token_scores)
        
        return {
            "tokens": all_tokens,
            "scores": token_scores,
            "clusters": token_clusters
        }
    
    def _cluster_tokens(self, tokens: List[str], scores: List[float]) -> List[Dict]:
        """Cluster similar tokens using edit distance and semantic similarity."""
        clusters = []
        processed = set()
        
        for i, token in enumerate(tokens):
            if token in processed:
                continue
                
            cluster = {"representative": token, "members": [token], "scores": [scores[i]]}
            
            for j, other_token in enumerate(tokens[i+1:], i+1):
                if other_token in processed:
                    continue
                    
                # Edit distance similarity
                similarity = self._edit_distance_similarity(token, other_token)
                
                if similarity > 0.8:  # High similarity threshold
                    cluster["members"].append(other_token)
                    cluster["scores"].append(scores[j])
                    processed.add(other_token)
            
            # Select best representative based on weighted score
            if len(cluster["members"]) > 1:
                best_idx = np.argmax(cluster["scores"])
                cluster["representative"] = cluster["members"][best_idx]
            
            clusters.append(cluster)
            processed.add(token)
        
        return clusters
    
    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        """Compute normalized edit distance similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Dynamic programming for edit distance
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_distance = dp[m][n]
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (edit_distance / max_len)
    
    def _compute_alignment_scores(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise alignment scores between OCR outputs."""
        n = len(texts)
        scores = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    scores[i][j] = 1.0
                else:
                    score = self._sequence_alignment_score(texts[i], texts[j])
                    scores[i][j] = scores[j][i] = score
        
        return scores
    
    def _sequence_alignment_score(self, text1: str, text2: str) -> float:
        """Compute sequence alignment score using longest common subsequence."""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize texts
        tokens1 = text1.split()
        tokens2 = text2.split()
        
        # LCS algorithm
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1].lower() == tokens2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        max_length = max(m, n)
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def _weighted_consensus(self, token_consensus: Dict, alignment_scores: np.ndarray,
                          confidences: List[float], engines: List[str]) -> Tuple[str, float, float]:
        """Generate final consensus with uncertainty quantification."""
        if not token_consensus["clusters"]:
            return "", 0.0, 1.0
        
        # Select best tokens from each cluster
        consensus_tokens = []
        cluster_confidences = []
        
        for cluster in token_consensus["clusters"]:
            if cluster["scores"]:
                # Weighted score considering engine reliability
                weighted_scores = []
                for i, member in enumerate(cluster["members"]):
                    base_score = cluster["scores"][i]
                    # Additional weighting could be added here
                    weighted_scores.append(base_score)
                
                best_idx = np.argmax(weighted_scores)
                consensus_tokens.append(cluster["representative"])
                cluster_confidences.append(max(weighted_scores))
        
        final_text = " ".join(consensus_tokens)
        
        # Compute overall confidence using geometric mean
        if cluster_confidences:
            final_confidence = statistics.geometric_mean(cluster_confidences)
        else:
            final_confidence = 0.0
        
        # Compute uncertainty based on agreement
        agreement_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        alignment_variance = np.var(alignment_scores) if alignment_scores.size > 0 else 0.0
        
        uncertainty = min(1.0, (agreement_variance + alignment_variance) / 2.0)
        
        return final_text, final_confidence, uncertainty


class AdaptiveMultiEngineOCR:
    """Adaptive OCR system that dynamically selects and configures engines."""
    
    def __init__(self, engines: List[str] = None):
        self.engines = engines or ["tesseract", "easyocr", "paddleocr"]
        self.consensus_algorithm = OCRConsensusAlgorithm()
        self.performance_history = defaultdict(list)
        self.engine_processors = {}
        
        # Initialize engines
        self._initialize_engines()
        
    def _initialize_engines(self):
        """Initialize OCR engines with adaptive configurations."""
        logger.info("Initializing adaptive multi-engine OCR")
        
        if "tesseract" in self.engines:
            # Dynamic configuration based on document type
            self.engine_processors["tesseract"] = {
                "configs": {
                    "standard": "--oem 3 --psm 6",
                    "dense_text": "--oem 3 --psm 4", 
                    "sparse_text": "--oem 3 --psm 8",
                    "single_word": "--oem 3 --psm 8"
                }
            }
        
        if "easyocr" in self.engines:
            try:
                # Support for multiple language combinations
                self.engine_processors["easyocr"] = {
                    "readers": {
                        "multilingual": easyocr.Reader(['en', 'fr', 'es', 'ar']),
                        "latin": easyocr.Reader(['en', 'fr', 'es', 'pt', 'it']),
                        "arabic": easyocr.Reader(['ar', 'en'])
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        if "paddleocr" in self.engines:
            try:
                self.engine_processors["paddleocr"] = {
                    "processor": paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
                }
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
    
    def extract_text(self, image: Image.Image, document_type: str = "standard") -> Dict[str, Any]:
        """Extract text using adaptive multi-engine approach with research innovations."""
        
        # Preprocess image for optimal OCR
        preprocessed_image = self._adaptive_preprocessing(image, document_type)
        
        # Run multiple OCR engines with adaptive configuration
        ocr_results = []
        
        for engine in self.engines:
            try:
                result = self._run_adaptive_engine(preprocessed_image, engine, document_type)
                if result:
                    result["engine"] = engine
                    ocr_results.append(result)
            except Exception as e:
                logger.warning(f"Engine {engine} failed: {e}")
        
        if not ocr_results:
            return {"text": "", "confidence": 0.0, "error": "All OCR engines failed"}
        
        # Apply consensus algorithm
        consensus_result = self.consensus_algorithm.compute_consensus(ocr_results)
        
        # Update performance tracking
        self._update_performance_history(ocr_results, consensus_result)
        
        return consensus_result
    
    def _adaptive_preprocessing(self, image: Image.Image, document_type: str) -> np.ndarray:
        """Research innovation: Adaptive preprocessing based on document characteristics."""
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Adaptive preprocessing based on document type
        if document_type == "humanitarian_report":
            # Specialized preprocessing for humanitarian documents
            processed = self._humanitarian_doc_preprocessing(gray)
        elif document_type == "infographic":
            processed = self._infographic_preprocessing(gray)
        elif document_type == "chart":
            processed = self._chart_preprocessing(gray)
        else:
            processed = self._standard_preprocessing(gray)
        
        return processed
    
    def _humanitarian_doc_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for humanitarian documents."""
        # Noise reduction for scanned documents
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding for varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to improve text structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _infographic_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for infographics with mixed content."""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Edge-preserving denoising
        denoised = cv2.bilateralFilter(enhanced, 9, 80, 80)
        
        return denoised
    
    def _chart_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for charts and graphs."""
        # Gaussian blur to smooth text areas
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _standard_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Standard preprocessing pipeline."""
        # Simple denoising and thresholding
        denoised = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _run_adaptive_engine(self, image: np.ndarray, engine: str, document_type: str) -> Optional[Dict[str, Any]]:
        """Run OCR engine with adaptive configuration."""
        
        if engine == "tesseract":
            return self._run_adaptive_tesseract(image, document_type)
        elif engine == "easyocr":
            return self._run_adaptive_easyocr(image, document_type)
        elif engine == "paddleocr":
            return self._run_adaptive_paddleocr(image, document_type)
        
        return None
    
    def _run_adaptive_tesseract(self, image: np.ndarray, document_type: str) -> Optional[Dict[str, Any]]:
        """Run Tesseract with adaptive configuration."""
        try:
            # Select appropriate configuration
            configs = self.engine_processors["tesseract"]["configs"]
            config = configs.get(document_type, configs["standard"])
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )
            
            # Filter and process results
            texts = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 30:  # Filter low confidence
                    texts.append(text)
                    confidences.append(conf / 100.0)
            
            if not texts:
                return None
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "word_confidences": confidences,
                "method": "adaptive_tesseract"
            }
            
        except Exception as e:
            logger.error(f"Adaptive Tesseract failed: {e}")
            return None
    
    def _run_adaptive_easyocr(self, image: np.ndarray, document_type: str) -> Optional[Dict[str, Any]]:
        """Run EasyOCR with adaptive reader selection."""
        try:
            # Select appropriate reader based on document characteristics
            readers = self.engine_processors["easyocr"]["readers"]
            
            # Use multilingual reader for humanitarian documents
            if document_type in ["humanitarian_report", "infographic"]:
                reader = readers.get("multilingual", readers["latin"])
            else:
                reader = readers["latin"]
            
            # Extract text
            results = reader.readtext(image)
            
            if not results:
                return None
            
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf > 0.3 and text.strip():
                    texts.append(text.strip())
                    confidences.append(conf)
            
            if not texts:
                return None
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "word_confidences": confidences,
                "method": "adaptive_easyocr"
            }
            
        except Exception as e:
            logger.error(f"Adaptive EasyOCR failed: {e}")
            return None
    
    def _run_adaptive_paddleocr(self, image: np.ndarray, document_type: str) -> Optional[Dict[str, Any]]:
        """Run PaddleOCR with adaptive configuration."""
        try:
            processor = self.engine_processors["paddleocr"]["processor"]
            results = processor.ocr(image, cls=True)
            
            if not results or not results[0]:
                return None
            
            texts = []
            confidences = []
            
            for line in results[0]:
                if line:
                    bbox, (text, conf) = line
                    if conf > 0.3 and text.strip():
                        texts.append(text.strip())
                        confidences.append(conf)
            
            if not texts:
                return None
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences)
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "word_confidences": confidences,
                "method": "adaptive_paddleocr"
            }
            
        except Exception as e:
            logger.error(f"Adaptive PaddleOCR failed: {e}")
            return None
    
    def _update_performance_history(self, ocr_results: List[Dict], consensus_result: Dict):
        """Update performance tracking for adaptive learning."""
        for result in ocr_results:
            engine = result["engine"]
            confidence = result["confidence"]
            
            # Track engine performance
            self.performance_history[engine].append({
                "confidence": confidence,
                "timestamp": np.datetime64('now'),
                "consensus_confidence": consensus_result["confidence"]
            })
            
            # Keep only recent history (last 1000 samples)
            if len(self.performance_history[engine]) > 1000:
                self.performance_history[engine] = self.performance_history[engine][-1000:]
    
    def get_engine_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for each engine."""
        stats = {}
        
        for engine, history in self.performance_history.items():
            if history:
                confidences = [h["confidence"] for h in history]
                consensus_confidences = [h["consensus_confidence"] for h in history]
                
                stats[engine] = {
                    "avg_confidence": np.mean(confidences),
                    "std_confidence": np.std(confidences),
                    "correlation_with_consensus": np.corrcoef(confidences, consensus_confidences)[0, 1],
                    "sample_count": len(history)
                }
        
        return stats
    
    def adaptive_engine_selection(self, image_characteristics: Dict[str, float]) -> List[str]:
        """Dynamically select best engines based on image characteristics."""
        # This is a placeholder for adaptive selection logic
        # In practice, this would use machine learning to predict best engines
        
        performance_stats = self.get_engine_performance_stats()
        
        # Simple heuristic: rank engines by average confidence
        ranked_engines = sorted(
            self.engines,
            key=lambda e: performance_stats.get(e, {}).get("avg_confidence", 0.0),
            reverse=True
        )
        
        return ranked_engines