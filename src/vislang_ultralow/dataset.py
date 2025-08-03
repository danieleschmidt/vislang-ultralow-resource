"""Dataset building functionality for vision-language models."""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import hashlib
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import pytesseract
import easyocr
import paddleocr
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import pipeline
import re
from collections import defaultdict
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


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
        self.output_dir = output_dir or Path("./datasets")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OCR engines
        self.ocr_engines = ocr_engines or ["tesseract", "easyocr", "paddleocr"]
        self._initialize_ocr_engines()
        
        # Initialize translation pipeline
        self._initialize_translation()
        
        # Instruction templates
        self.instruction_templates = self._load_instruction_templates()
        
        logger.info(f"Initialized DatasetBuilder for languages: {target_languages}")
    
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
        logger.info(f"Starting dataset building process with {len(documents)} documents")
        
        # Process documents and extract vision-language pairs
        dataset_items = []
        
        for doc in documents:
            try:
                items = self._process_document(
                    doc, include_infographics, include_maps, include_charts
                )
                dataset_items.extend(items)
                logger.debug(f"Processed document {doc.get('title', 'Unknown')} -> {len(items)} items")
            except Exception as e:
                logger.error(f"Error processing document {doc.get('url', 'Unknown')}: {e}")
                continue
        
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
                
                # Extract text from image using OCR
                ocr_results = self._extract_text_from_image(image_info)
                
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
    
    def _extract_text_from_image(self, image_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from image using multiple OCR engines."""
        ocr_results = []
        
        # For now, simulate OCR results since we don't have actual image data
        # In real implementation, you would load the image and run OCR
        
        # Simulate different OCR engine results
        if "tesseract" in self.ocr_processors:
            result = self._simulate_tesseract_ocr(image_info)
            if result:
                ocr_results.append(('tesseract', result))
        
        if "easyocr" in self.ocr_processors:
            result = self._simulate_easyocr_ocr(image_info)
            if result:
                ocr_results.append(('easyocr', result))
        
        if "paddleocr" in self.ocr_processors:
            result = self._simulate_paddleocr_ocr(image_info)
            if result:
                ocr_results.append(('paddleocr', result))
        
        if not ocr_results:
            return {'text': '', 'confidence': 0.0, 'engines': []}
        
        # Consensus OCR result
        return self._consensus_ocr_result(ocr_results)
    
    def _simulate_tesseract_ocr(self, image_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate Tesseract OCR results."""
        # This is a simulation - in real implementation, process actual image
        return {
            'text': f"Sample text extracted from {image_info.get('alt', 'image')}",
            'confidence': random.uniform(0.7, 0.95),
            'bbox': [[0, 0, 100, 20]]
        }
    
    def _simulate_easyocr_ocr(self, image_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate EasyOCR results."""
        return {
            'text': f"EasyOCR text from {image_info.get('alt', 'image')}",
            'confidence': random.uniform(0.75, 0.9),
            'bbox': [[0, 0, 100, 20]]
        }
    
    def _simulate_paddleocr_ocr(self, image_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate PaddleOCR results."""
        return {
            'text': f"PaddleOCR text from {image_info.get('alt', 'image')}",
            'confidence': random.uniform(0.8, 0.95),
            'bbox': [[0, 0, 100, 20]]
        }
    
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