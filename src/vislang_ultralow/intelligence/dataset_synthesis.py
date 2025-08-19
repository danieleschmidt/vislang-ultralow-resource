"""Intelligent dataset synthesis for ultra-low-resource scenarios.

Generation 1 Enhancement: Automated creation of synthetic vision-language datasets
using novel algorithms for data augmentation and cross-lingual transfer.
"""

import logging
import json
import hashlib
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from datetime import datetime
import threading
import queue
import math

# Conditional imports with fallbacks
try:
    import numpy as np
    from scipy import stats
except ImportError:
    np = None
    stats = None

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    Image = ImageDraw = ImageFont = ImageFilter = ImageEnhance = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generate synthetic vision-language datasets for ultra-low-resource languages."""
    
    def __init__(self, target_languages: List[str], base_templates: Optional[Dict] = None):
        self.target_languages = target_languages
        self.base_templates = base_templates or self._create_default_templates()
        self.generation_history = defaultdict(list)
        self.quality_metrics = {}
        
        # Synthesis strategies
        self.synthesis_strategies = {
            'template_variation': self._template_based_synthesis,
            'cross_lingual_adaptation': self._cross_lingual_synthesis,
            'visual_augmentation': self._visual_augmentation_synthesis,
            'contextual_expansion': self._contextual_expansion_synthesis
        }
        
        # Humanitarian domain lexicon
        self.humanitarian_vocabulary = self._build_humanitarian_vocabulary()
        
        logger.info(f"Initialized SyntheticDatasetGenerator for languages: {target_languages}")
    
    def generate_synthetic_dataset(self, num_samples: int = 1000, 
                                 strategy: str = 'mixed') -> Dict[str, Any]:
        """Generate synthetic vision-language dataset using specified strategy."""
        logger.info(f"Generating {num_samples} synthetic samples using {strategy} strategy")
        
        synthetic_samples = []
        generation_stats = defaultdict(int)
        
        if strategy == 'mixed':
            # Use all strategies with equal weight
            strategies = list(self.synthesis_strategies.keys())
            samples_per_strategy = num_samples // len(strategies)
            
            for strategy_name in strategies:
                strategy_func = self.synthesis_strategies[strategy_name]
                strategy_samples = strategy_func(samples_per_strategy)
                synthetic_samples.extend(strategy_samples)
                generation_stats[strategy_name] = len(strategy_samples)
        else:
            # Use single strategy
            if strategy in self.synthesis_strategies:
                strategy_func = self.synthesis_strategies[strategy]
                synthetic_samples = strategy_func(num_samples)
                generation_stats[strategy] = len(synthetic_samples)
            else:
                raise ValueError(f"Unknown synthesis strategy: {strategy}")
        
        # Add metadata and quality scores
        for sample in synthetic_samples:
            sample['synthetic'] = True
            sample['generation_timestamp'] = datetime.now().isoformat()
            sample['quality_score'] = self._compute_sample_quality(sample)
        
        # Store generation history
        self.generation_history[strategy].append({
            'timestamp': datetime.now(),
            'num_samples': len(synthetic_samples),
            'strategy': strategy,
            'stats': dict(generation_stats)
        })
        
        dataset = {
            'samples': synthetic_samples,
            'metadata': {
                'total_samples': len(synthetic_samples),
                'strategy': strategy,
                'generation_stats': dict(generation_stats),
                'target_languages': self.target_languages,
                'quality_distribution': self._analyze_quality_distribution(synthetic_samples)
            }
        }
        
        logger.info(f"Generated {len(synthetic_samples)} synthetic samples")
        return dataset
    
    def _template_based_synthesis(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples using template variation."""
        samples = []
        
        for _ in range(num_samples):
            # Select random template
            template_category = random.choice(list(self.base_templates.keys()))
            template = random.choice(self.base_templates[template_category])
            
            # Select target language
            target_lang = random.choice(self.target_languages)
            
            # Generate variations
            sample = self._create_template_variation(template, target_lang, template_category)
            samples.append(sample)
        
        return samples
    
    def _cross_lingual_synthesis(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples using cross-lingual adaptation."""
        samples = []
        
        for _ in range(num_samples):
            # Start with English template
            base_template = self._get_random_english_template()
            
            # Adapt to target language
            target_lang = random.choice(self.target_languages)
            
            sample = self._adapt_cross_lingually(base_template, target_lang)
            samples.append(sample)
        
        return samples
    
    def _visual_augmentation_synthesis(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples with visual augmentation."""
        samples = []
        
        for _ in range(num_samples):
            # Create base visual content
            visual_data = self._generate_synthetic_visual()
            
            # Generate corresponding text in multiple languages
            target_lang = random.choice(self.target_languages)
            
            sample = self._create_visually_grounded_sample(visual_data, target_lang)
            samples.append(sample)
        
        return samples
    
    def _contextual_expansion_synthesis(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples using contextual expansion."""
        samples = []
        
        for _ in range(num_samples):
            # Start with core humanitarian concept
            core_concept = random.choice(list(self.humanitarian_vocabulary.keys()))
            
            # Expand with context
            target_lang = random.choice(self.target_languages)
            
            sample = self._expand_contextually(core_concept, target_lang)
            samples.append(sample)
        
        return samples
    
    def _create_template_variation(self, template: Dict, target_lang: str, 
                                 category: str) -> Dict[str, Any]:
        """Create variation of a template for target language."""
        # Extract template structure
        text_template = template.get('text', '')
        visual_description = template.get('visual', '')
        instruction_type = template.get('type', 'description')
        
        # Apply linguistic variations
        varied_text = self._apply_linguistic_variations(text_template, target_lang)
        
        # Generate corresponding visual
        visual_data = self._generate_visual_from_description(visual_description)
        
        sample = {
            'id': hashlib.md5(f"{varied_text}_{target_lang}_{datetime.now()}".encode()).hexdigest()[:12],
            'text': varied_text,
            'language': target_lang,
            'visual_data': visual_data,
            'instruction_type': instruction_type,
            'category': category,
            'synthesis_method': 'template_variation',
            'template_id': template.get('id', 'unknown'),
            'difficulty_level': self._estimate_difficulty(varied_text, target_lang)
        }
        
        return sample
    
    def _adapt_cross_lingually(self, base_template: Dict, target_lang: str) -> Dict[str, Any]:
        """Adapt English template to target language."""
        english_text = base_template.get('text', '')
        
        # Translate using linguistic patterns
        translated_text = self._translate_with_patterns(english_text, 'en', target_lang)
        
        # Adapt cultural context
        culturally_adapted = self._adapt_cultural_context(translated_text, target_lang)
        
        # Generate or adapt visual content
        visual_data = base_template.get('visual_data')
        if not visual_data:
            visual_data = self._generate_visual_from_text(culturally_adapted)
        
        sample = {
            'id': hashlib.md5(f"{culturally_adapted}_{target_lang}_{datetime.now()}".encode()).hexdigest()[:12],
            'text': culturally_adapted,
            'language': target_lang,
            'visual_data': visual_data,
            'instruction_type': base_template.get('type', 'description'),
            'category': 'cross_lingual_adaptation',
            'synthesis_method': 'cross_lingual_adaptation',
            'source_language': 'en',
            'cultural_adaptation': True,
            'difficulty_level': self._estimate_difficulty(culturally_adapted, target_lang)
        }
        
        return sample
    
    def _create_visually_grounded_sample(self, visual_data: Dict, target_lang: str) -> Dict[str, Any]:
        """Create sample grounded in visual content."""
        # Generate text description from visual
        visual_description = self._describe_visual_content(visual_data, target_lang)
        
        # Create instruction variations
        instruction_variations = self._generate_instruction_variations(visual_description, target_lang)
        
        # Select best instruction
        selected_instruction = random.choice(instruction_variations)
        
        sample = {
            'id': hashlib.md5(f"{selected_instruction}_{target_lang}_{datetime.now()}".encode()).hexdigest()[:12],
            'text': selected_instruction,
            'language': target_lang,
            'visual_data': visual_data,
            'instruction_type': 'visual_description',
            'category': 'visual_grounded',
            'synthesis_method': 'visual_augmentation',
            'visual_complexity': self._assess_visual_complexity(visual_data),
            'grounding_strength': self._assess_grounding_strength(selected_instruction, visual_data),
            'difficulty_level': self._estimate_difficulty(selected_instruction, target_lang)
        }
        
        return sample
    
    def _expand_contextually(self, core_concept: str, target_lang: str) -> Dict[str, Any]:
        """Expand core concept with contextual information."""
        # Get concept vocabulary
        concept_vocab = self.humanitarian_vocabulary.get(core_concept, [])
        
        # Generate contextual expansion
        context_elements = random.sample(concept_vocab, min(3, len(concept_vocab)))
        
        # Create expanded description
        expanded_text = self._create_contextual_description(core_concept, context_elements, target_lang)
        
        # Generate appropriate visual
        visual_data = self._generate_concept_visual(core_concept, context_elements)
        
        sample = {
            'id': hashlib.md5(f"{expanded_text}_{target_lang}_{datetime.now()}".encode()).hexdigest()[:12],
            'text': expanded_text,
            'language': target_lang,
            'visual_data': visual_data,
            'instruction_type': 'contextual_description',
            'category': 'contextual_expansion',
            'synthesis_method': 'contextual_expansion',
            'core_concept': core_concept,
            'context_elements': context_elements,
            'context_richness': len(context_elements),
            'difficulty_level': self._estimate_difficulty(expanded_text, target_lang)
        }
        
        return sample
    
    def _create_default_templates(self) -> Dict[str, List[Dict]]:
        """Create default templates for humanitarian scenarios."""
        return {
            'emergency_response': [
                {
                    'id': 'emr_001',
                    'text': 'Emergency food distribution in progress at {location}',
                    'visual': 'People queuing for food packages',
                    'type': 'situation_report'
                },
                {
                    'id': 'emr_002', 
                    'text': 'Medical supplies delivered to {facility}',
                    'visual': 'Boxes of medical supplies',
                    'type': 'logistics_update'
                }
            ],
            'refugee_assistance': [
                {
                    'id': 'ref_001',
                    'text': 'Shelter construction completed for {number} families',
                    'visual': 'Rows of temporary shelters',
                    'type': 'progress_report'
                },
                {
                    'id': 'ref_002',
                    'text': 'Water sanitation systems operational in {camp_name}',
                    'visual': 'Water distribution point',
                    'type': 'infrastructure_update'
                }
            ],
            'disaster_assessment': [
                {
                    'id': 'das_001',
                    'text': 'Damage assessment shows {percentage}% of buildings affected',
                    'visual': 'Aerial view of damaged area',
                    'type': 'assessment_report'
                },
                {
                    'id': 'das_002',
                    'text': 'Critical infrastructure {status} in affected regions',
                    'visual': 'Infrastructure damage overview',
                    'type': 'infrastructure_assessment'
                }
            ]
        }
    
    def _build_humanitarian_vocabulary(self) -> Dict[str, List[str]]:
        """Build humanitarian domain vocabulary."""
        return {
            'emergency_response': [
                'evacuation', 'rescue', 'first_aid', 'shelter', 'food_distribution',
                'medical_assistance', 'emergency_communication', 'coordination'
            ],
            'refugee_assistance': [
                'registration', 'protection', 'resettlement', 'family_reunification',
                'legal_assistance', 'education', 'livelihood_support'
            ],
            'disaster_recovery': [
                'reconstruction', 'infrastructure_repair', 'community_rebuilding',
                'economic_recovery', 'psychosocial_support', 'risk_reduction'
            ],
            'health_crisis': [
                'vaccination', 'disease_surveillance', 'quarantine', 'treatment',
                'prevention', 'health_education', 'medical_equipment'
            ],
            'water_sanitation': [
                'clean_water', 'waste_management', 'hygiene_promotion',
                'latrine_construction', 'water_treatment', 'sewage_system'
            ]
        }
    
    def _apply_linguistic_variations(self, text: str, target_lang: str) -> str:
        """Apply linguistic variations for target language."""
        # Simple linguistic transformations
        variations = {
            'sw': lambda t: f"Kikufunika cha {t.lower()}",  # Swahili pattern
            'am': lambda t: f"የ{t} ሁኔታ",  # Amharic pattern
            'ha': lambda t: f"Halin {t.lower()}"  # Hausa pattern
        }
        
        if target_lang in variations:
            return variations[target_lang](text)
        
        return f"[{target_lang}] {text}"
    
    def _translate_with_patterns(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using linguistic patterns."""
        # Simple pattern-based translation
        patterns = {
            ('en', 'sw'): {
                'emergency': 'dharura',
                'food': 'chakula', 
                'water': 'maji',
                'shelter': 'makazi',
                'medical': 'kitiba'
            },
            ('en', 'am'): {
                'emergency': 'ድንገተኛ',
                'food': 'ምግብ',
                'water': 'ውሃ', 
                'shelter': 'መጠለያ',
                'medical': 'ሕክምና'
            },
            ('en', 'ha'): {
                'emergency': 'gaggawa',
                'food': 'abinci',
                'water': 'ruwa',
                'shelter': 'matsuguni', 
                'medical': 'likita'
            }
        }
        
        translation_dict = patterns.get((source_lang, target_lang), {})
        
        translated = text.lower()
        for english_word, target_word in translation_dict.items():
            translated = translated.replace(english_word, target_word)
        
        return translated
    
    def _adapt_cultural_context(self, text: str, target_lang: str) -> str:
        """Adapt text for cultural context."""
        # Add cultural context adaptations
        cultural_adaptations = {
            'sw': f"Kulingana na utamaduni wa Kiafrika: {text}",
            'am': f"በኢትዮጵያ አውድ ውስጥ: {text}", 
            'ha': f"A cikin al'adun Hausa: {text}"
        }
        
        return cultural_adaptations.get(target_lang, text)
    
    def _generate_synthetic_visual(self) -> Dict[str, Any]:
        """Generate synthetic visual content."""
        # Simple visual content generation
        visual_types = ['chart', 'map', 'infographic', 'photograph', 'diagram']
        visual_type = random.choice(visual_types)
        
        return {
            'type': visual_type,
            'width': random.randint(400, 800),
            'height': random.randint(300, 600),
            'color_scheme': random.choice(['warm', 'cool', 'neutral']),
            'complexity': random.choice(['simple', 'moderate', 'complex']),
            'dominant_colors': [f"color_{i}" for i in range(random.randint(2, 5))],
            'content_description': f"Synthetic {visual_type} for humanitarian scenario",
            'accessibility_score': random.uniform(0.6, 0.9)
        }
    
    def _generate_visual_from_description(self, description: str) -> Dict[str, Any]:
        """Generate visual content from text description."""
        # Extract visual elements from description
        elements = description.split()
        
        return {
            'type': 'generated_from_description',
            'source_description': description,
            'extracted_elements': elements[:5],  # Top 5 elements
            'estimated_complexity': len(elements) / 10.0,
            'visual_features': {
                'has_people': 'people' in description.lower(),
                'has_buildings': any(word in description.lower() for word in ['building', 'structure', 'facility']),
                'has_vehicles': any(word in description.lower() for word in ['vehicle', 'truck', 'car']),
                'outdoor_scene': any(word in description.lower() for word in ['outdoor', 'field', 'area'])
            }
        }
    
    def _generate_visual_from_text(self, text: str) -> Dict[str, Any]:
        """Generate visual content from text."""
        return self._generate_visual_from_description(text)
    
    def _describe_visual_content(self, visual_data: Dict, target_lang: str) -> str:
        """Generate text description of visual content."""
        visual_type = visual_data.get('type', 'unknown')
        complexity = visual_data.get('complexity', 'simple')
        
        descriptions = {
            'chart': f"Jedwali linaloonyesha takwimu muhimu",  # Swahili example
            'map': f"Ramani ya eneo la msaada wa kibinadamu",
            'infographic': f"Mchoro wa habari za kifupi",
            'photograph': f"Picha ya mazingira halisi",
            'diagram': f"Mchoro wa maelezo"
        }
        
        base_description = descriptions.get(visual_type, f"Maudhui ya {visual_type}")
        
        # Add complexity information
        if complexity == 'complex':
            base_description += " yenye maelezo makuu"
        
        return base_description
    
    def _generate_instruction_variations(self, description: str, target_lang: str) -> List[str]:
        """Generate instruction variations for description."""
        base_instructions = [
            f"Eleza kile unachokiona katika {description}",  # Swahili
            f"Toa maelezo ya {description}",
            f"Fafanua yaliyomo kwenye {description}"
        ]
        
        return base_instructions
    
    def _create_contextual_description(self, core_concept: str, context_elements: List[str], 
                                    target_lang: str) -> str:
        """Create contextual description from concept and elements."""
        # Simple contextual description generation
        context_text = " na ".join(context_elements)  # Swahili conjunction
        
        return f"Hali ya {core_concept} ikijumuisha {context_text}"
    
    def _generate_concept_visual(self, concept: str, context_elements: List[str]) -> Dict[str, Any]:
        """Generate visual for concept with context."""
        return {
            'type': 'concept_visual',
            'core_concept': concept,
            'context_elements': context_elements,
            'visual_representation': 'humanitarian_scenario',
            'context_richness': len(context_elements),
            'conceptual_clarity': random.uniform(0.7, 0.95)
        }
    
    def _get_random_english_template(self) -> Dict[str, Any]:
        """Get random English template."""
        categories = list(self.base_templates.keys())
        category = random.choice(categories)
        template = random.choice(self.base_templates[category])
        
        return {
            'text': template['text'],
            'visual': template['visual'],
            'type': template['type'],
            'category': category
        }
    
    def _compute_sample_quality(self, sample: Dict[str, Any]) -> float:
        """Compute quality score for generated sample."""
        factors = []
        
        # Text quality factors
        text = sample.get('text', '')
        if text:
            factors.append(min(1.0, len(text.split()) / 20.0))  # Word count factor
            factors.append(0.8 if any(char.isalpha() for char in text) else 0.3)  # Has letters
        
        # Visual quality factors
        visual_data = sample.get('visual_data', {})
        if visual_data:
            factors.append(visual_data.get('accessibility_score', 0.5))
            factors.append(visual_data.get('conceptual_clarity', 0.5))
        
        # Language specificity
        if sample.get('language') in self.target_languages:
            factors.append(0.9)
        else:
            factors.append(0.5)
        
        # Synthesis method bonus
        method_bonus = {
            'template_variation': 0.8,
            'cross_lingual_adaptation': 0.9,
            'visual_augmentation': 0.85,
            'contextual_expansion': 0.88
        }
        
        synthesis_method = sample.get('synthesis_method', 'unknown')
        factors.append(method_bonus.get(synthesis_method, 0.6))
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _analyze_quality_distribution(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze quality distribution of generated samples."""
        quality_scores = [sample.get('quality_score', 0.0) for sample in samples]
        
        if not quality_scores:
            return {}
        
        if np is not None:
            return {
                'mean_quality': float(np.mean(quality_scores)),
                'std_quality': float(np.std(quality_scores)),
                'min_quality': float(np.min(quality_scores)),
                'max_quality': float(np.max(quality_scores)),
                'median_quality': float(np.median(quality_scores))
            }
        else:
            # Fallback without numpy
            return {
                'mean_quality': sum(quality_scores) / len(quality_scores),
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores),
                'num_samples': len(quality_scores)
            }
    
    def _estimate_difficulty(self, text: str, language: str) -> str:
        """Estimate difficulty level of text."""
        word_count = len(text.split())
        
        if word_count < 5:
            return 'easy'
        elif word_count < 15:
            return 'medium'
        else:
            return 'hard'
    
    def _assess_visual_complexity(self, visual_data: Dict) -> str:
        """Assess complexity of visual content."""
        return visual_data.get('complexity', 'simple')
    
    def _assess_grounding_strength(self, text: str, visual_data: Dict) -> float:
        """Assess how well text is grounded in visual content."""
        # Simple heuristic for grounding strength
        visual_elements = visual_data.get('extracted_elements', [])
        
        if not visual_elements:
            return 0.5
        
        # Count mentions of visual elements in text
        matches = sum(1 for element in visual_elements if element.lower() in text.lower())
        
        return min(1.0, matches / len(visual_elements))
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics."""
        total_samples = sum(
            sum(session['num_samples'] for session in sessions)
            for sessions in self.generation_history.values()
        )
        
        strategy_usage = defaultdict(int)
        for strategy, sessions in self.generation_history.items():
            strategy_usage[strategy] = sum(session['num_samples'] for session in sessions)
        
        return {
            'total_generated_samples': total_samples,
            'strategy_usage': dict(strategy_usage),
            'target_languages': self.target_languages,
            'available_strategies': list(self.synthesis_strategies.keys()),
            'generation_sessions': len(self.generation_history),
            'vocabulary_size': sum(len(vocab) for vocab in self.humanitarian_vocabulary.values())
        }