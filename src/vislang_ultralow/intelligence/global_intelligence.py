"""Global intelligence systems for cross-cultural and multi-regional optimization.

Generation 4: Intelligent systems that adapt to cultural contexts and learn
across global deployments for humanitarian applications.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class CulturalContext:
    """Cultural context information for adaptation."""
    region: str
    language_codes: List[str]
    writing_direction: str  # 'ltr', 'rtl', 'ttb'
    cultural_values: Dict[str, float]
    communication_style: str  # 'direct', 'indirect', 'contextual'
    data_sensitivity: str  # 'high', 'medium', 'low'
    
class GlobalIntelligenceCoordinator:
    """Coordinates intelligence across global deployments."""
    
    def __init__(self):
        self.regional_intelligence = {}
        self.global_patterns = {}
        self.coordination_active = False
        
    def register_region(self, region_id: str, intelligence_node):
        """Register regional intelligence node."""
        self.regional_intelligence[region_id] = intelligence_node
        logger.info(f"Registered intelligence node for region: {region_id}")
    
    def coordinate_learning(self) -> Dict[str, Any]:
        """Coordinate learning across all regions."""
        if not self.regional_intelligence:
            return {}
        
        # Aggregate insights from all regions
        global_insights = self._aggregate_regional_insights()
        
        # Identify global patterns
        patterns = self._identify_global_patterns(global_insights)
        
        # Distribute learnings back to regions
        self._distribute_global_learnings(patterns)
        
        return {
            "regions_coordinated": len(self.regional_intelligence),
            "global_patterns": len(patterns),
            "coordination_timestamp": datetime.now().isoformat()
        }
    
    def _aggregate_regional_insights(self) -> Dict[str, Any]:
        """Aggregate insights from all regional nodes."""
        aggregated = {
            "performance_metrics": [],
            "cultural_adaptations": [],
            "language_patterns": [],
            "humanitarian_insights": []
        }
        
        for region_id, node in self.regional_intelligence.items():
            if hasattr(node, 'get_regional_insights'):
                insights = node.get_regional_insights()
                
                # Add region identifier to insights
                for category, data in insights.items():
                    if category in aggregated:
                        regional_data = {"region": region_id, "data": data}
                        aggregated[category].append(regional_data)
        
        return aggregated
    
    def _identify_global_patterns(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify patterns that span multiple regions."""
        patterns = []
        
        # Cross-regional performance patterns
        perf_pattern = self._analyze_cross_regional_performance(
            insights.get("performance_metrics", [])
        )
        if perf_pattern:
            patterns.append(perf_pattern)
        
        # Universal vs. culture-specific adaptations
        cultural_pattern = self._analyze_cultural_universals(
            insights.get("cultural_adaptations", [])
        )
        if cultural_pattern:
            patterns.append(cultural_pattern)
        
        return patterns
    
    def _analyze_cross_regional_performance(self, perf_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """Analyze performance patterns across regions."""
        if len(perf_data) < 2:
            return None
        
        # Extract performance metrics by region
        regional_performance = {}
        for entry in perf_data:
            region = entry["region"]
            data = entry["data"]
            
            if isinstance(data, dict) and "accuracy" in data:
                regional_performance[region] = data["accuracy"]
        
        if len(regional_performance) < 2:
            return None
        
        # Calculate performance statistics
        performances = list(regional_performance.values())
        avg_performance = sum(performances) / len(performances)
        
        # Identify high and low performing regions
        high_performers = [r for r, p in regional_performance.items() if p > avg_performance * 1.1]
        low_performers = [r for r, p in regional_performance.items() if p < avg_performance * 0.9]
        
        return {
            "type": "cross_regional_performance",
            "average_performance": avg_performance,
            "high_performing_regions": high_performers,
            "low_performing_regions": low_performers,
            "performance_variance": max(performances) - min(performances),
            "actionable_insights": self._generate_performance_insights(
                high_performers, low_performers, regional_performance
            )
        }
    
    def _generate_performance_insights(self, high_performers: List[str], 
                                     low_performers: List[str],
                                     regional_performance: Dict[str, float]) -> List[str]:
        """Generate actionable insights from performance analysis."""
        insights = []
        
        if high_performers:
            insights.append(f"Regions {high_performers} show superior performance - analyze their configurations")
        
        if low_performers:
            insights.append(f"Regions {low_performers} need optimization - consider knowledge transfer")
        
        # Identify performance gaps
        if regional_performance:
            max_perf = max(regional_performance.values())
            min_perf = min(regional_performance.values())
            
            if (max_perf - min_perf) > 0.1:  # 10% performance gap
                insights.append("Significant performance gap detected - standardization opportunity")
        
        return insights

class CrossRegionalLearning:
    """Learns patterns across different regions and cultures."""
    
    def __init__(self):
        self.regional_models = {}
        self.transfer_learning_cache = {}
        self.cross_regional_patterns = []
        
    def learn_from_region(self, region_id: str, cultural_context: CulturalContext,
                         performance_data: Dict[str, Any]):
        """Learn from regional deployment data."""
        logger.info(f"Learning from region: {region_id}")
        
        # Store regional model
        self.regional_models[region_id] = {
            "cultural_context": cultural_context,
            "performance_data": performance_data,
            "last_updated": datetime.now(),
            "adaptations": self._extract_regional_adaptations(cultural_context, performance_data)
        }
        
        # Update cross-regional patterns
        self._update_cross_regional_patterns()
    
    def suggest_adaptations_for_region(self, target_region: str, 
                                     cultural_context: CulturalContext) -> Dict[str, Any]:
        """Suggest adaptations for a new region based on similar regions."""
        logger.info(f"Suggesting adaptations for region: {target_region}")
        
        # Find similar regions
        similar_regions = self._find_similar_regions(cultural_context)
        
        if not similar_regions:
            return self._get_default_adaptations(cultural_context)
        
        # Aggregate successful adaptations from similar regions
        suggested_adaptations = self._aggregate_adaptations(similar_regions)
        
        # Apply cultural-specific modifications
        cultural_adaptations = self._apply_cultural_modifications(
            suggested_adaptations, cultural_context
        )
        
        return cultural_adaptations
    
    def _find_similar_regions(self, target_context: CulturalContext) -> List[str]:
        """Find regions with similar cultural contexts."""
        similar_regions = []
        
        for region_id, model_data in self.regional_models.items():
            stored_context = model_data["cultural_context"]
            similarity = self._calculate_cultural_similarity(target_context, stored_context)
            
            if similarity > 0.7:  # High similarity threshold
                similar_regions.append(region_id)
        
        return similar_regions
    
    def _calculate_cultural_similarity(self, context1: CulturalContext, 
                                     context2: CulturalContext) -> float:
        """Calculate similarity between cultural contexts."""
        similarity_factors = []
        
        # Language family similarity
        lang_overlap = len(set(context1.language_codes) & set(context2.language_codes))
        lang_similarity = lang_overlap / max(len(context1.language_codes), len(context2.language_codes), 1)
        similarity_factors.append(lang_similarity * 0.3)
        
        # Writing direction similarity
        if context1.writing_direction == context2.writing_direction:
            similarity_factors.append(0.2)
        
        # Communication style similarity
        if context1.communication_style == context2.communication_style:
            similarity_factors.append(0.2)
        
        # Cultural values similarity
        if context1.cultural_values and context2.cultural_values:
            value_similarities = []
            shared_values = set(context1.cultural_values.keys()) & set(context2.cultural_values.keys())
            
            for value in shared_values:
                v1, v2 = context1.cultural_values[value], context2.cultural_values[value]
                value_sim = 1.0 - abs(v1 - v2)  # Assuming values are normalized 0-1
                value_similarities.append(value_sim)
            
            if value_similarities:
                avg_value_sim = sum(value_similarities) / len(value_similarities)
                similarity_factors.append(avg_value_sim * 0.3)
        
        return sum(similarity_factors)
    
    def _extract_regional_adaptations(self, cultural_context: CulturalContext,
                                    performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract successful adaptations from regional data."""
        adaptations = {}
        
        # Language-specific adaptations
        if cultural_context.language_codes:
            adaptations["language_models"] = {
                "primary_languages": cultural_context.language_codes[:3],  # Top 3
                "multilingual_strategy": "balanced" if len(cultural_context.language_codes) > 2 else "focused"
            }
        
        # Writing direction adaptations
        if cultural_context.writing_direction == "rtl":
            adaptations["ui_layout"] = {
                "text_alignment": "right",
                "interface_mirroring": True,
                "reading_flow": "right_to_left"
            }
        elif cultural_context.writing_direction == "ttb":
            adaptations["ui_layout"] = {
                "text_orientation": "vertical",
                "scroll_direction": "vertical"
            }
        
        # Data sensitivity adaptations
        if cultural_context.data_sensitivity == "high":
            adaptations["privacy_controls"] = {
                "local_processing": True,
                "data_minimization": True,
                "explicit_consent": True
            }
        
        # Performance-based adaptations
        if "accuracy" in performance_data and performance_data["accuracy"] < 0.8:
            adaptations["model_tuning"] = {
                "increase_training_data": True,
                "cultural_fine_tuning": True,
                "local_validation": True
            }
        
        return adaptations

class CulturalContextAdapter:
    """Adapts system behavior to cultural contexts."""
    
    def __init__(self):
        self.cultural_profiles = {}
        self.adaptation_rules = self._initialize_adaptation_rules()
        
    def _initialize_adaptation_rules(self) -> Dict[str, Any]:
        """Initialize cultural adaptation rules."""
        return {
            "high_context_cultures": {
                "interface_style": "implicit",
                "information_density": "high",
                "relationship_emphasis": "strong"
            },
            "low_context_cultures": {
                "interface_style": "explicit",
                "information_density": "moderate",
                "task_focus": "direct"
            },
            "collectivist_cultures": {
                "collaboration_features": "prominent",
                "social_validation": "important",
                "group_decision_support": "enabled"
            },
            "individualist_cultures": {
                "personal_customization": "extensive",
                "individual_achievement": "highlighted",
                "self_service": "preferred"
            }
        }
    
    def adapt_interface(self, cultural_context: CulturalContext) -> Dict[str, Any]:
        """Adapt interface based on cultural context."""
        adaptations = {}
        
        # Apply communication style adaptations
        if cultural_context.communication_style == "indirect":
            adaptations.update(self.adaptation_rules["high_context_cultures"])
        elif cultural_context.communication_style == "direct":
            adaptations.update(self.adaptation_rules["low_context_cultures"])
        
        # Apply cultural value adaptations
        cultural_values = cultural_context.cultural_values or {}
        
        if cultural_values.get("collectivism", 0.5) > 0.6:
            adaptations.update(self.adaptation_rules["collectivist_cultures"])
        elif cultural_values.get("individualism", 0.5) > 0.6:
            adaptations.update(self.adaptation_rules["individualist_cultures"])
        
        # Language-specific adaptations
        adaptations["language_support"] = {
            "primary_languages": cultural_context.language_codes,
            "fallback_language": "en",
            "rtl_support": cultural_context.writing_direction == "rtl"
        }
        
        return adaptations
    
    def validate_cultural_appropriateness(self, content: Dict[str, Any],
                                        cultural_context: CulturalContext) -> Dict[str, Any]:
        """Validate content for cultural appropriateness."""
        validation_result = {
            "is_appropriate": True,
            "concerns": [],
            "suggestions": []
        }
        
        # Check for culturally sensitive elements
        if "images" in content:
            image_concerns = self._validate_image_content(content["images"], cultural_context)
            validation_result["concerns"].extend(image_concerns)
        
        if "text" in content:
            text_concerns = self._validate_text_content(content["text"], cultural_context)
            validation_result["concerns"].extend(text_concerns)
        
        # Update appropriateness based on concerns
        if validation_result["concerns"]:
            validation_result["is_appropriate"] = False
            validation_result["suggestions"] = self._generate_cultural_suggestions(
                validation_result["concerns"], cultural_context
            )
        
        return validation_result
    
    def _validate_image_content(self, images: List[Dict], 
                              cultural_context: CulturalContext) -> List[str]:
        """Validate image content for cultural appropriateness."""
        concerns = []
        
        # Basic cultural sensitivity checks
        for image in images:
            image_tags = image.get("tags", [])
            
            # Check for potentially sensitive content
            sensitive_tags = ["religious_symbols", "political_content", "cultural_stereotypes"]
            
            for tag in image_tags:
                if any(sensitive in tag.lower() for sensitive in sensitive_tags):
                    concerns.append(f"Image contains potentially sensitive content: {tag}")
        
        return concerns
    
    def _validate_text_content(self, text: str, 
                             cultural_context: CulturalContext) -> List[str]:
        """Validate text content for cultural appropriateness."""
        concerns = []
        
        # Language appropriateness
        primary_lang = cultural_context.language_codes[0] if cultural_context.language_codes else "en"
        
        # Simple heuristic: check if text contains appropriate language markers
        if primary_lang != "en" and len(text) > 100:
            # In real implementation, use language detection
            if text.isascii():  # Likely English for non-English primary language
                concerns.append(f"Text appears to be in English but primary language is {primary_lang}")
        
        return concerns
    
    def _generate_cultural_suggestions(self, concerns: List[str],
                                     cultural_context: CulturalContext) -> List[str]:
        """Generate suggestions to address cultural concerns."""
        suggestions = []
        
        for concern in concerns:
            if "sensitive content" in concern.lower():
                suggestions.append("Consider using culturally neutral imagery")
            elif "language" in concern.lower():
                primary_lang = cultural_context.language_codes[0] if cultural_context.language_codes else "en"
                suggestions.append(f"Provide content translation to {primary_lang}")
        
        return suggestions

class HumanitarianInsightEngine:
    """Generates insights for humanitarian applications."""
    
    def __init__(self):
        self.humanitarian_patterns = {}
        self.crisis_indicators = {}
        self.intervention_effectiveness = {}
        
    def analyze_humanitarian_data(self, data: Dict[str, Any], 
                                region: str) -> Dict[str, Any]:
        """Analyze humanitarian data for actionable insights."""
        logger.info(f"Analyzing humanitarian data for region: {region}")
        
        insights = {
            "region": region,
            "analysis_timestamp": datetime.now().isoformat(),
            "key_indicators": {},
            "recommendations": [],
            "urgency_level": "normal"
        }
        
        # Analyze key humanitarian indicators
        indicators = self._extract_humanitarian_indicators(data)
        insights["key_indicators"] = indicators
        
        # Generate recommendations
        recommendations = self._generate_humanitarian_recommendations(indicators, region)
        insights["recommendations"] = recommendations
        
        # Assess urgency
        urgency = self._assess_urgency_level(indicators)
        insights["urgency_level"] = urgency
        
        # Store patterns for future analysis
        self._update_humanitarian_patterns(region, indicators)
        
        return insights
    
    def _extract_humanitarian_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key humanitarian indicators from data."""
        indicators = {}
        
        # Population displacement indicators
        if "displacement_data" in data:
            displacement = data["displacement_data"]
            indicators["displaced_population"] = displacement.get("total_displaced", 0)
            indicators["displacement_trend"] = displacement.get("trend", "stable")
        
        # Food security indicators
        if "food_security" in data:
            food_data = data["food_security"]
            indicators["food_insecurity_rate"] = food_data.get("insecurity_rate", 0.0)
            indicators["malnutrition_rate"] = food_data.get("malnutrition_rate", 0.0)
        
        # Health indicators
        if "health_data" in data:
            health = data["health_data"]
            indicators["disease_outbreak_risk"] = health.get("outbreak_risk", "low")
            indicators["healthcare_access"] = health.get("access_rate", 1.0)
        
        # Infrastructure indicators
        if "infrastructure" in data:
            infra = data["infrastructure"]
            indicators["water_access"] = infra.get("water_access_rate", 1.0)
            indicators["shelter_adequacy"] = infra.get("shelter_adequacy", 1.0)
        
        return indicators
    
    def _generate_humanitarian_recommendations(self, indicators: Dict[str, Any],
                                             region: str) -> List[Dict[str, Any]]:
        """Generate humanitarian recommendations based on indicators."""
        recommendations = []
        
        # Displacement recommendations
        if indicators.get("displaced_population", 0) > 10000:
            recommendations.append({
                "category": "displacement",
                "priority": "high",
                "action": "Scale up temporary shelter capacity",
                "rationale": f"Large displaced population: {indicators['displaced_population']}"
            })
        
        # Food security recommendations
        food_insecurity = indicators.get("food_insecurity_rate", 0.0)
        if food_insecurity > 0.3:  # 30% food insecurity
            recommendations.append({
                "category": "food_security",
                "priority": "high" if food_insecurity > 0.5 else "medium",
                "action": "Increase food assistance programs",
                "rationale": f"High food insecurity rate: {food_insecurity:.1%}"
            })
        
        # Health recommendations
        if indicators.get("disease_outbreak_risk", "low") in ["high", "critical"]:
            recommendations.append({
                "category": "health",
                "priority": "critical",
                "action": "Implement disease prevention measures",
                "rationale": "High disease outbreak risk detected"
            })
        
        # Infrastructure recommendations
        water_access = indicators.get("water_access", 1.0)
        if water_access < 0.8:  # Less than 80% water access
            recommendations.append({
                "category": "infrastructure",
                "priority": "high",
                "action": "Improve water access infrastructure",
                "rationale": f"Limited water access: {water_access:.1%}"
            })
        
        return recommendations
    
    def _assess_urgency_level(self, indicators: Dict[str, Any]) -> str:
        """Assess overall urgency level based on indicators."""
        urgency_score = 0
        
        # Critical indicators
        if indicators.get("disease_outbreak_risk") == "critical":
            urgency_score += 4
        elif indicators.get("disease_outbreak_risk") == "high":
            urgency_score += 3
        
        if indicators.get("food_insecurity_rate", 0) > 0.5:
            urgency_score += 3
        elif indicators.get("food_insecurity_rate", 0) > 0.3:
            urgency_score += 2
        
        if indicators.get("displaced_population", 0) > 50000:
            urgency_score += 3
        elif indicators.get("displaced_population", 0) > 10000:
            urgency_score += 2
        
        if indicators.get("water_access", 1.0) < 0.5:
            urgency_score += 2
        
        # Determine urgency level
        if urgency_score >= 6:
            return "critical"
        elif urgency_score >= 4:
            return "high"
        elif urgency_score >= 2:
            return "medium"
        else:
            return "normal"
    
    def _update_humanitarian_patterns(self, region: str, indicators: Dict[str, Any]):
        """Update humanitarian patterns for trend analysis."""
        if region not in self.humanitarian_patterns:
            self.humanitarian_patterns[region] = deque(maxlen=100)
        
        pattern_entry = {
            "timestamp": datetime.now(),
            "indicators": indicators.copy()
        }
        
        self.humanitarian_patterns[region].append(pattern_entry)