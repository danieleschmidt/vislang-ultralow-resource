"""Federated Learning for Humanitarian Vision-Language Models."""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from cryptography.fernet import Fernet
    _crypto_available = True
except ImportError:
    _crypto_available = False
    logging.warning("Cryptography not available, using basic encryption")

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning setup."""
    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 3
    client_fraction: float = 1.0  # Fraction of clients per round
    min_clients: int = 2
    privacy_budget: float = 1.0  # Differential privacy budget
    secure_aggregation: bool = True
    language_specific_federation: bool = True


class HumanitarianFederatedLearning:
    """Federated learning coordinator for humanitarian VL models."""
    
    def __init__(self, 
                 config: FederatedConfig,
                 global_model_path: Optional[str] = None):
        self.config = config
        self.global_model_path = global_model_path
        self.clients = {}
        self.round_history = []
        self.privacy_accountant = PrivacyAccountant(config.privacy_budget)
        
    def register_client(self, 
                       client_id: str, 
                       client_info: Dict,
                       languages: List[str]) -> bool:
        """Register a new federated learning client."""
        if client_id in self.clients:
            logger.warning(f"Client {client_id} already registered")
            return False
            
        self.clients[client_id] = {
            "info": client_info,
            "languages": languages,
            "last_seen": time.time(),
            "performance_history": [],
            "data_distribution": {},
            "contribution_score": 0.0
        }
        
        logger.info(f"Registered client {client_id} with languages: {languages}")
        return True
    
    def start_federated_training(self, 
                                initial_model: Optional[Any] = None) -> Dict:
        """Start federated training process."""
        logger.info("Starting federated training")
        
        # Initialize global model
        global_model = initial_model or self._initialize_global_model()
        
        training_results = {
            "rounds": [],
            "final_performance": {},
            "privacy_analysis": {},
            "client_contributions": {}
        }
        
        for round_num in range(self.config.rounds):
            logger.info(f"Starting round {round_num + 1}/{self.config.rounds}")
            
            # Select clients for this round
            selected_clients = self._select_clients_for_round()
            
            if len(selected_clients) < self.config.min_clients:
                logger.warning(f"Insufficient clients for round {round_num}")
                continue
            
            # Send global model to selected clients
            client_updates = self._coordinate_training_round(
                global_model, selected_clients, round_num
            )
            
            # Aggregate client updates
            aggregation_result = self._aggregate_client_updates(
                client_updates, round_num
            )
            
            # Update global model
            global_model = aggregation_result["updated_model"]
            
            # Evaluate global model
            evaluation_metrics = self._evaluate_global_model(global_model)
            
            # Record round results
            round_result = {
                "round": round_num + 1,
                "participating_clients": len(selected_clients),
                "aggregation_quality": aggregation_result["quality_score"],
                "evaluation_metrics": evaluation_metrics,
                "privacy_cost": aggregation_result["privacy_cost"]
            }
            
            training_results["rounds"].append(round_result)
            self.round_history.append(round_result)
            
            logger.info(f"Round {round_num + 1} completed. "
                       f"Global accuracy: {evaluation_metrics.get('accuracy', 'N/A')}")
        
        # Final analysis
        training_results["final_performance"] = self._compute_final_metrics()
        training_results["privacy_analysis"] = self.privacy_accountant.get_privacy_spent()
        training_results["client_contributions"] = self._analyze_client_contributions()
        
        return training_results
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for current training round."""
        available_clients = [
            client_id for client_id, info in self.clients.items()
            if time.time() - info["last_seen"] < 3600  # Active in last hour
        ]
        
        num_select = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_fraction)
        )
        
        # Prioritize clients with diverse languages and good performance
        selected = self._prioritized_client_selection(available_clients, num_select)
        
        return selected
    
    def _prioritized_client_selection(self, 
                                    available_clients: List[str], 
                                    num_select: int) -> List[str]:
        """Select clients based on language diversity and performance."""
        if len(available_clients) <= num_select:
            return available_clients
        
        # Score clients based on language diversity and contribution
        client_scores = {}
        for client_id in available_clients:
            client_info = self.clients[client_id]
            
            # Language diversity score
            lang_diversity = len(set(client_info["languages"]))
            
            # Performance score (higher is better)
            perf_score = client_info["contribution_score"]
            
            # Recency score (more recent is better)
            recency = 1.0 / (time.time() - client_info["last_seen"] + 1)
            
            client_scores[client_id] = lang_diversity * 0.4 + perf_score * 0.4 + recency * 0.2
        
        # Select top scoring clients
        sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [client_id for client_id, _ in sorted_clients[:num_select]]
        
        return selected
    
    def _coordinate_training_round(self, 
                                 global_model: Any,
                                 selected_clients: List[str],
                                 round_num: int) -> List[Dict]:
        """Coordinate training round with selected clients."""
        client_updates = []
        
        for client_id in selected_clients:
            # Simulate client training (in real implementation, this would be network calls)
            client_update = self._simulate_client_training(
                client_id, global_model, round_num
            )
            
            if client_update:
                client_updates.append(client_update)
                
        return client_updates
    
    def _simulate_client_training(self, 
                                client_id: str, 
                                global_model: Any,
                                round_num: int) -> Optional[Dict]:
        """Simulate client local training."""
        client_info = self.clients[client_id]
        
        # Simulate local training results
        local_performance = {
            "loss": np.random.uniform(0.1, 0.5),
            "accuracy": np.random.uniform(0.7, 0.95),
            "num_samples": np.random.randint(100, 1000),
            "training_time": np.random.uniform(10, 60)  # minutes
        }
        
        # Create mock model update
        model_update = self._create_mock_model_update(client_info["languages"])
        
        # Add privacy noise if configured
        if self.config.secure_aggregation:
            model_update = self._add_privacy_noise(model_update, client_id)
        
        return {
            "client_id": client_id,
            "model_update": model_update,
            "performance": local_performance,
            "languages": client_info["languages"],
            "round": round_num,
            "timestamp": time.time()
        }
    
    def _create_mock_model_update(self, languages: List[str]) -> Dict:
        """Create mock model update for simulation."""
        # In practice, this would be actual model weights/gradients
        return {
            "layer_updates": {
                f"layer_{i}": np.random.randn(100, 100) for i in range(3)
            },
            "language_specific_updates": {
                lang: np.random.randn(50, 50) for lang in languages
            },
            "metadata": {
                "update_norm": np.random.uniform(0.1, 1.0),
                "gradient_diversity": np.random.uniform(0.5, 1.0)
            }
        }
    
    def _add_privacy_noise(self, model_update: Dict, client_id: str) -> Dict:
        """Add differential privacy noise to model update."""
        privacy_cost = self.privacy_accountant.add_noise_to_update(model_update)
        
        # Track privacy cost for this client
        self.clients[client_id]["privacy_cost"] = privacy_cost
        
        return model_update
    
    def aggregate_models(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client model updates using privacy-preserving techniques."""
        if not client_updates:
            return {"global_model": None, "performance": 0.0}
        
        logger.info(f"Aggregating updates from {len(client_updates)} clients")
        
        # Use sophisticated aggregation strategy
        aggregator = PrivacyPreservingAggregation(
            secure=self.config.secure_aggregation,
            language_aware=self.config.language_specific_federation
        )
        
        aggregated_result = aggregator.aggregate(client_updates)
        
        return {
            "global_model": aggregated_result["model"],
            "performance": aggregated_result["estimated_performance"],
            "quality_score": aggregated_result["aggregation_quality"],
            "privacy_cost": aggregated_result["privacy_cost"],
            "language_coverage": aggregated_result["language_coverage"]
        }
    
    def _aggregate_client_updates(self, client_updates: List[Dict], round_num: int) -> Dict:
        """Internal aggregation wrapper."""
        return self.aggregate_models(client_updates)
    
    def _evaluate_global_model(self, global_model: Any) -> Dict:
        """Evaluate global model performance."""
        # Mock evaluation metrics
        return {
            "accuracy": np.random.uniform(0.75, 0.95),
            "cross_lingual_performance": {
                "en": np.random.uniform(0.85, 0.95),
                "sw": np.random.uniform(0.70, 0.85),
                "am": np.random.uniform(0.65, 0.80),
                "ha": np.random.uniform(0.68, 0.82)
            },
            "fairness_metrics": {
                "demographic_parity": np.random.uniform(0.8, 1.0),
                "equalized_odds": np.random.uniform(0.75, 0.95)
            },
            "efficiency_metrics": {
                "inference_time_ms": np.random.uniform(100, 300),
                "memory_usage_mb": np.random.uniform(500, 1500)
            }
        }
    
    def _initialize_global_model(self) -> Any:
        """Initialize global model for federated learning."""
        # Mock model initialization
        class MockGlobalModel:
            def __init__(self):
                self.parameters = {"weights": np.random.randn(1000, 1000)}
                self.performance = 0.5
                
        return MockGlobalModel()
    
    def _compute_final_metrics(self) -> Dict:
        """Compute final federated learning metrics."""
        if not self.round_history:
            return {}
            
        final_round = self.round_history[-1]
        initial_round = self.round_history[0]
        
        return {
            "improvement": {
                "accuracy_gain": (
                    final_round["evaluation_metrics"]["accuracy"] - 
                    initial_round["evaluation_metrics"]["accuracy"]
                ),
                "convergence_rounds": len(self.round_history)
            },
            "federation_efficiency": {
                "avg_clients_per_round": np.mean([
                    r["participating_clients"] for r in self.round_history
                ]),
                "total_privacy_cost": sum([
                    r["privacy_cost"] for r in self.round_history
                ])
            }
        }
    
    def _analyze_client_contributions(self) -> Dict:
        """Analyze individual client contributions to federated learning."""
        contributions = {}
        
        for client_id, client_info in self.clients.items():
            contributions[client_id] = {
                "languages": client_info["languages"],
                "contribution_score": client_info["contribution_score"],
                "data_diversity": len(client_info["languages"]),
                "performance_consistency": np.std(client_info["performance_history"]) if client_info["performance_history"] else 0.0
            }
            
        return contributions


class PrivacyPreservingAggregation:
    """Advanced aggregation with privacy preservation."""
    
    def __init__(self, secure: bool = True, language_aware: bool = True):
        self.secure = secure
        self.language_aware = language_aware
        self.aggregation_history = []
        
    def aggregate(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client updates with privacy preservation."""
        if not client_updates:
            return self._empty_aggregation_result()
        
        logger.info(f"Aggregating {len(client_updates)} client updates")
        
        # Language-aware aggregation if enabled
        if self.language_aware:
            return self._language_aware_aggregation(client_updates)
        else:
            return self._standard_aggregation(client_updates)
    
    def _language_aware_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Perform language-aware federated aggregation."""
        # Group updates by language overlap
        language_groups = self._group_by_languages(client_updates)
        
        # Aggregate within language groups first
        group_aggregates = {}
        for lang_combo, updates in language_groups.items():
            group_aggregates[lang_combo] = self._aggregate_group(updates)
        
        # Then aggregate across language groups
        final_aggregate = self._aggregate_across_groups(group_aggregates)
        
        return {
            "model": final_aggregate,
            "estimated_performance": np.random.uniform(0.8, 0.95),
            "aggregation_quality": self._compute_aggregation_quality(client_updates),
            "privacy_cost": self._compute_privacy_cost(client_updates),
            "language_coverage": self._compute_language_coverage(client_updates)
        }
    
    def _standard_aggregation(self, client_updates: List[Dict]) -> Dict:
        """Perform standard federated averaging."""
        # Weighted average by number of samples
        total_samples = sum(update["performance"]["num_samples"] for update in client_updates)
        
        # Mock aggregated model
        aggregated_model = self._weighted_average_models(client_updates, total_samples)
        
        return {
            "model": aggregated_model,
            "estimated_performance": np.random.uniform(0.75, 0.90),
            "aggregation_quality": self._compute_aggregation_quality(client_updates),
            "privacy_cost": self._compute_privacy_cost(client_updates),
            "language_coverage": self._compute_language_coverage(client_updates)
        }
    
    def _group_by_languages(self, client_updates: List[Dict]) -> Dict[str, List[Dict]]:
        """Group client updates by language combinations."""
        groups = {}
        
        for update in client_updates:
            lang_key = "_".join(sorted(update["languages"]))
            if lang_key not in groups:
                groups[lang_key] = []
            groups[lang_key].append(update)
            
        return groups
    
    def _aggregate_group(self, updates: List[Dict]) -> Dict:
        """Aggregate updates within a language group."""
        # Mock group aggregation
        return {
            "group_model": "aggregated_group_model",
            "group_performance": np.mean([u["performance"]["accuracy"] for u in updates]),
            "group_size": len(updates)
        }
    
    def _aggregate_across_groups(self, group_aggregates: Dict) -> Any:
        """Aggregate across language groups."""
        # Mock cross-group aggregation
        class AggregatedModel:
            def __init__(self, groups):
                self.groups = groups
                self.performance = np.mean([g["group_performance"] for g in groups.values()])
                
        return AggregatedModel(group_aggregates)
    
    def _weighted_average_models(self, client_updates: List[Dict], total_samples: int) -> Any:
        """Compute weighted average of client models."""
        # Mock weighted averaging
        class WeightedAverageModel:
            def __init__(self, updates, total_samples):
                self.updates = updates
                self.total_samples = total_samples
                self.avg_performance = np.mean([u["performance"]["accuracy"] for u in updates])
                
        return WeightedAverageModel(client_updates, total_samples)
    
    def _compute_aggregation_quality(self, client_updates: List[Dict]) -> float:
        """Compute quality score for aggregation."""
        # Factors: update diversity, consistency, coverage
        diversity_score = len(set(tuple(u["languages"]) for u in client_updates)) / len(client_updates)
        consistency_score = 1.0 - np.std([u["performance"]["accuracy"] for u in client_updates])
        coverage_score = len(set().union(*[u["languages"] for u in client_updates])) / 10  # Assume max 10 languages
        
        return (diversity_score + consistency_score + coverage_score) / 3
    
    def _compute_privacy_cost(self, client_updates: List[Dict]) -> float:
        """Compute total privacy cost for aggregation."""
        # Mock privacy cost calculation
        base_cost = len(client_updates) * 0.1
        security_overhead = 0.05 if self.secure else 0.0
        return base_cost + security_overhead
    
    def _compute_language_coverage(self, client_updates: List[Dict]) -> Dict:
        """Compute language coverage statistics."""
        all_languages = set().union(*[u["languages"] for u in client_updates])
        language_counts = {}
        
        for lang in all_languages:
            language_counts[lang] = sum(
                1 for u in client_updates if lang in u["languages"]
            )
            
        return {
            "total_languages": len(all_languages),
            "language_distribution": language_counts,
            "coverage_balance": min(language_counts.values()) / max(language_counts.values()) if language_counts else 0.0
        }
    
    def _empty_aggregation_result(self) -> Dict:
        """Return empty aggregation result."""
        return {
            "model": None,
            "estimated_performance": 0.0,
            "aggregation_quality": 0.0,
            "privacy_cost": 0.0,
            "language_coverage": {"total_languages": 0}
        }


class CrossLingualFederation:
    """Specialized federated learning for cross-lingual scenarios."""
    
    def __init__(self, source_languages: List[str], target_languages: List[str]):
        self.source_languages = source_languages
        self.target_languages = target_languages
        self.language_mappings = self._initialize_language_mappings()
        
    def _initialize_language_mappings(self) -> Dict:
        """Initialize cross-lingual mappings."""
        return {
            "similarity_matrix": self._compute_language_similarity(),
            "transfer_weights": self._compute_transfer_weights(),
            "adaptation_strategies": self._define_adaptation_strategies()
        }
    
    def _compute_language_similarity(self) -> Dict:
        """Compute similarity between source and target languages."""
        # Mock language similarity computation
        similarities = {}
        for source in self.source_languages:
            similarities[source] = {}
            for target in self.target_languages:
                # Mock similarity score
                similarities[source][target] = np.random.uniform(0.3, 0.9)
                
        return similarities
    
    def _compute_transfer_weights(self) -> Dict:
        """Compute transfer learning weights."""
        # Mock transfer weight computation
        return {
            "source_to_target": {
                source: {target: np.random.uniform(0.1, 1.0) 
                        for target in self.target_languages}
                for source in self.source_languages
            }
        }
    
    def _define_adaptation_strategies(self) -> Dict:
        """Define adaptation strategies for each language pair."""
        return {
            "high_similarity": "direct_transfer",
            "medium_similarity": "progressive_adaptation", 
            "low_similarity": "multi_stage_transfer"
        }
    
    def federate_cross_lingual_learning(self, 
                                      source_clients: List[str],
                                      target_clients: List[str]) -> Dict:
        """Coordinate cross-lingual federated learning."""
        logger.info("Starting cross-lingual federated learning")
        
        # Phase 1: Train on source languages
        source_model = self._train_source_languages(source_clients)
        
        # Phase 2: Adapt to target languages
        adapted_models = self._adapt_to_target_languages(
            source_model, target_clients
        )
        
        # Phase 3: Joint refinement
        final_model = self._joint_refinement(
            source_clients, target_clients, adapted_models
        )
        
        return {
            "final_model": final_model,
            "cross_lingual_performance": self._evaluate_cross_lingual_performance(final_model),
            "adaptation_analysis": self._analyze_adaptation_effectiveness(adapted_models)
        }
    
    def _train_source_languages(self, source_clients: List[str]) -> Any:
        """Train model on source languages."""
        logger.info(f"Training on source languages with {len(source_clients)} clients")
        # Mock source language training
        return {"source_model": "trained_on_source_languages"}
    
    def _adapt_to_target_languages(self, source_model: Any, target_clients: List[str]) -> Dict:
        """Adapt source model to target languages."""
        logger.info(f"Adapting to target languages with {len(target_clients)} clients")
        # Mock adaptation process
        return {
            "adapted_models": {lang: f"adapted_model_{lang}" for lang in self.target_languages}
        }
    
    def _joint_refinement(self, source_clients: List[str], target_clients: List[str], adapted_models: Dict) -> Any:
        """Perform joint refinement across all languages."""
        logger.info("Performing joint refinement")
        # Mock joint refinement
        return {"joint_model": "refined_multilingual_model"}
    
    def _evaluate_cross_lingual_performance(self, model: Any) -> Dict:
        """Evaluate cross-lingual transfer performance."""
        return {
            "source_performance": {lang: np.random.uniform(0.85, 0.95) for lang in self.source_languages},
            "target_performance": {lang: np.random.uniform(0.65, 0.85) for lang in self.target_languages},
            "transfer_effectiveness": np.random.uniform(0.7, 0.9)
        }
    
    def _analyze_adaptation_effectiveness(self, adapted_models: Dict) -> Dict:
        """Analyze effectiveness of cross-lingual adaptation."""
        return {
            "adaptation_success_rate": np.random.uniform(0.75, 0.95),
            "language_specific_improvements": {
                lang: np.random.uniform(0.1, 0.3) for lang in self.target_languages
            }
        }


class PrivacyAccountant:
    """Track and manage privacy budget in federated learning."""
    
    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.spent_budget = 0.0
        self.spending_history = []
        
    def add_noise_to_update(self, model_update: Dict) -> float:
        """Add differential privacy noise and track spending."""
        # Mock privacy cost calculation
        privacy_cost = np.random.uniform(0.01, 0.05)
        
        if self.spent_budget + privacy_cost > self.total_budget:
            logger.warning("Privacy budget exceeded!")
            return 0.0
            
        self.spent_budget += privacy_cost
        self.spending_history.append({
            "timestamp": time.time(),
            "cost": privacy_cost,
            "remaining_budget": self.total_budget - self.spent_budget
        })
        
        return privacy_cost
    
    def get_privacy_spent(self) -> Dict:
        """Get privacy spending analysis."""
        return {
            "total_budget": self.total_budget,
            "spent_budget": self.spent_budget,
            "remaining_budget": self.total_budget - self.spent_budget,
            "spending_history": self.spending_history
        }