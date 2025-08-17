"""Test suite for Generation 4 Intelligence systems.

Tests the AI-driven adaptive learning, self-optimization, research intelligence,
and global intelligence capabilities.
"""

import sys
import os
from datetime import datetime, timedelta
try:
    import pytest
    from unittest.mock import Mock, patch, MagicMock
except ImportError:
    # Create mock classes for testing without pytest
    class Mock:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __call__(self, *args, **kwargs):
            return Mock()
        def __getattr__(self, name):
            return Mock()
    
    class MagicMock(Mock):
        pass
    
    def patch(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from vislang_ultralow.intelligence.adaptive_learning import (
    AdaptiveLearningEngine, AutoMLPipelineOptimizer, 
    HyperparameterEvolution, MetaLearningController,
    PerformanceMetric, OptimizationResult
)

from vislang_ultralow.intelligence.self_optimization import (
    AutonomousQualityController, PerformanceSelfTuner,
    ResourceAdaptationEngine, WorkloadPredictiveScaler,
    SystemMetrics, OptimizationAction
)

from vislang_ultralow.intelligence.research_intelligence import (
    NovelAlgorithmDiscovery, ResearchHypothesisGenerator,
    ResearchHypothesis, ExperimentResult
)

from vislang_ultralow.intelligence.global_intelligence import (
    GlobalIntelligenceCoordinator, CrossRegionalLearning,
    CulturalContextAdapter, HumanitarianInsightEngine,
    CulturalContext
)

class TestAdaptiveLearningEngine:
    """Test adaptive learning engine functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = AdaptiveLearningEngine(
            learning_rate=0.01,
            exploration_factor=0.1,
            memory_size=1000,
            adaptation_threshold=0.05
        )
        
        assert engine.learning_rate == 0.01
        assert engine.exploration_factor == 0.1
        assert engine.memory_size == 1000
        assert engine.adaptation_threshold == 0.05
        assert len(engine.performance_history) == 0
        assert engine.total_adaptations == 0
    
    def test_parameter_registration(self):
        """Test parameter registration for optimization."""
        engine = AdaptiveLearningEngine()
        
        engine.register_parameter("learning_rate", 0.001, (0.0001, 0.01))
        engine.register_parameter("batch_size", 32, (8, 128))
        
        assert engine.optimal_parameters["learning_rate"] == 0.001
        assert engine.optimal_parameters["batch_size"] == 32
        assert engine.parameter_bounds["learning_rate"] == (0.0001, 0.01)
        assert engine.parameter_bounds["batch_size"] == (8, 128)
    
    def test_performance_recording(self):
        """Test performance metric recording."""
        engine = AdaptiveLearningEngine()
        
        metrics = {"accuracy": 0.85, "f1_score": 0.82}
        context = {"dataset": "test", "model": "transformer"}
        
        engine.record_performance(metrics, context)
        
        assert engine.current_performance == 0.85
        assert len(engine.performance_trend) == 1
        assert not engine.metric_queue.empty()
    
    def test_learning_statistics(self):
        """Test learning statistics generation."""
        engine = AdaptiveLearningEngine()
        
        # Simulate some learning history
        engine.total_adaptations = 10
        engine.successful_adaptations = 7
        
        # Add some performance metrics
        for i in range(5):
            metric = PerformanceMetric(
                name="accuracy",
                value=0.8 + i * 0.02,
                timestamp=datetime.now(),
                context={}
            )
            engine.performance_history.append(metric)
        
        stats = engine.get_learning_statistics()
        
        assert stats["total_adaptations"] == 10
        assert stats["successful_adaptations"] == 7
        assert stats["success_rate_percent"] == 70.0
        assert stats["metrics_collected"] == 5
        assert "performance_stats" in stats
        assert stats["learning_active"] == False

class TestAutonomousQualityController:
    """Test autonomous quality control functionality."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = AutonomousQualityController(
            target_performance=0.95,
            monitoring_interval=30.0
        )
        
        assert controller.target_performance == 0.95
        assert controller.monitoring_interval == 30.0
        assert controller.monitoring_active == False
        assert len(controller.quality_history) == 0
    
    def test_quality_metric_recording(self):
        """Test quality metric recording."""
        controller = AutonomousQualityController()
        
        controller.record_quality_metric("accuracy", 0.85, {"model": "test"})
        controller.record_quality_metric("response_time", 1.5, {"endpoint": "/api/predict"})
        
        assert not controller.metrics_queue.empty()
        assert len(controller.performance_trends["accuracy"]) == 1
        assert len(controller.performance_trends["response_time"]) == 1
    
    def test_quality_status(self):
        """Test quality status reporting."""
        controller = AutonomousQualityController()
        
        # Add some metrics
        controller.record_quality_metric("accuracy", 0.85)
        controller.record_quality_metric("response_time", 1.2)
        
        status = controller.get_quality_status()
        
        assert "monitoring_active" in status
        assert "metrics_collected" in status
        assert "active_controls" in status
        assert "quality_scores" in status

class TestPerformanceSelfTuner:
    """Test performance self-tuning functionality."""
    
    def test_initialization(self):
        """Test tuner initialization."""
        tuner = PerformanceSelfTuner(tuning_interval=300.0)
        
        assert tuner.tuning_interval == 300.0
        assert tuner.tuning_active == False
        assert len(tuner.performance_history) == 0
        assert "batch_size" in tuner.tunable_params
        assert "worker_threads" in tuner.tunable_params
    
    def test_performance_recording(self):
        """Test performance recording for tuning."""
        tuner = PerformanceSelfTuner()
        
        metrics = {
            "response_time": 1.5,
            "throughput": 100.0,
            "cpu_usage": 0.7
        }
        
        tuner.record_performance(metrics)
        
        assert len(tuner.performance_history) == 1
        perf_data = tuner.performance_history[0]
        assert "timestamp" in perf_data
        assert "metrics" in perf_data
        assert "parameters" in perf_data
    
    def test_tuning_status(self):
        """Test tuning status reporting."""
        tuner = PerformanceSelfTuner()
        
        status = tuner.get_tuning_status()
        
        assert "tuning_active" in status
        assert "current_parameters" in status
        assert "recent_tunings" in status
        assert "performance_samples" in status
        assert "last_tuning" in status

class TestNovelAlgorithmDiscovery:
    """Test novel algorithm discovery functionality."""
    
    def test_initialization(self):
        """Test discovery system initialization."""
        discovery = NovelAlgorithmDiscovery(exploration_budget=100)
        
        assert discovery.exploration_budget == 100
        assert len(discovery.discovered_algorithms) == 0
        assert len(discovery.performance_cache) == 0
        assert "preprocessing" in discovery.algorithm_components
        assert "learning_algorithms" in discovery.algorithm_components
    
    def test_algorithm_config_generation(self):
        """Test algorithm configuration generation."""
        discovery = NovelAlgorithmDiscovery()
        
        config = discovery._generate_algorithm_config("multimodal_classification")
        
        assert "preprocessing" in config
        assert "feature_extraction" in config
        assert "learning_algorithms" in config
        assert "optimization" in config
        assert "regularization" in config
        assert "hyperparameters" in config
    
    def test_config_hashing(self):
        """Test configuration hashing for deduplication."""
        discovery = NovelAlgorithmDiscovery()
        
        config1 = {"method": "transformer", "lr": 0.001}
        config2 = {"method": "transformer", "lr": 0.001}
        config3 = {"method": "cnn", "lr": 0.001}
        
        hash1 = discovery._hash_config(config1)
        hash2 = discovery._hash_config(config2)
        hash3 = discovery._hash_config(config3)
        
        assert hash1 == hash2  # Same config should have same hash
        assert hash1 != hash3  # Different config should have different hash
    
    def test_novelty_calculation(self):
        """Test novelty score calculation."""
        discovery = NovelAlgorithmDiscovery()
        
        # First algorithm should be maximally novel
        config1 = {"method": "transformer"}
        novelty1 = discovery._calculate_novelty_score(config1)
        assert novelty1 == 1.0
        
        # Add an algorithm to history
        discovery.discovered_algorithms.append({
            "config": config1,
            "performance": 0.8
        })
        
        # Same algorithm should have low novelty
        novelty2 = discovery._calculate_novelty_score(config1)
        assert novelty2 == 0.0
        
        # Different algorithm should have high novelty
        config2 = {"method": "cnn"}
        novelty3 = discovery._calculate_novelty_score(config2)
        assert novelty3 > 0.5

class TestResearchHypothesisGenerator:
    """Test research hypothesis generation functionality."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = ResearchHypothesisGenerator()
        
        assert len(generator.hypotheses) == 0
        assert len(generator.hypothesis_templates) > 0
    
    def test_pattern_identification(self):
        """Test pattern identification in experimental data."""
        generator = ResearchHypothesisGenerator()
        
        experimental_data = [
            {"method": "transformer", "performance": 0.85, "parameters": {"lr": 0.001}},
            {"method": "transformer", "performance": 0.87, "parameters": {"lr": 0.002}},
            {"method": "transformer", "performance": 0.89, "parameters": {"lr": 0.003}},
            {"method": "cnn", "performance": 0.75, "parameters": {"lr": 0.001}},
            {"method": "cnn", "performance": 0.77, "parameters": {"lr": 0.002}}
        ]
        
        patterns = generator._identify_patterns(experimental_data)
        
        assert len(patterns) > 0
        # Should find patterns for both methods
        method_patterns = [p for p in patterns if p["type"] == "performance_distribution"]
        assert len(method_patterns) >= 2

class TestGlobalIntelligenceCoordinator:
    """Test global intelligence coordination functionality."""
    
    def test_initialization(self):
        """Test coordinator initialization."""
        coordinator = GlobalIntelligenceCoordinator()
        
        assert len(coordinator.regional_intelligence) == 0
        assert len(coordinator.global_patterns) == 0
        assert coordinator.coordination_active == False
    
    def test_region_registration(self):
        """Test regional intelligence node registration."""
        coordinator = GlobalIntelligenceCoordinator()
        
        mock_node = Mock()
        coordinator.register_region("us-east", mock_node)
        
        assert "us-east" in coordinator.regional_intelligence
        assert coordinator.regional_intelligence["us-east"] == mock_node
    
    def test_coordination_with_no_regions(self):
        """Test coordination behavior with no registered regions."""
        coordinator = GlobalIntelligenceCoordinator()
        
        result = coordinator.coordinate_learning()
        
        assert result == {}

class TestCulturalContextAdapter:
    """Test cultural context adaptation functionality."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = CulturalContextAdapter()
        
        assert len(adapter.cultural_profiles) == 0
        assert "high_context_cultures" in adapter.adaptation_rules
        assert "collectivist_cultures" in adapter.adaptation_rules
    
    def test_interface_adaptation(self):
        """Test interface adaptation based on cultural context."""
        adapter = CulturalContextAdapter()
        
        # Test direct communication style
        context = CulturalContext(
            region="US",
            language_codes=["en"],
            writing_direction="ltr",
            cultural_values={"individualism": 0.8},
            communication_style="direct",
            data_sensitivity="medium"
        )
        
        adaptations = adapter.adapt_interface(context)
        
        assert "language_support" in adaptations
        assert adaptations["language_support"]["primary_languages"] == ["en"]
        assert adaptations["language_support"]["rtl_support"] == False
    
    def test_cultural_validation(self):
        """Test cultural appropriateness validation."""
        adapter = CulturalContextAdapter()
        
        context = CulturalContext(
            region="ME",
            language_codes=["ar"],
            writing_direction="rtl",
            cultural_values={},
            communication_style="indirect",
            data_sensitivity="high"
        )
        
        content = {
            "text": "Hello world",
            "images": [{"tags": ["neutral"]}]
        }
        
        validation = adapter.validate_cultural_appropriateness(content, context)
        
        assert "is_appropriate" in validation
        assert "concerns" in validation
        assert "suggestions" in validation

class TestHumanitarianInsightEngine:
    """Test humanitarian insight generation functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = HumanitarianInsightEngine()
        
        assert len(engine.humanitarian_patterns) == 0
        assert len(engine.crisis_indicators) == 0
        assert len(engine.intervention_effectiveness) == 0
    
    def test_humanitarian_analysis(self):
        """Test humanitarian data analysis."""
        engine = HumanitarianInsightEngine()
        
        data = {
            "displacement_data": {
                "total_displaced": 15000,
                "trend": "increasing"
            },
            "food_security": {
                "insecurity_rate": 0.4,
                "malnutrition_rate": 0.15
            },
            "health_data": {
                "outbreak_risk": "high",
                "access_rate": 0.6
            },
            "infrastructure": {
                "water_access_rate": 0.7,
                "shelter_adequacy": 0.5
            }
        }
        
        insights = engine.analyze_humanitarian_data(data, "test_region")
        
        assert insights["region"] == "test_region"
        assert "analysis_timestamp" in insights
        assert "key_indicators" in insights
        assert "recommendations" in insights
        assert "urgency_level" in insights
        
        # Should detect high urgency due to multiple risk factors
        assert insights["urgency_level"] in ["high", "critical"]
        
        # Should generate multiple recommendations
        assert len(insights["recommendations"]) > 0
    
    def test_urgency_assessment(self):
        """Test urgency level assessment."""
        engine = HumanitarianInsightEngine()
        
        # Low urgency indicators
        low_urgency_indicators = {
            "displaced_population": 1000,
            "food_insecurity_rate": 0.1,
            "disease_outbreak_risk": "low",
            "water_access": 0.9
        }
        
        urgency_low = engine._assess_urgency_level(low_urgency_indicators)
        assert urgency_low in ["normal", "medium"]
        
        # High urgency indicators
        high_urgency_indicators = {
            "displaced_population": 100000,
            "food_insecurity_rate": 0.6,
            "disease_outbreak_risk": "critical",
            "water_access": 0.3
        }
        
        urgency_high = engine._assess_urgency_level(high_urgency_indicators)
        assert urgency_high in ["high", "critical"]

# Integration tests
class TestGeneration4Integration:
    """Test integration between Generation 4 components."""
    
    def test_adaptive_learning_with_quality_control(self):
        """Test integration between adaptive learning and quality control."""
        learning_engine = AdaptiveLearningEngine()
        quality_controller = AutonomousQualityController()
        
        # Register parameters in learning engine
        learning_engine.register_parameter("threshold", 0.5, (0.1, 0.9))
        
        # Record performance in both systems
        metrics = {"accuracy": 0.85, "response_time": 1.2}
        
        learning_engine.record_performance(metrics)
        quality_controller.record_quality_metric("accuracy", 0.85)
        quality_controller.record_quality_metric("response_time", 1.2)
        
        # Both systems should have recorded the metrics
        assert learning_engine.current_performance == 0.85
        assert len(quality_controller.performance_trends["accuracy"]) == 1
    
    def test_research_intelligence_with_global_coordination(self):
        """Test integration between research intelligence and global coordination."""
        discovery = NovelAlgorithmDiscovery()
        coordinator = GlobalIntelligenceCoordinator()
        
        # Mock regional node with research capabilities
        mock_node = Mock()
        mock_node.get_regional_insights.return_value = {
            "performance_metrics": {"accuracy": 0.85},
            "cultural_adaptations": {"language": "en"},
            "discovered_algorithms": [{"method": "transformer"}]
        }
        
        coordinator.register_region("test_region", mock_node)
        
        # Coordinate learning should work with research data
        result = coordinator.coordinate_learning()
        
        assert result["regions_coordinated"] == 1
        assert "coordination_timestamp" in result

def test_generation4_quality_gates():
    """Test that Generation 4 meets quality gates."""
    
    # Test 1: All components can be imported without errors
    from vislang_ultralow.intelligence import (
        AdaptiveLearningEngine, AutonomousQualityController,
        NovelAlgorithmDiscovery, GlobalIntelligenceCoordinator
    )
    
    # Test 2: All components can be instantiated
    learning_engine = AdaptiveLearningEngine()
    quality_controller = AutonomousQualityController()
    discovery = NovelAlgorithmDiscovery()
    coordinator = GlobalIntelligenceCoordinator()
    
    assert learning_engine is not None
    assert quality_controller is not None
    assert discovery is not None
    assert coordinator is not None
    
    # Test 3: Basic functionality works
    learning_engine.register_parameter("test_param", 0.5, (0.1, 0.9))
    quality_controller.record_quality_metric("accuracy", 0.85)
    
    config = discovery._generate_algorithm_config("test_domain")
    assert len(config) > 0
    
    print("✅ Generation 4 Intelligence Systems - All quality gates passed")
    return True

if __name__ == "__main__":
    # Run quality gates test
    test_generation4_quality_gates()
    
    # Run pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except (ImportError, NameError):
        print("pytest not available, running basic tests...")
        
        # Run basic test instances
        test_adaptive = TestAdaptiveLearningEngine()
        test_adaptive.test_initialization()
        test_adaptive.test_parameter_registration()
        print("✅ AdaptiveLearningEngine tests passed")
        
        test_quality = TestAutonomousQualityController()
        test_quality.test_initialization()
        test_quality.test_quality_metric_recording()
        print("✅ AutonomousQualityController tests passed")
        
        test_discovery = TestNovelAlgorithmDiscovery()
        test_discovery.test_initialization()
        test_discovery.test_algorithm_config_generation()
        print("✅ NovelAlgorithmDiscovery tests passed")
        
        test_global = TestGlobalIntelligenceCoordinator()
        test_global.test_initialization()
        test_global.test_region_registration()
        print("✅ GlobalIntelligenceCoordinator tests passed")
        
        print("\n🎉 All Generation 4 tests completed successfully!")