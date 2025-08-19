#!/usr/bin/env python3
"""
Enhanced Generation 1 Test Suite - Autonomous SDLC Execution

Tests the enhanced Generation 1 functionality including:
- Intelligent dataset synthesis
- Integrated pipeline orchestration  
- Research-grade cross-lingual alignment
- Novel algorithm discovery
- Adaptive OCR with consensus
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vislang_ultralow.intelligence import (
    SyntheticDatasetGenerator,
    IntelligentHumanitarianPipeline,
    PerformanceTracker,
    NovelAlgorithmDiscovery,
    ResearchHypothesisGenerator
)

from vislang_ultralow.research import (
    AdaptiveMultiEngineOCR,
    ZeroShotCrossLingual
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedGeneration1Test:
    """Test suite for enhanced Generation 1 functionality."""
    
    def __init__(self):
        self.target_languages = ['sw', 'am', 'ha', 'en']
        self.test_results = {}
        
    def run_all_tests(self) -> dict:
        """Run complete test suite."""
        logger.info("🚀 Starting Enhanced Generation 1 Test Suite")
        
        tests = [
            ("Synthetic Dataset Generation", self.test_synthetic_dataset_generation),
            ("Intelligent OCR Pipeline", self.test_intelligent_ocr_pipeline),
            ("Cross-Lingual Alignment", self.test_cross_lingual_alignment),
            ("Novel Algorithm Discovery", self.test_novel_algorithm_discovery),
            ("Research Hypothesis Generation", self.test_research_hypothesis_generation),
            ("Performance Tracking", self.test_performance_tracking),
            ("Integrated Pipeline", self.test_integrated_pipeline)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = {
                    "status": "PASSED" if result.get("success", False) else "FAILED",
                    "details": result
                }
                logger.info(f"✅ {test_name}: {self.test_results[test_name]['status']}")
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                logger.error(f"❌ {test_name}: ERROR - {e}")
        
        # Generate final report
        self.generate_test_report()
        return self.test_results
    
    def test_synthetic_dataset_generation(self) -> dict:
        """Test synthetic dataset generation capabilities."""
        logger.info("Testing synthetic dataset generation...")
        
        try:
            # Initialize generator
            generator = SyntheticDatasetGenerator(
                target_languages=self.target_languages
            )
            
            # Test different synthesis strategies
            strategies = ['template_variation', 'cross_lingual_adaptation', 'mixed']
            results = {}
            
            for strategy in strategies:
                logger.info(f"Testing strategy: {strategy}")
                
                dataset = generator.generate_synthetic_dataset(
                    num_samples=50,
                    strategy=strategy
                )
                
                # Validate dataset
                assert 'samples' in dataset
                assert 'metadata' in dataset
                assert len(dataset['samples']) > 0
                
                # Check sample quality
                samples = dataset['samples']
                quality_scores = [s.get('quality_score', 0) for s in samples]
                avg_quality = sum(quality_scores) / len(quality_scores)
                
                results[strategy] = {
                    'samples_generated': len(samples),
                    'average_quality': avg_quality,
                    'languages_covered': len(set(s.get('language') for s in samples)),
                    'metadata': dataset['metadata']
                }
                
                logger.info(f"Strategy {strategy}: {len(samples)} samples, avg quality: {avg_quality:.3f}")
            
            # Test generation statistics
            stats = generator.get_generation_statistics()
            assert 'total_generated_samples' in stats
            assert stats['total_generated_samples'] > 0
            
            return {
                "success": True,
                "strategy_results": results,
                "generation_statistics": stats,
                "total_samples_generated": sum(r['samples_generated'] for r in results.values())
            }
            
        except Exception as e:
            logger.error(f"Synthetic dataset generation test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_intelligent_ocr_pipeline(self) -> dict:
        """Test intelligent OCR pipeline with adaptive capabilities."""
        logger.info("Testing intelligent OCR pipeline...")
        
        try:
            # Initialize adaptive OCR
            ocr_system = AdaptiveMultiEngineOCR(
                engines=['tesseract', 'easyocr']
            )
            
            # Test different document types
            document_types = ['standard', 'humanitarian_report', 'infographic', 'chart']
            results = {}
            
            for doc_type in document_types:
                logger.info(f"Testing OCR for document type: {doc_type}")
                
                # Create mock image (simple test)
                mock_image = self._create_mock_image()
                
                # Extract text
                extraction_result = ocr_system.extract_text(mock_image, doc_type)
                
                # Validate result
                assert 'text' in extraction_result
                assert 'confidence' in extraction_result
                assert extraction_result['confidence'] > 0
                
                results[doc_type] = {
                    'text_extracted': len(extraction_result['text']),
                    'confidence': extraction_result['confidence'],
                    'consensus_method': extraction_result.get('consensus_method', 'unknown'),
                    'engines_used': len(extraction_result.get('individual_results', []))
                }
                
                logger.info(f"OCR {doc_type}: confidence {extraction_result['confidence']:.3f}")
            
            # Test performance statistics
            perf_stats = ocr_system.get_engine_performance_stats()
            
            return {
                "success": True,
                "document_type_results": results,
                "performance_statistics": perf_stats,
                "total_extractions": len(document_types)
            }
            
        except Exception as e:
            logger.error(f"OCR pipeline test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_cross_lingual_alignment(self) -> dict:
        """Test cross-lingual alignment capabilities."""
        logger.info("Testing cross-lingual alignment...")
        
        try:
            # Initialize alignment system
            aligner = ZeroShotCrossLingual()
            
            # Test alignment learning
            source_texts = [
                "Emergency food distribution in progress",
                "Medical supplies delivered successfully", 
                "Shelter construction completed"
            ]
            
            target_texts = [
                "Usambazaji wa chakula wa dharura unaendelea",
                "Vifaa vya kitibu vimefikishwa kwa ufanisi",
                "Ujenzi wa makazi umekamilika"
            ]
            
            # Learn alignment
            alignment_result = aligner.learn_alignment(
                source_texts=source_texts,
                target_texts=target_texts,
                source_lang='en',
                target_lang='sw'
            )
            
            # Validate alignment
            assert 'alignment_matrix' in alignment_result
            assert 'quality_metrics' in alignment_result
            assert alignment_result['num_samples'] == len(source_texts)
            
            # Test cross-lingual similarity
            similarity_tests = [
                ("emergency food", "chakula cha dharura", 'en', 'sw'),
                ("medical supplies", "vifaa vya kitibu", 'en', 'sw'),
                ("shelter construction", "ujenzi wa makazi", 'en', 'sw')
            ]
            
            similarity_results = []
            for text1, text2, lang1, lang2 in similarity_tests:
                similarity = aligner.compute_cross_lingual_similarity(text1, text2, lang1, lang2)
                similarity_results.append({
                    'text_pair': (text1, text2),
                    'languages': (lang1, lang2),
                    'similarity': similarity
                })
                logger.info(f"Similarity '{text1}' <-> '{text2}': {similarity:.3f}")
            
            # Test text alignment
            aligned_texts = aligner.align_texts(
                texts=["emergency response", "humanitarian aid"],
                source_lang='en',
                target_lang='sw'
            )
            
            return {
                "success": True,
                "alignment_learning": alignment_result,
                "similarity_tests": similarity_results,
                "text_alignment": aligned_texts,
                "average_similarity": sum(r['similarity'] for r in similarity_results) / len(similarity_results)
            }
            
        except Exception as e:
            logger.error(f"Cross-lingual alignment test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_novel_algorithm_discovery(self) -> dict:
        """Test novel algorithm discovery system."""
        logger.info("Testing novel algorithm discovery...")
        
        try:
            # Initialize discovery system
            discovery_system = NovelAlgorithmDiscovery(exploration_budget=100)
            
            # Mock performance evaluator
            def mock_evaluator(config):
                # Simulate performance based on config complexity
                complexity = len(str(config))
                import random
                base_performance = random.uniform(0.5, 0.9)
                return base_performance * (1 + complexity / 1000)
            
            # Discover algorithms
            discovered_algorithms = discovery_system.discover_algorithms(
                problem_domain="humanitarian_multimodal",
                performance_evaluator=mock_evaluator
            )
            
            # Validate discoveries
            assert isinstance(discovered_algorithms, list)
            
            algorithm_analysis = {
                'total_discovered': len(discovered_algorithms),
                'performance_range': (
                    min(a['performance'] for a in discovered_algorithms) if discovered_algorithms else 0,
                    max(a['performance'] for a in discovered_algorithms) if discovered_algorithms else 0
                ),
                'novelty_scores': [a.get('novelty_score', 0) for a in discovered_algorithms],
                'algorithm_types': list(set(a['config'].get('learning_algorithms', 'unknown') for a in discovered_algorithms))
            }
            
            logger.info(f"Discovered {len(discovered_algorithms)} novel algorithms")
            if discovered_algorithms:
                best_algorithm = max(discovered_algorithms, key=lambda x: x['performance'])
                logger.info(f"Best algorithm performance: {best_algorithm['performance']:.3f}")
            
            return {
                "success": True,
                "discovered_algorithms": discovered_algorithms[:5],  # Top 5
                "analysis": algorithm_analysis,
                "exploration_completed": True
            }
            
        except Exception as e:
            logger.error(f"Algorithm discovery test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_research_hypothesis_generation(self) -> dict:
        """Test research hypothesis generation."""
        logger.info("Testing research hypothesis generation...")
        
        try:
            # Initialize hypothesis generator
            hypothesis_gen = ResearchHypothesisGenerator()
            
            # Mock domain knowledge
            domain_knowledge = {
                "available_methods": ["adaptive_ocr", "cross_lingual_alignment", "synthetic_generation"],
                "target_languages": self.target_languages,
                "performance_baselines": {"ocr_accuracy": 0.75, "alignment_quality": 0.68}
            }
            
            # Mock experimental data
            experimental_data = [
                {"method": "adaptive_ocr", "performance": 0.82, "parameters": {"confidence_threshold": 0.7}},
                {"method": "adaptive_ocr", "performance": 0.79, "parameters": {"confidence_threshold": 0.6}},
                {"method": "cross_lingual_alignment", "performance": 0.71, "parameters": {"embedding_dim": 768}},
                {"method": "cross_lingual_alignment", "performance": 0.74, "parameters": {"embedding_dim": 512}},
                {"method": "synthetic_generation", "performance": 0.69, "parameters": {"num_templates": 100}}
            ]
            
            # Generate hypotheses
            hypotheses = hypothesis_gen.generate_hypotheses(
                domain_knowledge=domain_knowledge,
                experimental_data=experimental_data
            )
            
            # Validate hypotheses
            assert isinstance(hypotheses, list)
            
            hypothesis_analysis = {
                'total_generated': len(hypotheses),
                'confidence_distribution': [h.confidence for h in hypotheses] if hypotheses else [],
                'hypothesis_types': list(set(h.description.split()[0] for h in hypotheses)) if hypotheses else [],
                'validation_status': [h.validation_status for h in hypotheses] if hypotheses else []
            }
            
            logger.info(f"Generated {len(hypotheses)} research hypotheses")
            for i, hypothesis in enumerate(hypotheses[:3]):  # Log first 3
                logger.info(f"Hypothesis {i+1}: {hypothesis.description[:60]}...")
            
            return {
                "success": True,
                "hypotheses": [h.to_dict() for h in hypotheses[:5]],  # Top 5
                "analysis": hypothesis_analysis,
                "generation_completed": True
            }
            
        except Exception as e:
            logger.error(f"Hypothesis generation test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_performance_tracking(self) -> dict:
        """Test performance tracking system."""
        logger.info("Testing performance tracking...")
        
        try:
            # Initialize tracker
            tracker = PerformanceTracker()
            
            # Log various metrics
            metrics_to_track = {
                'ocr_accuracy': [0.75, 0.78, 0.82, 0.80, 0.85],
                'alignment_quality': [0.68, 0.71, 0.69, 0.74, 0.76],
                'synthesis_quality': [0.72, 0.74, 0.71, 0.78, 0.80],
                'overall_performance': [0.70, 0.73, 0.75, 0.77, 0.81]
            }
            
            for metric_name, values in metrics_to_track.items():
                for value in values:
                    tracker.log_metric(metric_name, value)
                    time.sleep(0.1)  # Small delay to simulate real timing
            
            # Get current metrics
            current_metrics = tracker.get_current_metrics()
            assert len(current_metrics) == len(metrics_to_track)
            
            # Get summary
            summary = tracker.get_summary()
            assert 'ocr_accuracy' in summary
            
            # Get detailed analysis
            detailed_analysis = tracker.get_detailed_analysis()
            assert 'total_metrics_tracked' in detailed_analysis
            assert detailed_analysis['total_metrics_tracked'] == len(metrics_to_track)
            
            logger.info(f"Tracked {detailed_analysis['data_points_collected']} data points across {len(metrics_to_track)} metrics")
            
            return {
                "success": True,
                "metrics_tracked": list(metrics_to_track.keys()),
                "summary": summary,
                "detailed_analysis": detailed_analysis,
                "tracking_operational": True
            }
            
        except Exception as e:
            logger.error(f"Performance tracking test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_integrated_pipeline(self) -> dict:
        """Test integrated intelligence pipeline."""
        logger.info("Testing integrated pipeline...")
        
        try:
            # Initialize pipeline
            pipeline = IntelligentHumanitarianPipeline(
                target_languages=self.target_languages,
                config={
                    'algorithm_exploration_budget': 50,
                    'ocr_engines': ['tesseract'],
                    'research_cycle_hours': 0.01  # Very short for testing
                }
            )
            
            # Test pipeline initialization
            assert pipeline.target_languages == self.target_languages
            assert hasattr(pipeline, 'adaptive_ocr')
            assert hasattr(pipeline, 'dataset_generator')
            assert hasattr(pipeline, 'algorithm_discovery')
            
            # Test individual component access
            dataset_result = pipeline.dataset_generator.generate_synthetic_dataset(
                num_samples=10,
                strategy='template_variation'
            )
            assert 'samples' in dataset_result
            assert len(dataset_result['samples']) > 0
            
            # Test OCR component
            mock_image = self._create_mock_image()
            ocr_result = pipeline.adaptive_ocr.extract_text(mock_image)
            assert 'text' in ocr_result
            assert 'confidence' in ocr_result
            
            # Test cross-lingual component
            aligned_text = pipeline.cross_lingual_aligner.align_cross_lingual(
                "emergency response", "sw"
            )
            assert isinstance(aligned_text, str)
            assert len(aligned_text) > 0
            
            logger.info("Integrated pipeline components all functional")
            
            return {
                "success": True,
                "pipeline_initialized": True,
                "components_functional": {
                    "dataset_generator": len(dataset_result['samples']) > 0,
                    "adaptive_ocr": ocr_result['confidence'] > 0,
                    "cross_lingual_aligner": len(aligned_text) > 0,
                    "algorithm_discovery": hasattr(pipeline.algorithm_discovery, 'discover_algorithms'),
                    "performance_tracker": hasattr(pipeline.performance_tracker, 'log_metric')
                },
                "target_languages": pipeline.target_languages
            }
            
        except Exception as e:
            logger.error(f"Integrated pipeline test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_mock_image(self):
        """Create mock image for testing."""
        # Simple mock that can be used with our OCR system
        return "mock_image_data_for_testing"
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] == 'PASSED')
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            "Enhanced Generation 1 Test Report": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": f"{success_rate:.1f}%",
                "test_results": self.test_results
            }
        }
        
        logger.info("=" * 80)
        logger.info("🎯 ENHANCED GENERATION 1 TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info("=" * 80)
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{status_emoji} {test_name}: {result['status']}")
        
        logger.info("=" * 80)
        
        # Save detailed report
        with open("enhanced_generation1_test_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("📄 Detailed report saved to: enhanced_generation1_test_report.json")


def main():
    """Run enhanced Generation 1 tests."""
    test_suite = EnhancedGeneration1Test()
    results = test_suite.run_all_tests()
    
    # Final summary
    passed = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total = len(results)
    
    if passed == total:
        logger.info("🎉 ALL ENHANCED GENERATION 1 TESTS PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed out of {total}")
        return 1


if __name__ == "__main__":
    exit(main())