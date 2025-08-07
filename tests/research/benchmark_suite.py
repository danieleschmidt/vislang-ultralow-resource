"""Research-grade benchmarking suite for vision-language models in humanitarian contexts."""

import time
import logging
import json
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import scipy.stats as stats

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result data structure."""
    task_name: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    execution_time: float
    memory_usage: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_name': self.task_name,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'metrics': self.metrics,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class HumanitarianVLBenchmark:
    """Comprehensive benchmark suite for humanitarian vision-language tasks."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.baseline_results = {}
        
        # Define benchmark tasks
        self.benchmark_tasks = {
            'humanitarian_classification': self._benchmark_humanitarian_classification,
            'crisis_detection': self._benchmark_crisis_detection,
            'multilingual_ocr': self._benchmark_multilingual_ocr,
            'cross_lingual_alignment': self._benchmark_cross_lingual_alignment,
            'scene_understanding': self._benchmark_scene_understanding,
            'document_analysis': self._benchmark_document_analysis
        }
        
        # Evaluation metrics
        self.metrics_functions = {
            'accuracy': accuracy_score,
            'precision_macro': lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='macro')[0],
            'recall_macro': lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='macro')[1],
            'f1_macro': lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='macro')[2],
            'precision_micro': lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='micro')[0],
            'recall_micro': lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='micro')[1],
            'f1_micro': lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='micro')[2]
        }
        
        logger.info("Humanitarian VL Benchmark initialized")
    
    def run_benchmark(self, models: List[Any], datasets: List[Any], 
                     tasks: List[str] = None, runs: int = 3) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across models and datasets."""
        if tasks is None:
            tasks = list(self.benchmark_tasks.keys())
        
        logger.info(f"Starting benchmark: {len(models)} models, {len(datasets)} datasets, {len(tasks)} tasks")
        
        all_results = defaultdict(list)
        
        for task_name in tasks:
            if task_name not in self.benchmark_tasks:
                logger.warning(f"Unknown task: {task_name}")
                continue
            
            logger.info(f"Benchmarking task: {task_name}")
            
            for model in models:
                for dataset in datasets:
                    # Run multiple times for statistical reliability
                    task_results = []
                    
                    for run_idx in range(runs):
                        logger.info(f"Run {run_idx + 1}/{runs} - Model: {model.__class__.__name__}, Dataset: {dataset.get('name', 'Unknown')}")
                        
                        try:
                            result = self._run_single_benchmark(
                                task_name, model, dataset, run_idx
                            )
                            task_results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Benchmark failed: {e}")
                            continue
                    
                    # Aggregate results across runs
                    if task_results:
                        aggregated_result = self._aggregate_results(task_results)
                        all_results[task_name].append(aggregated_result)
                        self.results.append(aggregated_result)
        
        # Save results
        self._save_results(all_results)
        
        logger.info("Benchmark completed")
        return dict(all_results)
    
    def _run_single_benchmark(self, task_name: str, model: Any, dataset: Any, run_idx: int) -> BenchmarkResult:
        """Run single benchmark instance."""
        import psutil
        import gc
        
        # Clear memory before benchmark
        gc.collect()
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Run benchmark task
        benchmark_func = self.benchmark_tasks[task_name]
        predictions, ground_truth, task_metadata = benchmark_func(model, dataset)
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truth, task_metadata)
        
        # Create result
        result = BenchmarkResult(
            task_name=task_name,
            model_name=model.__class__.__name__,
            dataset_name=dataset.get('name', 'Unknown'),
            metrics=metrics,
            execution_time=execution_time,
            memory_usage=memory_usage,
            parameters=getattr(model, 'get_config', lambda: {})(),
            metadata={
                'run_index': run_idx,
                'dataset_size': len(ground_truth),
                **task_metadata
            }
        )
        
        return result
    
    def _aggregate_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Aggregate results from multiple runs."""
        if len(results) == 1:
            return results[0]
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric_name in results[0].metrics.keys():
            values = [r.metrics[metric_name] for r in results]
            aggregated_metrics[metric_name] = statistics.mean(values)
            aggregated_metrics[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregated_metrics[f"{metric_name}_min"] = min(values)
            aggregated_metrics[f"{metric_name}_max"] = max(values)
        
        # Aggregate other fields
        base_result = results[0]
        base_result.metrics = aggregated_metrics
        base_result.execution_time = statistics.mean([r.execution_time for r in results])
        base_result.memory_usage = statistics.mean([r.memory_usage for r in results])
        base_result.metadata['num_runs'] = len(results)
        
        return base_result
    
    def _benchmark_humanitarian_classification(self, model: Any, dataset: Any) -> Tuple[List, List, Dict]:
        """Benchmark humanitarian image classification."""
        predictions = []
        ground_truth = []
        
        # Mock implementation - would use real humanitarian image dataset
        for i in range(100):  # Simulate 100 samples
            # Mock humanitarian scene categories
            true_label = np.random.choice(['refugee_camp', 'flood', 'drought', 'normal'])
            
            # Mock prediction - would use actual model inference
            if hasattr(model, 'predict'):
                pred_label = model.predict(f"sample_{i}")
            else:
                pred_label = np.random.choice(['refugee_camp', 'flood', 'drought', 'normal'])
            
            predictions.append(pred_label)
            ground_truth.append(true_label)
        
        metadata = {
            'task_type': 'classification',
            'num_classes': 4,
            'classes': ['refugee_camp', 'flood', 'drought', 'normal']
        }
        
        return predictions, ground_truth, metadata
    
    def _benchmark_crisis_detection(self, model: Any, dataset: Any) -> Tuple[List, List, Dict]:
        """Benchmark crisis detection in images."""
        predictions = []
        ground_truth = []
        
        # Mock binary crisis detection
        for i in range(200):
            true_label = np.random.choice([0, 1])  # 0: no crisis, 1: crisis
            
            if hasattr(model, 'detect_crisis'):
                pred_label = model.detect_crisis(f"sample_{i}")
            else:
                pred_label = np.random.choice([0, 1])
            
            predictions.append(pred_label)
            ground_truth.append(true_label)
        
        metadata = {
            'task_type': 'binary_classification',
            'positive_class': 'crisis',
            'negative_class': 'normal'
        }
        
        return predictions, ground_truth, metadata
    
    def _benchmark_multilingual_ocr(self, model: Any, dataset: Any) -> Tuple[List, List, Dict]:
        """Benchmark multilingual OCR accuracy."""
        predictions = []
        ground_truth = []
        
        # Mock OCR accuracy across different languages
        languages = ['en', 'fr', 'ar', 'sw', 'am']
        
        for lang in languages:
            for i in range(50):  # 50 samples per language
                # Mock ground truth text
                true_text = f"Sample text in {lang} language {i}"
                
                if hasattr(model, 'extract_text'):
                    pred_text = model.extract_text(f"image_{lang}_{i}")
                else:
                    # Mock OCR with some errors
                    pred_text = true_text if np.random.random() > 0.2 else f"Error text {i}"
                
                predictions.append(pred_text)
                ground_truth.append(true_text)
        
        metadata = {
            'task_type': 'text_extraction',
            'languages': languages,
            'samples_per_language': 50
        }
        
        return predictions, ground_truth, metadata
    
    def _benchmark_cross_lingual_alignment(self, model: Any, dataset: Any) -> Tuple[List, List, Dict]:
        """Benchmark cross-lingual text alignment."""
        predictions = []
        ground_truth = []
        
        # Mock alignment accuracy
        language_pairs = [('en', 'fr'), ('en', 'ar'), ('en', 'sw')]
        
        for source_lang, target_lang in language_pairs:
            for i in range(30):  # 30 samples per pair
                # Mock alignment score (0-1)
                true_score = np.random.uniform(0.5, 1.0)  # True alignment scores
                
                if hasattr(model, 'align_texts'):
                    pred_score = model.align_texts(f"text_{source_lang}_{i}", f"text_{target_lang}_{i}")
                else:
                    pred_score = np.random.uniform(0.0, 1.0)
                
                predictions.append(pred_score)
                ground_truth.append(true_score)
        
        metadata = {
            'task_type': 'alignment_scoring',
            'language_pairs': language_pairs,
            'samples_per_pair': 30
        }
        
        return predictions, ground_truth, metadata
    
    def _benchmark_scene_understanding(self, model: Any, dataset: Any) -> Tuple[List, List, Dict]:
        """Benchmark humanitarian scene understanding."""
        predictions = []
        ground_truth = []
        
        # Mock scene understanding with multiple attributes
        attributes = ['people_count', 'infrastructure_damage', 'weather_condition']
        
        for i in range(150):
            true_attrs = {
                'people_count': np.random.choice(['few', 'many', 'crowd']),
                'infrastructure_damage': np.random.choice(['none', 'moderate', 'severe']),
                'weather_condition': np.random.choice(['clear', 'rainy', 'stormy'])
            }
            
            if hasattr(model, 'understand_scene'):
                pred_attrs = model.understand_scene(f"scene_{i}")
            else:
                pred_attrs = {
                    'people_count': np.random.choice(['few', 'many', 'crowd']),
                    'infrastructure_damage': np.random.choice(['none', 'moderate', 'severe']),
                    'weather_condition': np.random.choice(['clear', 'rainy', 'stormy'])
                }
            
            predictions.append(pred_attrs)
            ground_truth.append(true_attrs)
        
        metadata = {
            'task_type': 'multi_attribute_classification',
            'attributes': attributes,
            'num_samples': 150
        }
        
        return predictions, ground_truth, metadata
    
    def _benchmark_document_analysis(self, model: Any, dataset: Any) -> Tuple[List, List, Dict]:
        """Benchmark humanitarian document analysis."""
        predictions = []
        ground_truth = []
        
        # Mock document type classification
        doc_types = ['report', 'infographic', 'map', 'chart', 'table']
        
        for i in range(100):
            true_type = np.random.choice(doc_types)
            
            if hasattr(model, 'analyze_document'):
                pred_type = model.analyze_document(f"document_{i}")
            else:
                pred_type = np.random.choice(doc_types)
            
            predictions.append(pred_type)
            ground_truth.append(true_type)
        
        metadata = {
            'task_type': 'document_classification',
            'document_types': doc_types,
            'num_samples': 100
        }
        
        return predictions, ground_truth, metadata
    
    def _calculate_metrics(self, predictions: List, ground_truth: List, 
                          task_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate evaluation metrics based on task type."""
        metrics = {}
        
        task_type = task_metadata.get('task_type', 'classification')
        
        if task_type in ['classification', 'binary_classification', 'document_classification']:
            # Classification metrics
            for metric_name, metric_func in self.metrics_functions.items():
                try:
                    metrics[metric_name] = metric_func(ground_truth, predictions)
                except Exception as e:
                    logger.warning(f"Could not calculate {metric_name}: {e}")
                    metrics[metric_name] = 0.0
        
        elif task_type == 'text_extraction':
            # OCR-specific metrics
            metrics.update(self._calculate_ocr_metrics(predictions, ground_truth))
        
        elif task_type == 'alignment_scoring':
            # Alignment-specific metrics
            metrics.update(self._calculate_alignment_metrics(predictions, ground_truth))
        
        elif task_type == 'multi_attribute_classification':
            # Multi-attribute metrics
            metrics.update(self._calculate_multi_attribute_metrics(predictions, ground_truth))
        
        return metrics
    
    def _calculate_ocr_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Calculate OCR-specific metrics."""
        from difflib import SequenceMatcher
        
        # Character-level accuracy
        char_accuracies = []
        word_accuracies = []
        
        for pred, true in zip(predictions, ground_truth):
            # Character accuracy
            char_acc = SequenceMatcher(None, pred, true).ratio()
            char_accuracies.append(char_acc)
            
            # Word accuracy
            pred_words = set(pred.split())
            true_words = set(true.split())
            if true_words:
                word_acc = len(pred_words & true_words) / len(true_words)
                word_accuracies.append(word_acc)
        
        return {
            'character_accuracy': statistics.mean(char_accuracies),
            'word_accuracy': statistics.mean(word_accuracies),
            'exact_match': sum(1 for p, t in zip(predictions, ground_truth) if p == t) / len(predictions)
        }
    
    def _calculate_alignment_metrics(self, predictions: List[float], ground_truth: List[float]) -> Dict[str, float]:
        """Calculate alignment-specific metrics."""
        # Regression metrics for alignment scores
        mse = statistics.mean([(p - t) ** 2 for p, t in zip(predictions, ground_truth)])
        mae = statistics.mean([abs(p - t) for p, t in zip(predictions, ground_truth)])
        
        # Correlation
        correlation = np.corrcoef(predictions, ground_truth)[0, 1] if len(predictions) > 1 else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'correlation': correlation
        }
    
    def _calculate_multi_attribute_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate multi-attribute classification metrics."""
        metrics = {}
        
        if not predictions or not ground_truth:
            return metrics
        
        # Get all attributes
        attributes = set()
        for gt in ground_truth:
            attributes.update(gt.keys())
        
        # Calculate accuracy for each attribute
        for attr in attributes:
            attr_pred = [p.get(attr) for p in predictions]
            attr_true = [t.get(attr) for t in ground_truth]
            
            # Filter out None values
            valid_pairs = [(p, t) for p, t in zip(attr_pred, attr_true) if p is not None and t is not None]
            
            if valid_pairs:
                attr_pred_valid, attr_true_valid = zip(*valid_pairs)
                accuracy = sum(1 for p, t in valid_pairs if p == t) / len(valid_pairs)
                metrics[f'{attr}_accuracy'] = accuracy
        
        # Overall accuracy (all attributes correct)
        exact_matches = sum(
            1 for p, t in zip(predictions, ground_truth)
            if all(p.get(k) == t.get(k) for k in t.keys())
        )
        metrics['exact_match_all_attributes'] = exact_matches / len(predictions)
        
        return metrics
    
    def _save_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to files."""
        # Save as JSON
        json_results = {
            task_name: [result.to_dict() for result in task_results]
            for task_name, task_results in results.items()
        }
        
        output_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Generate report
        self._generate_report(results)
    
    def _generate_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate benchmark report with visualizations."""
        report_dir = self.output_dir / f"report_{int(time.time())}"
        report_dir.mkdir(exist_ok=True)
        
        # Create visualizations
        self._create_performance_plots(results, report_dir)
        
        # Generate summary statistics
        summary = self._generate_summary_stats(results)
        
        # Save summary
        with open(report_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(results, summary, report_dir)
        
        logger.info(f"Benchmark report generated in {report_dir}")
    
    def _create_performance_plots(self, results: Dict[str, List[BenchmarkResult]], output_dir: Path):
        """Create performance visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            
            # Accuracy comparison across tasks
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Humanitarian Vision-Language Benchmark Results')
            
            # Task performance comparison
            task_names = list(results.keys())
            model_names = list(set([r.model_name for task_results in results.values() for r in task_results]))
            
            # Accuracy heatmap
            accuracy_matrix = np.zeros((len(task_names), len(model_names)))
            for i, task_name in enumerate(task_names):
                for j, model_name in enumerate(model_names):
                    task_results = [r for r in results[task_name] if r.model_name == model_name]
                    if task_results:
                        # Use first available accuracy metric
                        for metric_name in ['accuracy', 'exact_match', 'character_accuracy']:
                            if metric_name in task_results[0].metrics:
                                accuracy_matrix[i, j] = task_results[0].metrics[metric_name]
                                break
            
            sns.heatmap(accuracy_matrix, 
                       xticklabels=model_names,
                       yticklabels=task_names,
                       annot=True, fmt='.3f', 
                       cmap='viridis',
                       ax=axes[0, 0])
            axes[0, 0].set_title('Model Performance Across Tasks')
            
            # Execution time comparison
            exec_times = defaultdict(list)
            for task_results in results.values():
                for result in task_results:
                    exec_times[result.model_name].append(result.execution_time)
            
            model_names_time = list(exec_times.keys())
            avg_times = [statistics.mean(exec_times[model]) for model in model_names_time]
            
            axes[0, 1].bar(model_names_time, avg_times)
            axes[0, 1].set_title('Average Execution Time by Model')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Memory usage comparison
            memory_usage = defaultdict(list)
            for task_results in results.values():
                for result in task_results:
                    memory_usage[result.model_name].append(result.memory_usage)
            
            model_names_mem = list(memory_usage.keys())
            avg_memory = [statistics.mean(memory_usage[model]) for model in model_names_mem]
            
            axes[1, 0].bar(model_names_mem, avg_memory)
            axes[1, 0].set_title('Average Memory Usage by Model')
            axes[1, 0].set_ylabel('Memory (MB)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Task difficulty analysis
            task_avg_scores = []
            for task_name in task_names:
                task_results = results[task_name]
                scores = []
                for result in task_results:
                    # Get primary metric
                    if 'accuracy' in result.metrics:
                        scores.append(result.metrics['accuracy'])
                    elif 'exact_match' in result.metrics:
                        scores.append(result.metrics['exact_match'])
                
                if scores:
                    task_avg_scores.append(statistics.mean(scores))
                else:
                    task_avg_scores.append(0.0)
            
            axes[1, 1].bar(task_names, task_avg_scores)
            axes[1, 1].set_title('Task Difficulty (Average Performance)')
            axes[1, 1].set_ylabel('Average Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'benchmark_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance plots created")
            
        except Exception as e:
            logger.error(f"Could not create plots: {e}")
    
    def _generate_summary_stats(self, results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_tasks': len(results),
            'total_benchmarks': sum(len(task_results) for task_results in results.values()),
            'tasks': {},
            'models': defaultdict(lambda: defaultdict(list)),
            'overall_stats': {}
        }
        
        all_results = []
        for task_results in results.values():
            all_results.extend(task_results)
        
        # Overall statistics
        if all_results:
            summary['overall_stats'] = {
                'avg_execution_time': statistics.mean([r.execution_time for r in all_results]),
                'avg_memory_usage': statistics.mean([r.memory_usage for r in all_results]),
                'total_execution_time': sum([r.execution_time for r in all_results])
            }
        
        # Task-specific statistics
        for task_name, task_results in results.items():
            if not task_results:
                continue
            
            task_stats = {
                'num_benchmarks': len(task_results),
                'models': list(set([r.model_name for r in task_results])),
                'datasets': list(set([r.dataset_name for r in task_results])),
                'avg_execution_time': statistics.mean([r.execution_time for r in task_results]),
                'avg_memory_usage': statistics.mean([r.memory_usage for r in task_results])
            }
            
            # Aggregate metrics
            all_metrics = defaultdict(list)
            for result in task_results:
                for metric_name, metric_value in result.metrics.items():
                    all_metrics[metric_name].append(metric_value)
            
            task_stats['metrics'] = {}
            for metric_name, values in all_metrics.items():
                task_stats['metrics'][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values)
                }
            
            summary['tasks'][task_name] = task_stats
        
        # Model comparison
        for result in all_results:
            model_name = result.model_name
            for metric_name, metric_value in result.metrics.items():
                summary['models'][model_name][metric_name].append(metric_value)
        
        # Calculate model averages
        model_summary = {}
        for model_name, metrics in summary['models'].items():
            model_summary[model_name] = {}
            for metric_name, values in metrics.items():
                model_summary[model_name][metric_name] = {
                    'mean': statistics.mean(values),
                    'count': len(values)
                }
        
        summary['models'] = model_summary
        
        return summary
    
    def _generate_markdown_report(self, results: Dict[str, List[BenchmarkResult]], 
                                 summary: Dict[str, Any], output_dir: Path):
        """Generate markdown benchmark report."""
        report_content = f"""# Humanitarian Vision-Language Benchmark Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overview

- **Total Tasks**: {summary['total_tasks']}
- **Total Benchmarks**: {summary['total_benchmarks']}
- **Average Execution Time**: {summary['overall_stats'].get('avg_execution_time', 0):.3f}s
- **Average Memory Usage**: {summary['overall_stats'].get('avg_memory_usage', 0):.1f}MB

## Task Performance Summary

"""
        
        for task_name, task_stats in summary['tasks'].items():
            report_content += f"""### {task_name.replace('_', ' ').title()}

- **Models Tested**: {len(task_stats['models'])}
- **Datasets Used**: {len(task_stats['datasets'])}
- **Average Execution Time**: {task_stats['avg_execution_time']:.3f}s
- **Average Memory Usage**: {task_stats['avg_memory_usage']:.1f}MB

**Key Metrics:**
"""
            
            for metric_name, metric_stats in task_stats['metrics'].items():
                if not metric_name.endswith('_std') and not metric_name.endswith('_min') and not metric_name.endswith('_max'):
                    report_content += f"- **{metric_name.replace('_', ' ').title()}**: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}\n"
            
            report_content += "\n"
        
        report_content += """## Model Comparison

| Model | Tasks Completed | Avg Accuracy | Avg Execution Time |
|-------|----------------|--------------|-------------------|
"""
        
        for model_name, model_stats in summary['models'].items():
            accuracy = model_stats.get('accuracy', {}).get('mean', 0.0)
            task_count = sum(stats['count'] for stats in model_stats.values())
            
            # Find execution time from original results
            model_results = [r for task_results in results.values() for r in task_results if r.model_name == model_name]
            avg_exec_time = statistics.mean([r.execution_time for r in model_results]) if model_results else 0.0
            
            report_content += f"| {model_name} | {task_count} | {accuracy:.3f} | {avg_exec_time:.3f}s |\n"
        
        report_content += """
## Recommendations

Based on the benchmark results:

1. **Performance Optimization**: Focus on models with high accuracy but reasonable execution times
2. **Resource Efficiency**: Consider memory usage for deployment in resource-constrained environments
3. **Task-Specific Tuning**: Some tasks may benefit from specialized model architectures
4. **Cross-Lingual Performance**: Evaluate multilingual capabilities for humanitarian contexts

## Files Generated

- `benchmark_overview.png`: Performance visualization
- `summary.json`: Detailed statistics in JSON format
- Raw results in benchmark_results_*.json files

"""
        
        # Save report
        with open(output_dir / "README.md", 'w') as f:
            f.write(report_content)
        
        logger.info("Markdown report generated")


class CrossLingualBenchmark:
    """Specialized benchmark for cross-lingual capabilities."""
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['en', 'fr', 'es', 'ar', 'sw', 'am', 'ha']
        self.results = []
        
    def run_cross_lingual_benchmark(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Run cross-lingual benchmark."""
        results = {}
        
        # Test each language
        for lang in self.languages:
            lang_results = self._test_language_performance(model, lang, test_data.get(lang, []))
            results[lang] = lang_results
        
        # Test cross-lingual alignment
        alignment_results = self._test_cross_lingual_alignment(model, test_data)
        results['cross_lingual_alignment'] = alignment_results
        
        return results
    
    def _test_language_performance(self, model: Any, language: str, test_samples: List[Any]) -> Dict[str, float]:
        """Test model performance on specific language."""
        # Mock language-specific testing
        return {
            'accuracy': np.random.uniform(0.6, 0.95),
            'bleu_score': np.random.uniform(0.4, 0.8),
            'processing_time': np.random.uniform(0.1, 2.0)
        }
    
    def _test_cross_lingual_alignment(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Test cross-lingual alignment capabilities."""
        # Mock alignment testing
        return {
            'alignment_accuracy': np.random.uniform(0.5, 0.9),
            'zero_shot_performance': np.random.uniform(0.3, 0.7)
        }