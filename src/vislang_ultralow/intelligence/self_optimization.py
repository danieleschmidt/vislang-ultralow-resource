"""Self-optimization systems that autonomously improve performance.

Generation 4: Intelligent systems that monitor their own performance and
automatically adjust parameters for optimal efficiency.
"""

import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
from datetime import datetime, timedelta
import queue
import math
import os

# Conditional imports with fallbacks
try:
    import numpy as np
    from scipy import stats
except ImportError:
    np = None
    stats = None

try:
    import psutil
except ImportError:
    # Mock psutil for environments without it
    class psutil:
        @staticmethod
        def cpu_count(): return 4
        @staticmethod
        def cpu_percent(interval=1): return 50.0
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 60.0
                available = 8 * 1024**3
                total = 16 * 1024**3
            return MockMemory()
        @staticmethod
        def Process(pid=None):
            class MockProcess:
                def memory_info(self):
                    class MockMemInfo:
                        rss = 100 * 1024**2  # 100MB
                    return MockMemInfo()
                def cpu_percent(self): return 25.0
            return MockProcess()

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: int
    response_time: float
    throughput: float
    error_rate: float
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_available": self.memory_available,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate
        }

@dataclass
class OptimizationAction:
    """An optimization action taken by the system."""
    timestamp: datetime
    action_type: str
    parameters: Dict[str, Any]
    expected_impact: float
    actual_impact: Optional[float] = None
    success: bool = True

class AutonomousQualityController:
    """Monitors and autonomously controls system quality metrics."""
    
    def __init__(self, target_performance: float = 0.95, 
                 monitoring_interval: float = 30.0):
        """Initialize autonomous quality controller.
        
        Args:
            target_performance: Target performance threshold (0-1)
            monitoring_interval: Monitoring interval in seconds
        """
        self.target_performance = target_performance
        self.monitoring_interval = monitoring_interval
        
        # Quality metrics tracking
        self.quality_history: deque = deque(maxlen=1000)
        self.performance_trends = defaultdict(lambda: deque(maxlen=100))
        
        # Control parameters
        self.quality_thresholds = {
            "accuracy": 0.90,
            "response_time": 2.0,  # seconds
            "error_rate": 0.05,     # 5%
            "availability": 0.99    # 99%
        }
        
        # Control actions
        self.control_actions: List[OptimizationAction] = []
        self.active_controls = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        
        logger.info("AutonomousQualityController initialized")
    
    def start_monitoring(self):
        """Start autonomous quality monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Started autonomous quality monitoring")
    
    def stop_monitoring(self):
        """Stop autonomous quality monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped autonomous quality monitoring")
    
    def record_quality_metric(self, metric_name: str, value: float, 
                            context: Dict[str, Any] = None):
        """Record a quality metric for monitoring.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Additional context
        """
        metric_data = {
            "name": metric_name,
            "value": value,
            "timestamp": datetime.now(),
            "context": context or {}
        }
        
        self.metrics_queue.put(metric_data)
        self.performance_trends[metric_name].append(value)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Process queued metrics
                self._process_queued_metrics()
                
                # Analyze quality trends
                self._analyze_quality_trends()
                
                # Take corrective actions if needed
                self._take_corrective_actions()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause before retry
    
    def _process_queued_metrics(self):
        """Process all queued quality metrics."""
        processed = 0
        
        while not self.metrics_queue.empty() and processed < 100:
            try:
                metric = self.metrics_queue.get_nowait()
                self.quality_history.append(metric)
                processed += 1
            except queue.Empty:
                break
        
        if processed > 0:
            logger.debug(f"Processed {processed} quality metrics")
    
    def _analyze_quality_trends(self):
        """Analyze quality trends and identify issues."""
        current_time = datetime.now()
        issues_detected = []
        
        for metric_name, threshold in self.quality_thresholds.items():
            if metric_name in self.performance_trends:
                recent_values = list(self.performance_trends[metric_name])[-10:]
                
                if len(recent_values) >= 5:
                    avg_value = sum(recent_values) / len(recent_values)
                    
                    # Check threshold violations
                    if metric_name in ["accuracy", "availability"]:
                        # Higher is better
                        if avg_value < threshold:
                            issues_detected.append({
                                "metric": metric_name,
                                "current": avg_value,
                                "threshold": threshold,
                                "severity": "high" if avg_value < threshold * 0.9 else "medium"
                            })
                    else:
                        # Lower is better (response_time, error_rate)
                        if avg_value > threshold:
                            issues_detected.append({
                                "metric": metric_name,
                                "current": avg_value,
                                "threshold": threshold,
                                "severity": "high" if avg_value > threshold * 1.5 else "medium"
                            })
        
        # Log detected issues
        for issue in issues_detected:
            logger.warning(f"Quality issue detected: {issue['metric']} = {issue['current']:.4f} "
                         f"(threshold: {issue['threshold']}, severity: {issue['severity']})")
        
        # Store issues for action planning
        self.detected_issues = issues_detected
    
    def _take_corrective_actions(self):
        """Take corrective actions based on detected issues."""
        if not hasattr(self, 'detected_issues') or not self.detected_issues:
            return
        
        for issue in self.detected_issues:
            action = self._plan_corrective_action(issue)
            if action:
                self._execute_action(action)
    
    def _plan_corrective_action(self, issue: Dict[str, Any]) -> Optional[OptimizationAction]:
        """Plan corrective action for a quality issue."""
        metric = issue["metric"]
        severity = issue["severity"]
        
        action_params = {}
        expected_impact = 0.1  # Default expected improvement
        
        if metric == "response_time":
            if severity == "high":
                action_params = {
                    "increase_cache_size": True,
                    "enable_connection_pooling": True,
                    "optimize_query_batch_size": True
                }
                expected_impact = 0.3
            else:
                action_params = {
                    "increase_worker_threads": True,
                    "enable_request_batching": True
                }
                expected_impact = 0.15
                
        elif metric == "error_rate":
            action_params = {
                "increase_retry_attempts": True,
                "enable_graceful_degradation": True,
                "improve_input_validation": True
            }
            expected_impact = 0.25
            
        elif metric == "accuracy":
            action_params = {
                "increase_model_ensemble_size": True,
                "enable_confidence_filtering": True,
                "tune_decision_thresholds": True
            }
            expected_impact = 0.05
            
        elif metric == "availability":
            action_params = {
                "enable_health_checks": True,
                "increase_replica_count": True,
                "implement_circuit_breaker": True
            }
            expected_impact = 0.02
        
        if action_params:
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type=f"correct_{metric}",
                parameters=action_params,
                expected_impact=expected_impact
            )
        
        return None
    
    def _execute_action(self, action: OptimizationAction):
        """Execute a corrective action."""
        try:
            logger.info(f"Executing corrective action: {action.action_type}")
            
            # Simulate action execution
            # In real implementation, this would apply actual system changes
            success = self._simulate_action_execution(action)
            
            action.success = success
            action.actual_impact = self._measure_action_impact(action)
            
            self.control_actions.append(action)
            
            if success:
                logger.info(f"Action {action.action_type} executed successfully, "
                          f"impact: {action.actual_impact:.4f}")
            else:
                logger.warning(f"Action {action.action_type} failed to execute")
                
        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {e}")
            action.success = False
    
    def _simulate_action_execution(self, action: OptimizationAction) -> bool:
        """Simulate action execution (placeholder for real implementation)."""
        # In real implementation, this would:
        # - Update system configuration
        # - Restart services if needed
        # - Apply resource adjustments
        # - Update monitoring parameters
        
        action_type = action.action_type
        params = action.parameters
        
        # Simulate different action types
        if "response_time" in action_type:
            # Simulate cache/performance optimizations
            if params.get("increase_cache_size"):
                self.active_controls["cache_size_multiplier"] = 2.0
            if params.get("enable_connection_pooling"):
                self.active_controls["connection_pool_enabled"] = True
                
        elif "error_rate" in action_type:
            # Simulate reliability improvements
            if params.get("increase_retry_attempts"):
                self.active_controls["max_retries"] = 5
            if params.get("enable_graceful_degradation"):
                self.active_controls["graceful_degradation"] = True
                
        elif "accuracy" in action_type:
            # Simulate accuracy improvements
            if params.get("tune_decision_thresholds"):
                self.active_controls["decision_threshold"] = 0.85
                
        # Simulate 90% success rate
        if np is not None:
            return np.random.random() > 0.1
        else:
            return True  # Assume success in fallback
    
    def _measure_action_impact(self, action: OptimizationAction) -> float:
        """Measure the actual impact of an executed action."""
        # In real implementation, this would compare metrics before/after action
        
        # Simulate impact measurement
        expected = action.expected_impact
        
        if np is not None:
            # Add noise to expected impact
            actual = expected * np.random.normal(1.0, 0.2)
            return max(0.0, actual)
        else:
            return expected * 0.8  # Slightly less than expected
    
    def get_quality_status(self) -> Dict[str, Any]:
        """Get current quality control status."""
        status = {
            "monitoring_active": self.monitoring_active,
            "metrics_collected": len(self.quality_history),
            "active_controls": self.active_controls.copy(),
            "recent_actions": len([a for a in self.control_actions 
                                 if a.timestamp > datetime.now() - timedelta(hours=1)]),
            "quality_scores": {}
        }
        
        # Calculate current quality scores
        for metric_name in self.quality_thresholds:
            if metric_name in self.performance_trends:
                recent_values = list(self.performance_trends[metric_name])[-5:]
                if recent_values:
                    avg_value = sum(recent_values) / len(recent_values)
                    status["quality_scores"][metric_name] = avg_value
        
        return status

class PerformanceSelfTuner:
    """Automatically tunes system performance parameters."""
    
    def __init__(self, tuning_interval: float = 300.0):
        """Initialize performance self-tuner.
        
        Args:
            tuning_interval: Tuning interval in seconds
        """
        self.tuning_interval = tuning_interval
        
        # Performance metrics
        self.performance_history: deque = deque(maxlen=1000)
        self.tuning_history: List[Dict] = []
        
        # Tunable parameters
        self.tunable_params = {
            "batch_size": {"current": 32, "min": 8, "max": 128, "step": 8},
            "worker_threads": {"current": 4, "min": 1, "max": 16, "step": 1},
            "cache_ttl": {"current": 300, "min": 60, "max": 3600, "step": 60},
            "connection_timeout": {"current": 30, "min": 5, "max": 120, "step": 5}
        }
        
        # Tuning state
        self.tuning_active = False
        self.tuning_thread = None
        self.last_tuning = datetime.now()
        
        logger.info("PerformanceSelfTuner initialized")
    
    def start_tuning(self):
        """Start automatic performance tuning."""
        if self.tuning_active:
            return
        
        self.tuning_active = True
        self.tuning_thread = threading.Thread(target=self._tuning_loop)
        self.tuning_thread.daemon = True
        self.tuning_thread.start()
        
        logger.info("Started automatic performance tuning")
    
    def stop_tuning(self):
        """Stop automatic performance tuning."""
        self.tuning_active = False
        if self.tuning_thread:
            self.tuning_thread.join(timeout=5.0)
        
        logger.info("Stopped automatic performance tuning")
    
    def record_performance(self, metrics: Dict[str, float]):
        """Record performance metrics for tuning."""
        perf_data = {
            "timestamp": datetime.now(),
            "metrics": metrics.copy(),
            "parameters": {name: config["current"] 
                         for name, config in self.tunable_params.items()}
        }
        
        self.performance_history.append(perf_data)
    
    def _tuning_loop(self):
        """Main tuning loop."""
        while self.tuning_active:
            try:
                # Check if tuning is due
                time_since_tuning = datetime.now() - self.last_tuning
                
                if time_since_tuning.total_seconds() >= self.tuning_interval:
                    self._perform_tuning()
                    self.last_tuning = datetime.now()
                
                # Sleep until next check
                time.sleep(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in tuning loop: {e}")
                time.sleep(30.0)
    
    def _perform_tuning(self):
        """Perform parameter tuning based on performance history."""
        if len(self.performance_history) < 10:
            logger.info("Insufficient performance data for tuning")
            return
        
        logger.info("Starting performance parameter tuning")
        
        # Analyze current performance
        recent_metrics = list(self.performance_history)[-10:]
        baseline_performance = self._calculate_performance_score(recent_metrics)
        
        # Try tuning each parameter
        best_improvement = 0.0
        best_param = None
        best_value = None
        
        for param_name, config in self.tunable_params.items():
            improvement = self._evaluate_parameter_change(param_name, config, recent_metrics)
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_param = param_name
                best_value = self._suggest_parameter_value(param_name, config, improvement)
        
        # Apply best parameter change
        if best_param and best_improvement > 0.05:  # 5% improvement threshold
            old_value = self.tunable_params[best_param]["current"]
            self.tunable_params[best_param]["current"] = best_value
            
            tuning_record = {
                "timestamp": datetime.now().isoformat(),
                "parameter": best_param,
                "old_value": old_value,
                "new_value": best_value,
                "expected_improvement": best_improvement,
                "baseline_performance": baseline_performance
            }
            
            self.tuning_history.append(tuning_record)
            
            logger.info(f"Tuned {best_param}: {old_value} -> {best_value}, "
                       f"expected improvement: {best_improvement:.4f}")
        else:
            logger.info("No beneficial parameter changes found")
    
    def _calculate_performance_score(self, metrics_list: List[Dict]) -> float:
        """Calculate overall performance score from metrics."""
        if not metrics_list:
            return 0.0
        
        # Extract key performance indicators
        scores = []
        
        for metric_data in metrics_list:
            metrics = metric_data["metrics"]
            
            # Weighted performance score
            score = 0.0
            
            # Response time (lower is better)
            if "response_time" in metrics:
                response_score = max(0, 1.0 - metrics["response_time"] / 10.0)
                score += response_score * 0.3
            
            # Throughput (higher is better)
            if "throughput" in metrics:
                throughput_score = min(1.0, metrics["throughput"] / 100.0)
                score += throughput_score * 0.3
            
            # Error rate (lower is better)
            if "error_rate" in metrics:
                error_score = max(0, 1.0 - metrics["error_rate"])
                score += error_score * 0.2
            
            # CPU usage (moderate is best)
            if "cpu_usage" in metrics:
                cpu_usage = metrics["cpu_usage"]
                cpu_score = 1.0 - abs(cpu_usage - 0.7) / 0.7  # Target 70% CPU
                score += max(0, cpu_score) * 0.1
            
            # Memory usage (lower is better)
            if "memory_usage" in metrics:
                memory_score = max(0, 1.0 - metrics["memory_usage"])
                score += memory_score * 0.1
            
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _evaluate_parameter_change(self, param_name: str, config: Dict, 
                                 recent_metrics: List[Dict]) -> float:
        """Evaluate potential improvement from changing a parameter."""
        current_value = config["current"]
        
        # Simulate parameter change impact
        if param_name == "batch_size":
            # Larger batch size typically improves throughput but may increase latency
            if current_value < 64:
                return 0.1  # Expect 10% improvement
            else:
                return -0.05  # May hurt performance
                
        elif param_name == "worker_threads":
            # More threads help with concurrent workloads
            cpu_usage = self._get_average_metric(recent_metrics, "cpu_usage", 0.5)
            if cpu_usage < 0.8 and current_value < 8:
                return 0.15  # Good opportunity for parallelization
            elif cpu_usage > 0.9:
                return -0.1  # Already CPU bound
            else:
                return 0.05
                
        elif param_name == "cache_ttl":
            # Longer TTL reduces load but may serve stale data
            hit_rate = self._get_average_metric(recent_metrics, "cache_hit_rate", 0.7)
            if hit_rate < 0.8:
                return 0.08  # Improve cache effectiveness
            else:
                return 0.02
                
        elif param_name == "connection_timeout":
            # Shorter timeout fails faster, longer timeout is more patient
            error_rate = self._get_average_metric(recent_metrics, "error_rate", 0.05)
            if error_rate > 0.1:
                return 0.12  # Reduce timeout errors
            else:
                return 0.03
        
        return 0.0
    
    def _get_average_metric(self, metrics_list: List[Dict], metric_name: str, 
                           default: float) -> float:
        """Get average value of a metric from metrics list."""
        values = []
        
        for metric_data in metrics_list:
            if metric_name in metric_data["metrics"]:
                values.append(metric_data["metrics"][metric_name])
        
        return sum(values) / len(values) if values else default
    
    def _suggest_parameter_value(self, param_name: str, config: Dict, 
                                improvement: float) -> Any:
        """Suggest new value for a parameter based on expected improvement."""
        current = config["current"]
        min_val, max_val, step = config["min"], config["max"], config["step"]
        
        # Scale change based on expected improvement
        if improvement > 0.1:  # Large improvement expected
            change_factor = 2
        elif improvement > 0.05:  # Moderate improvement
            change_factor = 1
        else:  # Small improvement
            change_factor = 1
        
        # Direction of change depends on parameter type
        if param_name in ["batch_size", "worker_threads", "cache_ttl"]:
            # Generally, increasing these helps performance (up to a point)
            new_value = min(max_val, current + step * change_factor)
        else:
            # For connection_timeout, optimal value depends on workload
            if improvement > 0.08:
                new_value = max(min_val, current - step * change_factor)
            else:
                new_value = min(max_val, current + step * change_factor)
        
        return new_value
    
    def get_tuning_status(self) -> Dict[str, Any]:
        """Get current tuning status."""
        return {
            "tuning_active": self.tuning_active,
            "current_parameters": {name: config["current"] 
                                 for name, config in self.tunable_params.items()},
            "recent_tunings": len([t for t in self.tuning_history 
                                 if datetime.fromisoformat(t["timestamp"]) > 
                                 datetime.now() - timedelta(hours=24)]),
            "performance_samples": len(self.performance_history),
            "last_tuning": self.last_tuning.isoformat()
        }

class ResourceAdaptationEngine:
    """Adapts resource allocation based on workload patterns."""
    
    def __init__(self):
        self.resource_history: deque = deque(maxlen=1000)
        self.allocation_rules = {}
        self.adaptation_active = False
        
    def monitor_resources(self):
        """Monitor system resource usage."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage=psutil.virtual_memory().percent,
            memory_available=psutil.virtual_memory().available,
            response_time=0.0,  # Would be measured from actual requests
            throughput=0.0,     # Would be measured from actual requests
            error_rate=0.0      # Would be measured from actual requests
        )
        
        self.resource_history.append(metrics)
        return metrics
    
    def adapt_resources(self) -> Dict[str, Any]:
        """Adapt resource allocation based on current usage patterns."""
        if len(self.resource_history) < 10:
            return {}
        
        recent_metrics = list(self.resource_history)[-10:]
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        adaptations = {}
        
        # CPU-based adaptations
        if avg_cpu > 80:
            adaptations["scale_up_cpu"] = True
            adaptations["recommended_cpu_cores"] = min(16, psutil.cpu_count() * 2)
        elif avg_cpu < 30:
            adaptations["scale_down_cpu"] = True
            adaptations["recommended_cpu_cores"] = max(1, psutil.cpu_count() // 2)
        
        # Memory-based adaptations
        if avg_memory > 85:
            adaptations["scale_up_memory"] = True
            adaptations["recommended_memory_gb"] = 16
        elif avg_memory < 40:
            adaptations["optimize_memory"] = True
            adaptations["enable_memory_compression"] = True
        
        return adaptations

class WorkloadPredictiveScaler:
    """Predicts workload patterns and preemptively scales resources."""
    
    def __init__(self):
        self.workload_history: deque = deque(maxlen=2000)
        self.prediction_models = {}
        self.scaling_decisions = []
        
    def record_workload(self, workload_metrics: Dict[str, float]):
        """Record workload metrics for pattern learning."""
        workload_data = {
            "timestamp": datetime.now(),
            "metrics": workload_metrics.copy()
        }
        
        self.workload_history.append(workload_data)
    
    def predict_workload(self, horizon_minutes: int = 60) -> Dict[str, float]:
        """Predict workload for the next time horizon."""
        if len(self.workload_history) < 100:
            # Insufficient data for prediction
            return {"confidence": 0.0}
        
        # Simple time-series prediction (in real implementation, use proper forecasting)
        recent_data = list(self.workload_history)[-100:]
        
        # Extract time patterns (hourly, daily, weekly)
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Find similar time periods
        similar_periods = []
        for data in recent_data:
            data_hour = data["timestamp"].hour
            data_day = data["timestamp"].weekday()
            
            if abs(data_hour - current_hour) <= 1 and data_day == current_day:
                similar_periods.append(data["metrics"])
        
        if similar_periods:
            # Average similar periods
            prediction = {}
            for metric_name in similar_periods[0].keys():
                values = [period[metric_name] for period in similar_periods]
                prediction[metric_name] = sum(values) / len(values)
            
            prediction["confidence"] = min(1.0, len(similar_periods) / 10.0)
            return prediction
        
        return {"confidence": 0.0}
    
    def recommend_scaling(self, predicted_workload: Dict[str, float]) -> Dict[str, Any]:
        """Recommend scaling actions based on predicted workload."""
        if predicted_workload.get("confidence", 0) < 0.5:
            return {"action": "no_scaling", "reason": "low_prediction_confidence"}
        
        recommendations = {}
        
        # Predict resource needs
        predicted_requests = predicted_workload.get("requests_per_second", 0)
        predicted_cpu = predicted_workload.get("cpu_usage", 0.5)
        
        if predicted_requests > 100:  # High load expected
            recommendations["scale_up"] = True
            recommendations["target_replicas"] = min(10, max(3, int(predicted_requests / 50)))
        elif predicted_requests < 20:  # Low load expected
            recommendations["scale_down"] = True
            recommendations["target_replicas"] = max(1, int(predicted_requests / 10))
        
        recommendations["predicted_load"] = predicted_workload
        recommendations["confidence"] = predicted_workload["confidence"]
        
        return recommendations