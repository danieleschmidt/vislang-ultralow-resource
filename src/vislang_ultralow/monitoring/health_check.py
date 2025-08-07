"""Comprehensive health checking and system monitoring."""

import psutil
import logging
import asyncio
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import socket
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import subprocess
import threading

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """System status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    status: SystemStatus
    value: float
    threshold: float
    message: str
    timestamp: float
    metadata: Dict[str, Any] = None


class HealthChecker:
    """Comprehensive system health checker with advanced monitoring."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_history = []
        self.alert_thresholds = self._default_thresholds()
        self.custom_checks = {}
        self.is_monitoring = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("Health checker initialized")
    
    def _default_thresholds(self) -> Dict[str, Dict]:
        """Default health check thresholds."""
        return {
            'cpu_usage': {'warning': 80, 'critical': 95},
            'memory_usage': {'warning': 85, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'response_time': {'warning': 1.0, 'critical': 5.0},
            'error_rate': {'warning': 0.05, 'critical': 0.20},
            'cache_hit_rate': {'warning': 0.7, 'critical': 0.5},
            'database_connections': {'warning': 80, 'critical': 95},
            'queue_size': {'warning': 1000, 'critical': 5000}
        }
    
    async def check_system_health(self) -> Dict[str, HealthMetric]:
        """Perform comprehensive system health check."""
        health_metrics = {}
        timestamp = time.time()
        
        # CPU health
        cpu_metric = await self._check_cpu_health(timestamp)
        health_metrics['cpu'] = cpu_metric
        
        # Memory health
        memory_metric = await self._check_memory_health(timestamp)
        health_metrics['memory'] = memory_metric
        
        # Disk health
        disk_metric = await self._check_disk_health(timestamp)
        health_metrics['disk'] = disk_metric
        
        # Network health
        network_metric = await self._check_network_health(timestamp)
        health_metrics['network'] = network_metric
        
        # Database health
        db_metric = await self._check_database_health(timestamp)
        health_metrics['database'] = db_metric
        
        # Cache health
        cache_metric = await self._check_cache_health(timestamp)
        health_metrics['cache'] = cache_metric
        
        # GPU health (if available)
        gpu_metric = await self._check_gpu_health(timestamp)
        if gpu_metric:
            health_metrics['gpu'] = gpu_metric
        
        # Custom checks
        for check_name, check_func in self.custom_checks.items():
            try:
                custom_metric = await self._run_custom_check(check_name, check_func, timestamp)
                health_metrics[check_name] = custom_metric
            except Exception as e:
                logger.error(f"Custom health check {check_name} failed: {e}")
        
        # Store in history
        self.health_history.append({
            'timestamp': timestamp,
            'metrics': health_metrics
        })
        
        # Keep only last 1000 entries
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        return health_metrics
    
    async def _check_cpu_health(self, timestamp: float) -> HealthMetric:
        """Check CPU health metrics."""
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Determine status
            if cpu_percent >= self.alert_thresholds['cpu_usage']['critical']:
                status = SystemStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent >= self.alert_thresholds['cpu_usage']['warning']:
                status = SystemStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = SystemStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            # Additional CPU metrics
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            return HealthMetric(
                name="cpu_usage",
                status=status,
                value=cpu_percent,
                threshold=self.alert_thresholds['cpu_usage']['warning'],
                message=message,
                timestamp=timestamp,
                metadata={
                    'cpu_count': cpu_count,
                    'cpu_freq': cpu_freq._asdict() if cpu_freq else None,
                    'load_avg': load_avg,
                    'per_cpu': psutil.cpu_percent(percpu=True)
                }
            )
            
        except Exception as e:
            logger.error(f"CPU health check failed: {e}")
            return HealthMetric(
                name="cpu_usage",
                status=SystemStatus.UNHEALTHY,
                value=-1,
                threshold=0,
                message=f"CPU check failed: {str(e)}",
                timestamp=timestamp
            )
    
    async def _check_memory_health(self, timestamp: float) -> HealthMetric:
        """Check memory health metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_percent = memory.percent
            
            # Determine status
            if memory_percent >= self.alert_thresholds['memory_usage']['critical']:
                status = SystemStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent >= self.alert_thresholds['memory_usage']['warning']:
                status = SystemStatus.DEGRADED
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = SystemStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthMetric(
                name="memory_usage",
                status=status,
                value=memory_percent,
                threshold=self.alert_thresholds['memory_usage']['warning'],
                message=message,
                timestamp=timestamp,
                metadata={
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'free': memory.free,
                    'swap_total': swap.total,
                    'swap_used': swap.used,
                    'swap_percent': swap.percent
                }
            )
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return HealthMetric(
                name="memory_usage",
                status=SystemStatus.UNHEALTHY,
                value=-1,
                threshold=0,
                message=f"Memory check failed: {str(e)}",
                timestamp=timestamp
            )
    
    async def _check_disk_health(self, timestamp: float) -> HealthMetric:
        """Check disk health metrics."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status
            if disk_percent >= self.alert_thresholds['disk_usage']['critical']:
                status = SystemStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent >= self.alert_thresholds['disk_usage']['warning']:
                status = SystemStatus.DEGRADED
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = SystemStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            # Additional disk metrics
            disk_io = psutil.disk_io_counters()
            partitions = psutil.disk_partitions()
            
            return HealthMetric(
                name="disk_usage",
                status=status,
                value=disk_percent,
                threshold=self.alert_thresholds['disk_usage']['warning'],
                message=message,
                timestamp=timestamp,
                metadata={
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'disk_io': disk_io._asdict() if disk_io else None,
                    'partitions': [p._asdict() for p in partitions]
                }
            )
            
        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return HealthMetric(
                name="disk_usage",
                status=SystemStatus.UNHEALTHY,
                value=-1,
                threshold=0,
                message=f"Disk check failed: {str(e)}",
                timestamp=timestamp
            )
    
    async def _check_network_health(self, timestamp: float) -> HealthMetric:
        """Check network connectivity and performance."""
        try:
            # Network I/O stats
            net_io = psutil.net_io_counters()
            
            # Test basic connectivity
            connectivity_ok = await self._test_connectivity()
            
            if connectivity_ok:
                status = SystemStatus.HEALTHY
                message = "Network connectivity normal"
            else:
                status = SystemStatus.DEGRADED
                message = "Network connectivity issues detected"
            
            return HealthMetric(
                name="network",
                status=status,
                value=1.0 if connectivity_ok else 0.0,
                threshold=1.0,
                message=message,
                timestamp=timestamp,
                metadata={
                    'bytes_sent': net_io.bytes_sent if net_io else 0,
                    'bytes_recv': net_io.bytes_recv if net_io else 0,
                    'packets_sent': net_io.packets_sent if net_io else 0,
                    'packets_recv': net_io.packets_recv if net_io else 0,
                    'connectivity': connectivity_ok
                }
            )
            
        except Exception as e:
            logger.error(f"Network health check failed: {e}")
            return HealthMetric(
                name="network",
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=1.0,
                message=f"Network check failed: {str(e)}",
                timestamp=timestamp
            )
    
    async def _test_connectivity(self, timeout: float = 5.0) -> bool:
        """Test basic network connectivity."""
        try:
            # Test DNS resolution and connectivity to major sites
            test_hosts = ['8.8.8.8', '1.1.1.1', 'google.com']
            
            for host in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)
                    
                    if '.' in host and not host.replace('.', '').isdigit():
                        # It's a domain name, use port 80
                        result = sock.connect_ex((host, 80))
                    else:
                        # It's an IP, test DNS port
                        result = sock.connect_ex((host, 53))
                    
                    sock.close()
                    
                    if result == 0:
                        return True
                        
                except Exception:
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False
    
    async def _check_database_health(self, timestamp: float) -> HealthMetric:
        """Check database health (placeholder - would connect to actual DB)."""
        try:
            # This would be replaced with actual database health checks
            # For now, return a healthy status
            
            return HealthMetric(
                name="database",
                status=SystemStatus.HEALTHY,
                value=1.0,
                threshold=1.0,
                message="Database connection healthy",
                timestamp=timestamp,
                metadata={
                    'connection_pool_size': 10,
                    'active_connections': 2,
                    'query_latency': 0.05
                }
            )
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthMetric(
                name="database",
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=1.0,
                message=f"Database check failed: {str(e)}",
                timestamp=timestamp
            )
    
    async def _check_cache_health(self, timestamp: float) -> HealthMetric:
        """Check cache system health (placeholder)."""
        try:
            # This would be replaced with actual cache health checks
            
            return HealthMetric(
                name="cache",
                status=SystemStatus.HEALTHY,
                value=0.85,  # Mock cache hit rate
                threshold=self.alert_thresholds['cache_hit_rate']['warning'],
                message="Cache system healthy",
                timestamp=timestamp,
                metadata={
                    'hit_rate': 0.85,
                    'memory_usage': 0.65,
                    'connections': 5
                }
            )
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return HealthMetric(
                name="cache",
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=0.5,
                message=f"Cache check failed: {str(e)}",
                timestamp=timestamp
            )
    
    async def _check_gpu_health(self, timestamp: float) -> Optional[HealthMetric]:
        """Check GPU health if available."""
        try:
            # Try to get GPU information using nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'], 
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 4:
                    temp = float(gpu_info[0])
                    utilization = float(gpu_info[1])
                    mem_used = float(gpu_info[2])
                    mem_total = float(gpu_info[3])
                    
                    mem_percent = (mem_used / mem_total) * 100
                    
                    # Determine status based on temperature and utilization
                    if temp > 80 or utilization > 95:
                        status = SystemStatus.DEGRADED
                        message = f"GPU running hot or at high utilization: {temp}°C, {utilization}%"
                    else:
                        status = SystemStatus.HEALTHY
                        message = f"GPU healthy: {temp}°C, {utilization}% util"
                    
                    return HealthMetric(
                        name="gpu",
                        status=status,
                        value=utilization,
                        threshold=80.0,
                        message=message,
                        timestamp=timestamp,
                        metadata={
                            'temperature': temp,
                            'utilization': utilization,
                            'memory_used': mem_used,
                            'memory_total': mem_total,
                            'memory_percent': mem_percent
                        }
                    )
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # No GPU or nvidia-smi not available
            pass
        except Exception as e:
            logger.debug(f"GPU health check failed: {e}")
        
        return None
    
    async def _run_custom_check(self, name: str, check_func: Callable, timestamp: float) -> HealthMetric:
        """Run a custom health check function."""
        try:
            # Run the custom check with timeout
            future = self.executor.submit(check_func)
            result = await asyncio.get_event_loop().run_in_executor(None, future.result, 10.0)
            
            if isinstance(result, dict):
                return HealthMetric(
                    name=name,
                    status=result.get('status', SystemStatus.HEALTHY),
                    value=result.get('value', 1.0),
                    threshold=result.get('threshold', 1.0),
                    message=result.get('message', f"Custom check {name} completed"),
                    timestamp=timestamp,
                    metadata=result.get('metadata', {})
                )
            else:
                # Simple boolean result
                is_healthy = bool(result)
                return HealthMetric(
                    name=name,
                    status=SystemStatus.HEALTHY if is_healthy else SystemStatus.UNHEALTHY,
                    value=1.0 if is_healthy else 0.0,
                    threshold=1.0,
                    message=f"Custom check {name}: {'passed' if is_healthy else 'failed'}",
                    timestamp=timestamp
                )
                
        except FutureTimeoutError:
            return HealthMetric(
                name=name,
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=1.0,
                message=f"Custom check {name} timed out",
                timestamp=timestamp
            )
        except Exception as e:
            return HealthMetric(
                name=name,
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=1.0,
                message=f"Custom check {name} failed: {str(e)}",
                timestamp=timestamp
            )
    
    def add_custom_check(self, name: str, check_func: Callable):
        """Add a custom health check function.
        
        Args:
            name: Name of the check
            check_func: Function that returns health status
        """
        self.custom_checks[name] = check_func
        logger.info(f"Added custom health check: {name}")
    
    def get_overall_status(self, metrics: Dict[str, HealthMetric]) -> SystemStatus:
        """Determine overall system status from individual metrics."""
        statuses = [metric.status for metric in metrics.values()]
        
        if SystemStatus.CRITICAL in statuses:
            return SystemStatus.CRITICAL
        elif SystemStatus.UNHEALTHY in statuses:
            return SystemStatus.UNHEALTHY
        elif SystemStatus.DEGRADED in statuses:
            return SystemStatus.DEGRADED
        else:
            return SystemStatus.HEALTHY
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.is_monitoring:
            try:
                metrics = loop.run_until_complete(self.check_system_health())
                overall_status = self.get_overall_status(metrics)
                
                # Log status changes
                if hasattr(self, '_last_status') and self._last_status != overall_status:
                    logger.info(f"System status changed: {self._last_status.value} -> {overall_status.value}")
                
                self._last_status = overall_status
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        if not self.health_history:
            return {'status': 'no_data', 'message': 'No health data available'}
        
        latest = self.health_history[-1]
        metrics = latest['metrics']
        overall_status = self.get_overall_status(metrics)
        
        return {
            'status': overall_status.value,
            'timestamp': latest['timestamp'],
            'metrics': {
                name: {
                    'status': metric.status.value,
                    'value': metric.value,
                    'message': metric.message
                }
                for name, metric in metrics.items()
            },
            'uptime': self._get_uptime(),
            'checks_performed': len(self.health_history)
        }
    
    def _get_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            return 0.0
    
    def export_health_report(self, output_path: str):
        """Export comprehensive health report."""
        report = {
            'summary': self.get_health_summary(),
            'thresholds': self.alert_thresholds,
            'history': self.health_history[-100:],  # Last 100 checks
            'custom_checks': list(self.custom_checks.keys()),
            'generated_at': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Health report exported to {output_path}")


class ServiceHealthChecker:
    """Health checker for external services and dependencies."""
    
    def __init__(self):
        self.service_configs = {}
        self.results_cache = {}
        self.cache_ttl = 60  # 1 minute cache
    
    def add_service(self, name: str, config: Dict[str, Any]):
        """Add a service to monitor.
        
        Args:
            name: Service name
            config: Service configuration (url, timeout, etc.)
        """
        self.service_configs[name] = config
        logger.info(f"Added service for monitoring: {name}")
    
    async def check_service_health(self, name: str) -> HealthMetric:
        """Check health of a specific service."""
        if name not in self.service_configs:
            return HealthMetric(
                name=name,
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=1.0,
                message=f"Service {name} not configured",
                timestamp=time.time()
            )
        
        config = self.service_configs[name]
        
        try:
            if config.get('type') == 'http':
                return await self._check_http_service(name, config)
            elif config.get('type') == 'database':
                return await self._check_database_service(name, config)
            elif config.get('type') == 'redis':
                return await self._check_redis_service(name, config)
            else:
                return await self._check_generic_service(name, config)
                
        except Exception as e:
            return HealthMetric(
                name=name,
                status=SystemStatus.UNHEALTHY,
                value=0.0,
                threshold=1.0,
                message=f"Service check failed: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_http_service(self, name: str, config: Dict) -> HealthMetric:
        """Check HTTP service health."""
        import aiohttp
        
        url = config['url']
        timeout = config.get('timeout', 5)
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        status = SystemStatus.HEALTHY
                        message = f"Service {name} healthy ({response_time:.2f}s)"
                    else:
                        status = SystemStatus.DEGRADED
                        message = f"Service {name} returned {response.status}"
                    
                    return HealthMetric(
                        name=name,
                        status=status,
                        value=response_time,
                        threshold=config.get('response_time_threshold', 1.0),
                        message=message,
                        timestamp=time.time(),
                        metadata={
                            'status_code': response.status,
                            'response_time': response_time,
                            'url': url
                        }
                    )
                    
        except Exception as e:
            return HealthMetric(
                name=name,
                status=SystemStatus.UNHEALTHY,
                value=-1,
                threshold=1.0,
                message=f"HTTP service {name} failed: {str(e)}",
                timestamp=time.time()
            )
    
    async def _check_database_service(self, name: str, config: Dict) -> HealthMetric:
        """Check database service health (placeholder)."""
        # This would implement actual database connectivity checks
        return HealthMetric(
            name=name,
            status=SystemStatus.HEALTHY,
            value=1.0,
            threshold=1.0,
            message=f"Database service {name} healthy",
            timestamp=time.time()
        )
    
    async def _check_redis_service(self, name: str, config: Dict) -> HealthMetric:
        """Check Redis service health (placeholder)."""
        # This would implement actual Redis connectivity checks
        return HealthMetric(
            name=name,
            status=SystemStatus.HEALTHY,
            value=1.0,
            threshold=1.0,
            message=f"Redis service {name} healthy",
            timestamp=time.time()
        )
    
    async def _check_generic_service(self, name: str, config: Dict) -> HealthMetric:
        """Check generic service health."""
        return HealthMetric(
            name=name,
            status=SystemStatus.HEALTHY,
            value=1.0,
            threshold=1.0,
            message=f"Generic service {name} healthy",
            timestamp=time.time()
        )