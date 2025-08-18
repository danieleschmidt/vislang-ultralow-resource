#!/usr/bin/env python3
"""
Comprehensive health check script for VisLang-UltraLow-Resource system.
Checks all system components and reports overall health status.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
import psycopg2
import redis
from pathlib import Path


@dataclass
class HealthCheckResult:
    """Health check result for a component."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    latency_ms: Optional[float] = None
    details: Optional[Dict] = None


class HealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.results: List[HealthCheckResult] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def check_api_health(self) -> HealthCheckResult:
        """Check main API health."""
        component = "api"
        try:
            start_time = time.time()
            url = f"{self.config['api']['base_url']}/health"
            
            response = requests.get(
                url, 
                timeout=self.config['api']['timeout']
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                return HealthCheckResult(
                    component=component,
                    status="healthy",
                    message="API is responding normally",
                    latency_ms=latency_ms,
                    details=health_data
                )
            else:
                return HealthCheckResult(
                    component=component,
                    status="unhealthy",
                    message=f"API returned status {response.status_code}",
                    latency_ms=latency_ms
                )
                
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message="API health check timed out"
            )
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message="Cannot connect to API"
            )
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"API health check failed: {str(e)}"
            )
    
    def check_database_health(self) -> HealthCheckResult:
        """Check PostgreSQL database health."""
        component = "database"
        try:
            start_time = time.time()
            
            conn = psycopg2.connect(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                database=self.config['database']['name'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                connect_timeout=self.config['database']['timeout']
            )
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                
                # Check connection count
                cursor.execute("""
                    SELECT count(*) as active_connections 
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """)
                active_connections = cursor.fetchone()[0]
                
                # Check database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                """)
                db_size = cursor.fetchone()[0]
            
            conn.close()
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=component,
                status="healthy",
                message="Database is accessible and responding",
                latency_ms=latency_ms,
                details={
                    "version": version,
                    "active_connections": active_connections,
                    "database_size": db_size
                }
            )
            
        except psycopg2.OperationalError as e:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"Database connection failed: {str(e)}"
            )
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"Database health check failed: {str(e)}"
            )
    
    def check_redis_health(self) -> HealthCheckResult:
        """Check Redis cache health."""
        component = "redis"
        try:
            start_time = time.time()
            
            r = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                socket_timeout=self.config['redis']['timeout']
            )
            
            # Test connection
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Check memory usage
            memory_usage = (info['used_memory'] / info['maxmemory']) * 100 if info.get('maxmemory', 0) > 0 else 0
            
            status = "healthy"
            if memory_usage > 90:
                status = "degraded"
            
            return HealthCheckResult(
                component=component,
                status=status,
                message="Redis is accessible and responding",
                latency_ms=latency_ms,
                details={
                    "version": info['redis_version'],
                    "connected_clients": info['connected_clients'],
                    "memory_usage_percent": round(memory_usage, 2),
                    "total_commands_processed": info['total_commands_processed'],
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0)
                }
            )
            
        except redis.ConnectionError:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message="Cannot connect to Redis"
            )
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"Redis health check failed: {str(e)}"
            )
    
    def check_storage_health(self) -> HealthCheckResult:
        """Check storage/filesystem health."""
        component = "storage"
        try:
            import shutil
            
            storage_path = Path(self.config['storage']['path'])
            
            if not storage_path.exists():
                return HealthCheckResult(
                    component=component,
                    status="unhealthy",
                    message=f"Storage path does not exist: {storage_path}"
                )
            
            # Check disk usage
            total, used, free = shutil.disk_usage(storage_path)
            usage_percent = (used / total) * 100
            
            status = "healthy"
            if usage_percent > 90:
                status = "unhealthy"
            elif usage_percent > 80:
                status = "degraded"
            
            return HealthCheckResult(
                component=component,
                status=status,
                message="Storage is accessible",
                details={
                    "path": str(storage_path),
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "usage_percent": round(usage_percent, 2)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                message=f"Storage health check failed: {str(e)}"
            )
    
    def check_model_health(self) -> HealthCheckResult:
        """Check if ML models are accessible."""
        component = "models"
        try:
            # Check if model files exist
            model_path = Path(self.config['models']['path'])
            
            if not model_path.exists():
                return HealthCheckResult(
                    component=component,
                    status="degraded",
                    message="Model directory does not exist - models will be downloaded on demand"
                )
            
            # Count available models
            model_files = list(model_path.glob("**/*.bin")) + list(model_path.glob("**/*.safetensors"))
            
            return HealthCheckResult(
                component=component,
                status="healthy",
                message=f"Found {len(model_files)} model files",
                details={
                    "model_path": str(model_path),
                    "model_count": len(model_files)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="degraded",
                message=f"Model health check failed: {str(e)}"
            )
    
    def check_external_services(self) -> HealthCheckResult:
        """Check external service dependencies."""
        component = "external_services"
        try:
            external_checks = []
            
            # Check HuggingFace Hub
            try:
                response = requests.get("https://huggingface.co", timeout=5)
                hf_status = "accessible" if response.status_code == 200 else "degraded"
            except:
                hf_status = "inaccessible"
            
            external_checks.append({"service": "huggingface", "status": hf_status})
            
            # Check UN OCHA API (example humanitarian data source)
            try:
                response = requests.get("https://api.reliefweb.int/v1/reports?limit=1", timeout=5)
                ocha_status = "accessible" if response.status_code == 200 else "degraded"
            except:
                ocha_status = "inaccessible"
            
            external_checks.append({"service": "reliefweb", "status": ocha_status})
            
            # Determine overall external service status
            accessible_count = sum(1 for check in external_checks if check["status"] == "accessible")
            total_count = len(external_checks)
            
            if accessible_count == total_count:
                status = "healthy"
                message = "All external services are accessible"
            elif accessible_count > 0:
                status = "degraded"
                message = f"{accessible_count}/{total_count} external services accessible"
            else:
                status = "unhealthy"
                message = "No external services are accessible"
            
            return HealthCheckResult(
                component=component,
                status=status,
                message=message,
                details={"services": external_checks}
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="degraded",
                message=f"External services check failed: {str(e)}"
            )
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        self.logger.info("Starting comprehensive health check...")
        
        checks = [
            self.check_api_health,
            self.check_database_health,
            self.check_redis_health,
            self.check_storage_health,
            self.check_model_health,
            self.check_external_services
        ]
        
        results = []
        for check_func in checks:
            try:
                result = check_func()
                results.append(result)
                self.logger.info(f"{result.component}: {result.status} - {result.message}")
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                results.append(HealthCheckResult(
                    component="unknown",
                    status="unhealthy",
                    message=f"Health check exception: {str(e)}"
                ))
        
        self.results = results
        return results
    
    def get_overall_status(self) -> Tuple[str, str]:
        """Get overall system health status."""
        if not self.results:
            return "unknown", "No health checks performed"
        
        unhealthy_count = sum(1 for r in self.results if r.status == "unhealthy")
        degraded_count = sum(1 for r in self.results if r.status == "degraded")
        
        if unhealthy_count > 0:
            return "unhealthy", f"{unhealthy_count} components are unhealthy"
        elif degraded_count > 0:
            return "degraded", f"{degraded_count} components are degraded"
        else:
            return "healthy", "All components are healthy"
    
    def generate_report(self, format: str = "json") -> str:
        """Generate health check report."""
        overall_status, overall_message = self.get_overall_status()
        
        report_data = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "overall_message": overall_message,
            "components": []
        }
        
        for result in self.results:
            component_data = {
                "component": result.component,
                "status": result.status,
                "message": result.message
            }
            
            if result.latency_ms is not None:
                component_data["latency_ms"] = result.latency_ms
            
            if result.details is not None:
                component_data["details"] = result.details
            
            report_data["components"].append(component_data)
        
        if format == "json":
            return json.dumps(report_data, indent=2)
        elif format == "summary":
            lines = [
                f"System Health: {overall_status.upper()}",
                f"Message: {overall_message}",
                f"Timestamp: {time.ctime(report_data['timestamp'])}",
                "",
                "Component Status:"
            ]
            
            for result in self.results:
                status_symbol = "✓" if result.status == "healthy" else "⚠" if result.status == "degraded" else "✗"
                lines.append(f"  {status_symbol} {result.component}: {result.status} - {result.message}")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


def load_config(config_path: str) -> Dict:
    """Load configuration from file or use defaults."""
    default_config = {
        "api": {
            "base_url": "http://localhost:8000",
            "timeout": 10
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "vislang_db",
            "user": "vislang",
            "password": "vislang_password",
            "timeout": 5
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "timeout": 5
        },
        "storage": {
            "path": "/app/data"
        },
        "models": {
            "path": "/app/models"
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults
            for key, value in file_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration")
    
    return default_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VisLang Health Check")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--format", choices=["json", "summary"], default="summary",
                       help="Output format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--exit-code", action="store_true",
                       help="Exit with non-zero code if system is not healthy")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run health checks
    checker = HealthChecker(config)
    checker.run_all_checks()
    
    # Generate report
    report = checker.generate_report(args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Health check report written to {args.output}")
    else:
        print(report)
    
    # Exit with appropriate code
    if args.exit_code:
        overall_status, _ = checker.get_overall_status()
        sys.exit(0 if overall_status == "healthy" else 1)


if __name__ == "__main__":
    main()