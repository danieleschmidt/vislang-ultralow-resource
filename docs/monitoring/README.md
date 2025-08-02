# Monitoring and Observability

This document outlines the monitoring and observability strategy for VisLang-UltraLow-Resource.

## Overview

The monitoring stack provides comprehensive observability across all components:

- **Metrics Collection**: Prometheus for application and system metrics
- **Visualization**: Grafana dashboards for real-time monitoring
- **Logging**: Structured logging with correlation IDs
- **Health Checks**: Application and dependency health monitoring
- **Alerting**: Proactive alerts for critical issues
- **Tracing**: Distributed tracing for request flow analysis

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Application   │───▶│  Prometheus  │───▶│   Grafana   │
│                 │    │              │    │             │
│ - Metrics       │    │ - Collection │    │ - Dashboards│
│ - Health checks │    │ - Storage    │    │ - Alerts    │
│ - Logs          │    │ - Querying   │    │ - Analysis  │
└─────────────────┘    └──────────────┘    └─────────────┘
        │
        ▼
┌─────────────────┐
│   Log Storage   │
│                 │
│ - Structured    │
│ - Searchable    │
│ - Retention     │
└─────────────────┘
```

## Metrics

### Application Metrics

**Request Metrics**
- `http_requests_total{method, status_code, endpoint}`: Total HTTP requests
- `http_request_duration_seconds{method, endpoint}`: Request latency histogram
- `http_requests_in_flight`: Active requests gauge

**Processing Metrics**
- `dataset_processing_total{source, language, status}`: Dataset processing operations
- `dataset_processing_duration_seconds{source, language}`: Processing time
- `ocr_operations_total{engine, language, status}`: OCR operations
- `model_inference_duration_seconds{model_name}`: Model inference time
- `model_loading_duration_seconds{model_name}`: Model loading time

**Business Metrics**
- `documents_processed_total{source, language}`: Total documents processed
- `annotations_created_total{language, type}`: Generated annotations
- `training_examples_total{language, quality}`: Training examples created
- `data_quality_score{dataset, language}`: Data quality metrics

**System Metrics**
- `memory_usage_bytes{component}`: Memory consumption
- `gpu_utilization_percent{device_id}`: GPU utilization
- `disk_usage_bytes{mount_point}`: Disk space usage
- `database_connections{state}`: Database connection pool
- `cache_operations_total{operation, result}`: Cache hit/miss rates

### Infrastructure Metrics

**PostgreSQL**
- Connection pool status
- Query performance
- Transaction rates
- Lock waits

**Redis**
- Memory usage
- Hit/miss ratios
- Command latency
- Connected clients

**Storage (MinIO/S3)**
- Request rates
- Transfer bytes
- Error rates
- Bucket sizes

## Health Checks

### Application Health Endpoints

```python
# /health - Basic health check
{
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "1.0.0",
    "environment": "production"
}

# /health/detailed - Comprehensive health check
{
    "status": "healthy",
    "checks": {
        "database": {
            "status": "healthy",
            "response_time_ms": 12,
            "connection_pool": {
                "active": 3,
                "idle": 7,
                "total": 10
            }
        },
        "redis": {
            "status": "healthy",
            "response_time_ms": 2,
            "memory_usage_mb": 45
        },
        "models": {
            "ocr_engine": {
                "status": "healthy",
                "loaded_languages": ["en", "sw", "am", "ha"]
            },
            "vision_language_model": {
                "status": "healthy",
                "model_name": "facebook/mblip-mt0-xl",
                "loaded": true
            }
        },
        "storage": {
            "status": "healthy",
            "available_space_gb": 2048,
            "accessible_buckets": ["datasets", "models", "cache"]
        }
    }
}
```

### Health Check Implementation

```python
from typing import Dict, Any
from datetime import datetime
import asyncio
import psutil

class HealthChecker:
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        start_time = datetime.now()
        try:
            # Simple query to check connectivity
            result = await database.execute("SELECT 1")
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            pool_status = database.get_pool_status()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "connection_pool": pool_status
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": None
            }
    
    async def check_models(self) -> Dict[str, Any]:
        """Check model loading status and capabilities."""
        checks = {}
        
        # Check OCR engine
        try:
            ocr_status = await ocr_service.health_check()
            checks["ocr_engine"] = {
                "status": "healthy",
                "loaded_languages": ocr_status.get("languages", [])
            }
        except Exception as e:
            checks["ocr_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check vision-language model
        try:
            model_status = await vision_model.health_check()
            checks["vision_language_model"] = {
                "status": "healthy",
                "model_name": model_status.get("model_name"),
                "loaded": model_status.get("loaded", False)
            }
        except Exception as e:
            checks["vision_language_model"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return checks
```

## Logging

### Structured Logging Configuration

```python
import structlog
from structlog.stdlib import LoggerFactory

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage examples
logger.info(
    "Dataset processing started",
    source="unhcr",
    language="swahili",
    document_count=150,
    correlation_id=request.correlation_id
)

logger.error(
    "OCR processing failed",
    document_id="doc_123",
    error_type="language_detection_failed",
    retry_count=3,
    correlation_id=request.correlation_id
)
```

### Log Levels and Categories

**ERROR**: System errors, failed operations, exceptions
**WARN**: Degraded performance, retry attempts, deprecated usage
**INFO**: Normal operations, processing milestones, configuration changes
**DEBUG**: Detailed processing steps, parameter values, intermediate results

**Categories**:
- `api`: HTTP request/response logging
- `processing`: Dataset and document processing
- `ml`: Model training and inference
- `storage`: File and database operations
- `auth`: Authentication and authorization
- `monitoring`: Health checks and metrics

## Prometheus Configuration

### Metrics Collection Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'vislang-app'
    static_configs:
      - targets: ['vislang-app:9091']
    metrics_path: /metrics
    scrape_interval: 15s
    scrape_timeout: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Custom Metrics Implementation

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Processing metrics
DATASET_PROCESSING = Counter(
    'dataset_processing_total',
    'Total dataset processing operations',
    ['source', 'language', 'status']
)

OCR_OPERATIONS = Counter(
    'ocr_operations_total',
    'Total OCR operations',
    ['engine', 'language', 'status']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name']
)

# System metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id']
)

# Usage example
@REQUEST_LATENCY.time()
@REQUEST_COUNT.labels(method='POST', endpoint='/api/v1/process')
async def process_document(request):
    try:
        result = await process_document_impl(request)
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/api/v1/process',
            status_code='200'
        ).inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/api/v1/process',
            status_code='500'
        ).inc()
        raise
```

## Alerting Rules

### Critical Alerts

```yaml
# monitoring/alert_rules.yml
groups:
  - name: vislang.rules
    rules:
    # Application availability
    - alert: ApplicationDown
      expr: up{job="vislang-app"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "VisLang application is down"
        description: "The VisLang application has been down for more than 1 minute."

    # High error rate
    - alert: HighErrorRate
      expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} errors per second."

    # Database connectivity
    - alert: DatabaseConnectionIssues
      expr: database_connections{state="active"} / database_connections{state="total"} > 0.8
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Database connection pool nearly exhausted"
        description: "Database connection pool is {{ $value }}% full."

    # Model inference latency
    - alert: SlowModelInference
      expr: histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 30
      for: 3m
      labels:
        severity: warning
      annotations:
        summary: "Model inference is slow"
        description: "95th percentile inference time is {{ $value }} seconds."

    # Storage space
    - alert: LowDiskSpace
      expr: (disk_usage_bytes{mount_point="/data"} / disk_total_bytes{mount_point="/data"}) > 0.85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Low disk space"
        description: "Disk usage is {{ $value }}% on /data mount point."

    # Memory usage
    - alert: HighMemoryUsage
      expr: memory_usage_bytes{component="application"} / memory_total_bytes > 0.9
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High memory usage"
        description: "Memory usage is {{ $value }}% for application component."
```

## Grafana Dashboards

### Application Overview Dashboard

Key panels:
- Request rate and latency
- Error rate by endpoint
- Active users and sessions
- Database query performance
- Model inference metrics
- System resource utilization

### Processing Pipeline Dashboard

Key panels:
- Documents processed per hour
- Processing success/failure rates
- OCR accuracy by language
- Data quality scores
- Training example generation rate
- Storage utilization

### Infrastructure Dashboard

Key panels:
- CPU and memory utilization
- Database performance metrics
- Cache hit/miss ratios
- Network I/O
- GPU utilization (if applicable)
- Container restart rates

## Performance Monitoring

### SLI/SLO Definitions

**Service Level Indicators (SLIs)**:
- API availability: % of successful requests
- API latency: 95th percentile response time
- Processing throughput: Documents processed per hour
- Data quality: % of high-quality training examples

**Service Level Objectives (SLOs)**:
- API availability: 99.5% uptime
- API latency: 95th percentile < 2 seconds
- Processing success rate: 95% of documents processed successfully
- Data quality: 90% of examples meet quality threshold

### Monitoring Runbooks

Located in `docs/runbooks/` with procedures for:
- High error rate investigation
- Performance degradation analysis
- Database connectivity issues
- Model inference problems
- Storage capacity management
- Alert triage and escalation

## Implementation Checklist

- [ ] Configure Prometheus metrics collection
- [ ] Set up Grafana dashboards
- [ ] Implement structured logging
- [ ] Add health check endpoints
- [ ] Define alerting rules
- [ ] Create monitoring runbooks
- [ ] Set up log aggregation
- [ ] Configure metric retention policies
- [ ] Test alert notifications
- [ ] Document monitoring procedures