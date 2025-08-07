# VisLang UltraLow Resource - Production Deployment Guide

This guide covers the complete production deployment of the VisLang UltraLow Resource system, a research-grade framework for vision-language models in humanitarian contexts.

## 🎯 System Overview

The VisLang UltraLow Resource system provides:
- **Novel OCR Algorithms**: Adaptive multi-engine OCR with consensus mechanisms
- **Cross-Lingual Alignment**: Zero-shot cross-lingual text alignment using geometric methods
- **Humanitarian Scene Understanding**: Specialized computer vision for crisis scenarios  
- **Production-Ready API**: FastAPI-based inference service with monitoring
- **Research Infrastructure**: Comprehensive benchmarking and statistical validation

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 100GB+ SSD for models and data
- **GPU**: Optional but recommended for model inference (CUDA-compatible)

### Software Dependencies
- Docker 20.10+
- Kubernetes 1.24+ (for production scaling)
- PostgreSQL 13+
- Redis 6+
- Python 3.8+ (for development)

## 🐳 Docker Deployment

### 1. Build Production Image

```bash
# Build the production Docker image
docker build -f deployment/docker/Dockerfile.production -t vislang-ultralow:latest .

# Verify the build
docker run --rm vislang-ultralow:latest python -c "import vislang_ultralow; print('✅ Build successful')"
```

### 2. Environment Configuration

Create `.env.prod`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/vislang_prod
POSTGRES_USER=vislang_user
POSTGRES_PASSWORD=secure_password_here
POSTGRES_DB=vislang_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
LOG_LEVEL=INFO
WORKERS=4
MAX_UPLOAD_SIZE=100MB

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
PROMETHEUS_PORT=9090

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=api.yourdomain.com,localhost

# Model Storage
MODEL_STORAGE_PATH=/app/models
CACHE_STORAGE_PATH=/app/cache

# OCR Configuration
TESSERACT_CONFIG=--oem 3 --psm 6
ENABLE_GPU_INFERENCE=false
```

### 3. Run with Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f vislang-api
```

## ☸️ Kubernetes Deployment

### 1. Create Namespace and Secrets

```bash
# Create production namespace
kubectl create namespace vislang-prod

# Create secrets
kubectl create secret generic vislang-secrets \
  --from-literal=database-url="postgresql://user:pass@db:5432/vislang" \
  --from-literal=redis-url="redis://redis:6379/0" \
  --from-literal=secret-key="your-secret-key" \
  -n vislang-prod
```

### 2. Deploy to Kubernetes

```bash
# Apply all Kubernetes manifests
kubectl apply -f deployment/k8s/

# Verify deployment
kubectl get pods -n vislang-prod

# Check service status
kubectl get svc -n vislang-prod

# View logs
kubectl logs -f deployment/vislang-api -n vislang-prod
```

### 3. Configure Horizontal Pod Autoscaler

The HPA is pre-configured to scale based on CPU/memory usage:
- **Min replicas**: 3
- **Max replicas**: 20  
- **CPU target**: 70%
- **Memory target**: 80%

Monitor scaling:
```bash
kubectl get hpa -n vislang-prod -w
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `WORKERS` | Number of API workers | 4 |
| `MAX_UPLOAD_SIZE` | Maximum file upload size | 100MB |
| `ENABLE_GPU_INFERENCE` | Enable GPU acceleration | false |
| `MODEL_CACHE_SIZE` | Model cache size limit | 2GB |

### Model Configuration

```python
# models/config.yaml
models:
  default:
    name: "facebook/mblip-mt0-xl"
    languages: ["en", "fr", "es", "ar", "sw"]
    max_length: 512
    batch_size: 8
  
  humanitarian:
    name: "custom/humanitarian-vl-model"
    specialized: true
    crisis_detection: true
    
ocr:
  engines: ["tesseract", "easyocr", "paddleocr"]
  confidence_threshold: 0.7
  consensus_algorithm: "adaptive_weighted_voting"
```

## 📊 Monitoring and Observability

### Health Endpoints

- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive health status
- `GET /metrics` - Prometheus metrics
- `GET /stats` - System statistics

### Monitoring Stack

**Prometheus Configuration**:
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vislang-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

**Grafana Dashboard**: Import `monitoring/grafana/dashboards/application-overview.json`

### Alerts Configuration

```yaml
# monitoring/alert_rules.yml
groups:
  - name: vislang.rules
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
          
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High memory usage
```

## 🔒 Security Considerations

### API Security
- **Authentication**: JWT tokens with refresh mechanism
- **Rate Limiting**: 100 requests/minute per IP
- **Input Validation**: Comprehensive request validation
- **File Upload Security**: Type and size validation
- **CORS**: Configured for production domains

### Infrastructure Security
- **Container Security**: Non-root user, minimal attack surface
- **Network Policies**: Kubernetes network policies applied  
- **Secrets Management**: Kubernetes secrets for sensitive data
- **TLS/SSL**: Let's Encrypt certificates for HTTPS
- **Security Scanning**: Regular vulnerability scans

### Compliance
- **GDPR**: Data processing consent and right to erasure
- **SOC2**: Logging and access controls
- **HIPAA**: Healthcare data handling (if applicable)

## 🚀 Performance Optimization

### Production Tuning

```python
# deployment/config/production.py
PERFORMANCE_CONFIG = {
    "workers": 4,  # Adjust based on CPU cores
    "worker_class": "uvicorn.workers.UvicornWorker",
    "max_requests": 1000,
    "max_requests_jitter": 100,
    "preload_app": True,
    "timeout": 30,
    "keepalive": 2,
}

CACHE_CONFIG = {
    "redis_max_connections": 50,
    "cache_ttl": 3600,
    "model_cache_size": "2GB",
    "result_cache_enabled": True,
}

OCR_CONFIG = {
    "batch_size": 8,
    "parallel_engines": True,
    "adaptive_optimization": True,
    "memory_pooling": True,
}
```

### Database Optimization

```sql
-- PostgreSQL optimization
-- Indexes for common queries
CREATE INDEX CONCURRENTLY idx_documents_source_language ON documents(source, language);
CREATE INDEX CONCURRENTLY idx_training_runs_status ON training_runs(status);
CREATE INDEX CONCURRENTLY idx_dataset_items_quality ON dataset_items(quality_score);

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
```

## 📈 Scaling Guidelines

### Horizontal Scaling
- **API Servers**: Scale based on CPU/memory usage
- **Database**: Read replicas for query scaling  
- **Cache**: Redis cluster for high availability
- **File Storage**: Distributed storage (S3/GCS)

### Vertical Scaling
- **Memory**: Increase for larger models and batch processing
- **CPU**: More cores for parallel OCR processing
- **GPU**: Add GPUs for faster inference

### Load Testing
```bash
# API load testing with artillery
artillery run deployment/load-tests/api-load-test.yml

# OCR performance testing
python scripts/performance_test.py --test-ocr --batch-sizes 1,4,8,16
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: |
          python -m pytest tests/ -v --cov=src --cov-report=xml
          
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and Push Docker Image
        run: |
          docker build -f deployment/docker/Dockerfile.production -t vislang:${{ github.ref_name }} .
          docker push vislang:${{ github.ref_name }}
          
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/vislang-api vislang-api=vislang:${{ github.ref_name }} -n vislang-prod
```

## 📚 API Documentation

### Core Endpoints

**Inference**:
```bash
# Upload image for analysis
curl -X POST "https://api.vislang.terragonlabs.ai/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@humanitarian_image.jpg" \
  -F "instruction=Describe what you see in this humanitarian context"

# Response
{
  "response": "This image shows a refugee camp with temporary shelters...",
  "confidence": 0.87,
  "processing_time_ms": 1250,
  "model_info": {
    "name": "humanitarian-vl-model",
    "version": "v1.0"
  }
}
```

**Model Management**:
```bash
# List available models
curl "https://api.vislang.terragonlabs.ai/models"

# Load specific model
curl -X POST "https://api.vislang.terragonlabs.ai/load_model" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/humanitarian-specialized"}'
```

### Research Endpoints

```bash
# Run benchmark suite
curl -X POST "https://api.vislang.terragonlabs.ai/research/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["base", "humanitarian"],
    "tasks": ["classification", "ocr", "alignment"],
    "datasets": ["humanitarian-eval-v1"]
  }'

# Cross-lingual alignment test
curl -X POST "https://api.vislang.terragonlabs.ai/research/alignment" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "Emergency shelter needed",
    "source_lang": "en",
    "target_lang": "fr"
  }'
```

## 🆘 Troubleshooting

### Common Issues

**High Memory Usage**:
```bash
# Check memory usage
kubectl top pods -n vislang-prod

# Scale down temporarily
kubectl scale deployment vislang-api --replicas=2 -n vislang-prod

# Clear model cache
curl -X POST "https://api.vislang.terragonlabs.ai/admin/clear_cache"
```

**OCR Performance Issues**:
```bash
# Check OCR engine status
curl "https://api.vislang.terragonlabs.ai/health/ocr"

# Restart with different engines
kubectl set env deployment/vislang-api OCR_ENGINES="tesseract,easyocr" -n vislang-prod
```

**Database Connection Issues**:
```bash
# Check database connectivity
kubectl exec -it deployment/vislang-api -- python -c "
from vislang_ultralow.database import get_database_manager
db = get_database_manager()
print(db.health_check())
"

# Check connection pool
kubectl logs -f deployment/vislang-api -n vislang-prod | grep "database"
```

### Log Analysis

```bash
# Search for errors
kubectl logs -f deployment/vislang-api -n vislang-prod | grep ERROR

# Monitor API requests
kubectl logs -f deployment/vislang-api -n vislang-prod | grep "POST\|GET"

# Check performance metrics  
curl -s "https://api.vislang.terragonlabs.ai/metrics" | grep "request_duration"
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review error logs and performance metrics
   - Update model cache and clear old data
   - Check security vulnerability scans

2. **Monthly**:
   - Update dependencies and security patches
   - Review and rotate API keys/secrets  
   - Analyze usage patterns and optimize

3. **Quarterly**:
   - Full system backup and disaster recovery test
   - Performance benchmarking and optimization
   - Security audit and penetration testing

### Contact Information

- **Technical Support**: daniel@terragonlabs.ai
- **Emergency Contact**: +1-XXX-XXX-XXXX
- **Documentation**: https://docs.terragonlabs.ai/vislang
- **Status Page**: https://status.terragonlabs.ai

---

## 🎉 Deployment Complete!

Your VisLang UltraLow Resource system is now deployed and ready for production use. The system provides:

✅ **Research-Grade Performance**: Novel algorithms with statistical validation  
✅ **Production Reliability**: Comprehensive monitoring and error handling  
✅ **Horizontal Scalability**: Kubernetes-native with auto-scaling  
✅ **Security Compliance**: Enterprise-grade security controls  
✅ **Observability**: Full monitoring, logging, and alerting stack  

For additional support or advanced configuration, please refer to the technical documentation or contact our support team.