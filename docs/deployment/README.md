# Deployment Guide

This guide covers deployment strategies for VisLang-UltraLow-Resource in various environments.

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

### Single Container Deployment

```bash
# Build production image
docker build --target production -t vislang-ultralow:latest .

# Run with minimal configuration
docker run -d \
  --name vislang-app \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  vislang-ultralow:latest
```

## Environment Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/vislang_db

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Hugging Face Hub
HUGGINGFACE_HUB_TOKEN=your_token_here
HUGGINGFACE_HUB_CACHE=/app/cache/huggingface

# Object Storage (MinIO/S3)
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin123
S3_BUCKET_NAME=vislang-datasets

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=info

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

### Optional Configuration

```bash
# Monitoring
PROMETHEUS_METRICS_ENABLED=true
PROMETHEUS_METRICS_PORT=9091

# OCR Configuration
TESSERACT_LANGUAGES=eng,ara,amh,swa,fra,spa
EASYOCR_GPU=true

# Training Configuration
TORCH_DEVICE=cuda
MIXED_PRECISION=true
GRADIENT_CHECKPOINTING=true
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl and helm
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Deployment Steps

```bash
# Create namespace
kubectl create namespace vislang

# Deploy with Helm (chart not included in this repo)
helm install vislang ./charts/vislang \
  --namespace vislang \
  --values values.prod.yaml

# Monitor deployment
kubectl get pods -n vislang -w
```

## Cloud Platform Deployment

### AWS ECS with Fargate

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker build -t vislang-ultralow .
docker tag vislang-ultralow:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/vislang-ultralow:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/vislang-ultralow:latest

# Deploy with ECS CLI or Terraform
ecs-cli compose --project-name vislang service up --cluster-config vislang-cluster
```

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/vislang-ultralow .
gcloud run deploy vislang-app \
  --image gcr.io/PROJECT-ID/vislang-ultralow \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 80 \
  --timeout 900
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry myregistry --image vislang-ultralow .

# Deploy container group
az container create \
  --resource-group myResourceGroup \
  --name vislang-app \
  --image myregistry.azurecr.io/vislang-ultralow:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables DATABASE_URL=... REDIS_URL=...
```

## Production Considerations

### Security

- Use secrets management (AWS Secrets Manager, Azure Key Vault, K8s Secrets)
- Enable HTTPS/TLS with proper certificates
- Implement proper authentication and authorization
- Regular security updates and vulnerability scanning
- Network segmentation and firewall rules

### Monitoring and Observability

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vislang-app'
    static_configs:
      - targets: ['vislang-app:9091']
```

### Backup and Disaster Recovery

```bash
# Database backups
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Model and dataset backups
aws s3 sync s3://vislang-datasets s3://vislang-backups/datasets/
aws s3 sync s3://vislang-models s3://vislang-backups/models/
```

### Scaling Strategies

1. **Horizontal Pod Autoscaling (HPA)**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: vislang-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: vislang-app
     minReplicas: 2
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

2. **Vertical Pod Autoscaling (VPA)**
3. **Cluster Autoscaling** for GPU nodes

### Performance Optimization

- Use GPU instances for model training and inference
- Implement model caching and quantization
- Configure connection pooling for databases
- Use CDN for static assets and model artifacts
- Implement request queuing for batch processing

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Increase memory limits
   - Enable gradient checkpointing
   - Reduce batch size

2. **GPU Not Available**
   - Verify CUDA installation
   - Check GPU node selectors in K8s
   - Ensure proper device requests

3. **Database Connection Issues**
   - Check connection string format
   - Verify network connectivity
   - Review connection pool settings

### Health Checks

```bash
# Application health
curl -f http://localhost:8000/health

# Database connectivity
curl -f http://localhost:8000/health/db

# Model loading status
curl -f http://localhost:8000/health/models
```

### Logs and Debugging

```bash
# Container logs
docker logs vislang-app --tail 100 -f

# Kubernetes logs
kubectl logs -f deployment/vislang-app -n vislang

# Application metrics
curl http://localhost:9091/metrics
```