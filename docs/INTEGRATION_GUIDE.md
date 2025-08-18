# Integration Guide for VisLang-UltraLow-Resource

This guide provides comprehensive instructions for integrating and configuring all project components to work together seamlessly.

## Overview

The VisLang project consists of several integrated components:

1. **Core Application** - Main ML pipeline and processing
2. **Development Environment** - VSCode, linting, testing
3. **Build & Containerization** - Docker, multi-arch builds, SBOM
4. **Monitoring & Observability** - Prometheus, Grafana, Alertmanager
5. **Automation** - Metrics collection, dependency updates, releases
6. **CI/CD Workflows** - GitHub Actions for testing, deployment, security

## Quick Start

### Prerequisites

- Python 3.8+
- Docker Desktop
- Git
- GitHub account with appropriate permissions

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danieleschmidt/vislang-ultralow-resource.git
   cd vislang-ultralow-resource
   ```

2. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -e ".[dev,test]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run initial setup**:
   ```bash
   make setup
   ```

## Component Integration

### 1. Development Environment Integration

The development environment is pre-configured with:

- **VSCode Settings** (`.vscode/settings.json`)
- **Debug Configurations** (`.vscode/launch.json`) 
- **Task Definitions** (`.vscode/tasks.json`)

**Setup:**
```bash
# VSCode will automatically detect the configuration
# Install recommended extensions when prompted
code .
```

**Key Features:**
- Python debugging with breakpoints
- Integrated terminal with proper environment
- Code formatting on save (Black, isort)
- Type checking with mypy
- Testing integration with pytest

### 2. Testing Infrastructure Integration

Comprehensive testing setup with multiple test types:

**Directory Structure:**
```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests  
├── e2e/              # End-to-end tests
├── performance/      # Performance benchmarks
├── fixtures/         # Test data and fixtures
├── configs/          # Test configurations
└── test_utils.py     # Testing utilities
```

**Run Tests:**
```bash
# All tests
make test

# Specific test types
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/performance/    # Performance tests

# With coverage
pytest --cov=src --cov-report=html
```

### 3. Build & Containerization Integration

Multi-stage Docker builds with security scanning and SBOM generation:

**Build Commands:**
```bash
# Local development build
make docker-build

# Production build with multi-arch
make docker-build-prod

# Security scan
make docker-scan

# Generate SBOM
make docker-sbom
```

**Docker Compose for Development:**
```bash
# Start all services
docker-compose -f docker-compose.dev.yml up

# Start specific services
docker-compose -f docker-compose.dev.yml up app database
```

### 4. Monitoring & Observability Integration

Complete monitoring stack with Prometheus, Grafana, and Alertmanager:

**Setup Monitoring Stack:**
```bash
# Start monitoring services
cd monitoring
docker-compose up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
```

**Health Check Integration:**
```bash
# Run health checks
python scripts/monitoring/health-check.py

# Monitor continuously
python scripts/monitoring/health-check.py --continuous --interval 60
```

**Custom Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Application metrics
requests_total = Counter('http_requests_total', 'Total HTTP requests')
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
active_users = Gauge('active_users', 'Active users')
```

### 5. Automation Integration

Automated metrics collection, dependency updates, and release management:

**Metrics Collection:**
```bash
# Manual run
python scripts/metrics_collection.py

# With specific types
python scripts/metrics_collection.py --types repository quality security

# Generate summary report
python scripts/metrics_collection.py --output-summary metrics-summary.md
```

**Dependency Updates:**
```bash
# Check for updates
python scripts/automation/dependency-updater.py

# Create GitHub issue
python scripts/automation/dependency-updater.py --create-issue

# Save analysis
python scripts/automation/dependency-updater.py --json dependency-analysis.json
```

**Release Automation:**
```bash
# Create patch release
python scripts/automation/release-automation.py patch

# Dry run (preview changes)
python scripts/automation/release-automation.py minor --dry-run

# Skip tests (not recommended)
python scripts/automation/release-automation.py major --skip-tests
```

### 6. CI/CD Workflow Integration

GitHub Actions workflows for comprehensive CI/CD:

**Workflow Files:**
- `docs/workflows/examples/ci.yml` - Main CI pipeline
- `docs/workflows/examples/docker-build.yml` - Docker builds
- `docs/workflows/examples/security.yml` - Security scanning
- `docs/workflows/examples/performance.yml` - Performance testing
- `docs/workflows/examples/deploy.yml` - Deployment workflows

**Required GitHub Secrets:**
```bash
# Repository secrets
CODECOV_TOKEN           # Code coverage reporting
PYPI_API_TOKEN         # PyPI package publishing
SNYK_TOKEN             # Security scanning
SLACK_WEBHOOK          # Team notifications

# Environment secrets (per environment)
AWS_ACCESS_KEY_ID      # AWS deployment
AWS_SECRET_ACCESS_KEY  # AWS deployment
KUBECONFIG             # Kubernetes configuration
```

**Branch Protection Setup:**
```bash
# Use GitHub CLI to configure branch protection
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["CI","Security","Tests"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2}'
```

## Configuration Files

### Environment Variables

Create `.env` file with required configuration:

```env
# Application
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/vislang

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ML Configuration
MODEL_CACHE_DIR=./models
MAX_BATCH_SIZE=32
GPU_MEMORY_FRACTION=0.8

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
METRICS_ENABLED=true

# External Services
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200

# Security
CORS_ORIGINS=["http://localhost:3000"]
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
NOTIFICATION_EMAIL=alerts@example.com

# Cloud Storage
AWS_REGION=us-west-2
S3_BUCKET=vislang-data
AZURE_CONTAINER=vislang-container

# Development
DEVELOPMENT_MODE=false
AUTO_RELOAD=false
DEBUG_PROFILING=false
```

### Configuration Validation

Validate your configuration:

```bash
# Validate environment
python scripts/validate-config.py

# Check all integrations
python scripts/integration-test.py

# Verify Docker setup
make docker-verify

# Test monitoring setup
make monitoring-test
```

## Deployment Integration

### Local Development

```bash
# Full local setup
make dev-setup

# Start all services
make dev-start

# Stop services
make dev-stop

# Clean up
make dev-clean
```

### Staging Deployment

```bash
# Deploy to staging
make deploy-staging

# Run smoke tests
make test-staging

# Monitor deployment
make monitor-staging
```

### Production Deployment

```bash
# Deploy to production (requires approvals)
make deploy-production

# Monitor production
make monitor-production

# Emergency rollback if needed
make rollback-production
```

## Troubleshooting

### Common Integration Issues

#### 1. Import Errors

```bash
# Reinstall in development mode
pip uninstall vislang_ultralow
pip install -e ".[dev,test]"

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

#### 2. Docker Build Issues

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t vislang-ultralow-resource .

# Check build context
docker build --progress=plain -t vislang-ultralow-resource .
```

#### 3. Test Failures

```bash
# Run tests with verbose output
pytest -v -s tests/

# Debug specific test
pytest -v -s tests/unit/test_specific.py::TestClass::test_method --pdb

# Check test configuration
pytest --collect-only
```

#### 4. Monitoring Issues

```bash
# Check service status
docker-compose -f monitoring/docker-compose.yml ps

# View logs
docker-compose -f monitoring/docker-compose.yml logs grafana
docker-compose -f monitoring/docker-compose.yml logs prometheus

# Restart services
docker-compose -f monitoring/docker-compose.yml restart
```

#### 5. GitHub Actions Issues

```bash
# Test workflows locally with act
act -j test

# Check workflow syntax
gh workflow view

# Debug failed runs
gh run view --log
```

### Performance Optimization

#### Database Optimization

```python
# Connection pooling
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

#### Caching Optimization

```python
# Redis caching
REDIS_CACHE_TTL=3600
REDIS_MAX_CONNECTIONS=100
MODEL_CACHE_SIZE=10
```

#### Docker Optimization

```dockerfile
# Multi-stage builds to reduce image size
# Use .dockerignore to exclude unnecessary files
# Layer caching optimization
```

## Security Integration

### Security Scanning

```bash
# Run all security checks
make security-scan

# Specific scans
bandit -r src/                    # Code analysis
safety check                     # Dependency vulnerabilities
trivy image vislang-ultralow     # Container scan
```

### Secrets Management

```bash
# Scan for secrets
detect-secrets scan --all-files
truffleHog --regex --entropy=False .

# Create secrets baseline
detect-secrets scan --all-files --baseline .secrets.baseline
```

### Access Control

```bash
# Setup GitHub branch protection
gh api repos/:owner/:repo/branches/main/protection --method PUT --input protection-rules.json

# Configure environment protection
# (Done through GitHub UI: Settings > Environments)
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Weekly tasks
make maintenance-weekly

# Monthly tasks  
make maintenance-monthly

# Update dependencies
make update-dependencies

# Security updates
make security-updates
```

### Monitoring Maintenance

```bash
# Cleanup old metrics
make cleanup-metrics

# Rotate logs
make rotate-logs

# Update dashboards
make update-dashboards
```

## Support and Resources

### Documentation

- [Development Guide](./DEVELOPMENT.md)
- [API Documentation](./API.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Workflow Guide](./workflows/WORKFLOW_GUIDE.md)

### Tools and Services

- **Monitoring**: Prometheus, Grafana, Alertmanager
- **Security**: Trivy, Bandit, Safety, detect-secrets
- **Quality**: SonarQube, CodeClimate, Codecov
- **CI/CD**: GitHub Actions, Docker, Kubernetes

### Getting Help

1. Check this integration guide
2. Review component-specific documentation
3. Search existing GitHub issues
4. Create new issue with `integration` label
5. Contact the development team

## Best Practices

### Development Workflow

1. Use feature branches with descriptive names
2. Write tests before implementing features
3. Run pre-commit hooks before committing
4. Keep commits small and focused
5. Write descriptive commit messages
6. Request code reviews for all changes

### Security Practices

1. Never commit secrets or credentials
2. Use environment variables for configuration
3. Regularly update dependencies
4. Run security scans in CI/CD
5. Follow principle of least privilege
6. Monitor security advisories

### Monitoring Practices

1. Set up appropriate alerting thresholds
2. Monitor business metrics, not just technical
3. Create runbooks for common issues
4. Regular review and tuning of alerts
5. Implement proper logging strategies

### Performance Practices

1. Profile application regularly
2. Monitor resource usage trends
3. Optimize database queries
4. Use caching strategically  
5. Plan for scalability from start

---

*This integration guide is maintained as part of the VisLang-UltraLow-Resource project documentation.*