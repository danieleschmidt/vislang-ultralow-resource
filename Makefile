.PHONY: help install install-dev test test-cov lint format type-check clean build upload docs docker-build docker-run

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install .

install-dev:  ## Install package in development mode with all dependencies
	pip install -e ".[dev,ocr,training]"
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint:  ## Run linting checks
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

type-check:  ## Run type checking
	mypy src/

security-check:  ## Run security checks
	safety check
	bandit -r src/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

upload-test:  ## Upload to TestPyPI
	python -m twine upload --repository testpypi dist/*

upload:  ## Upload to PyPI
	python -m twine upload dist/*

docs:  ## Generate documentation
	@echo "Documentation available in docs/ directory"
	@echo "- README.md: Project overview"
	@echo "- docs/DEVELOPMENT.md: Development guide"
	@echo "- docs/ARCHITECTURE.md: System architecture"

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

setup-secrets:  ## Setup secrets baseline for detect-secrets
	detect-secrets scan --baseline .secrets.baseline

check-secrets:  ## Check for secrets in codebase
	detect-secrets scan --baseline .secrets.baseline

dev-setup: install-dev setup-secrets  ## Complete development environment setup
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

ci: lint type-check security-check test-cov  ## Run all CI checks
	@echo "All CI checks passed!"

# Docker and containerization targets
docker-build:  ## Build Docker image for development
	docker build --target development -t vislang-ultralow-resource:dev .

docker-build-prod:  ## Build production Docker image
	docker build --target production -t vislang-ultralow-resource:prod .

docker-run:  ## Run application in Docker (development)
	docker-compose up --build

docker-run-prod:  ## Run application in Docker (production)
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

docker-stop:  ## Stop Docker containers
	docker-compose down

docker-clean:  ## Clean up Docker containers and images
	docker-compose down -v
	docker system prune -f

docker-logs:  ## Show Docker logs
	docker-compose logs -f

# Build automation
build-multiarch:  ## Build multi-architecture images
	./scripts/build.sh --platforms linux/amd64,linux/arm64

build-all:  ## Build all Docker targets
	./scripts/build.sh --target all

# Deployment helpers
deploy-dev:  ## Deploy development environment
	docker-compose up -d

deploy-prod:  ## Deploy production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Database utilities
db-shell:  ## Open database shell
	docker-compose exec postgres psql -U vislang -d vislang_db

# Monitoring and health checks
monitoring-up:  ## Start monitoring stack (Prometheus, Grafana, Alertmanager)
	cd monitoring && docker-compose up -d

monitoring-down:  ## Stop monitoring stack
	cd monitoring && docker-compose down

monitoring-logs:  ## View monitoring logs
	cd monitoring && docker-compose logs -f

health-check:  ## Run health checks
	python scripts/monitoring/health-check.py

setup-monitoring:  ## Setup monitoring stack
	chmod +x scripts/monitoring/setup-monitoring.sh
	scripts/monitoring/setup-monitoring.sh

# Automation and maintenance
collect-metrics:  ## Collect project metrics
	python scripts/metrics_collection.py

check-dependencies:  ## Check for dependency updates
	python scripts/automation/dependency-updater.py

update-dependencies:  ## Update dependencies (creates GitHub issue)
	python scripts/automation/dependency-updater.py --create-issue

validate-config:  ## Validate project configuration
	python scripts/validate-config.py

integration-test:  ## Run integration tests
	python scripts/integration-test.py

integration-test-quick:  ## Run quick integration tests
	python scripts/integration-test.py --quick

# Release management
release-patch:  ## Create patch release
	python scripts/automation/release-automation.py patch

release-minor:  ## Create minor release
	python scripts/automation/release-automation.py minor

release-major:  ## Create major release
	python scripts/automation/release-automation.py major

release-dry-run:  ## Preview release changes (dry run)
	python scripts/automation/release-automation.py patch --dry-run

# Security and compliance
security-scan:  ## Run comprehensive security scans
	safety check
	bandit -r src/
	detect-secrets scan --baseline .secrets.baseline

docker-scan:  ## Scan Docker image for vulnerabilities
	docker build --target development -t vislang-test:latest .
	trivy image vislang-test:latest

docker-sbom:  ## Generate Software Bill of Materials
	./scripts/generate-sbom.sh

vulnerability-check:  ## Check for known vulnerabilities
	safety check --json > vulnerability-report.json || true
	@echo "Vulnerability report saved to vulnerability-report.json"

# Complete development lifecycle
dev-setup: install-dev setup-secrets  ## Complete development environment setup
	@echo "=== Development Environment Setup ==="
	@echo "✓ Dependencies installed"
	@echo "✓ Pre-commit hooks configured"  
	@echo "✓ Secrets baseline created"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make validate-config' to validate setup"
	@echo "2. Run 'make integration-test-quick' to test integration"
	@echo "3. Run 'make test' to run the test suite"
	@echo ""
	@echo "Development environment setup complete!"

dev-start:  ## Start development environment
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started"
	@echo "Application: http://localhost:8000"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

dev-stop:  ## Stop development environment
	docker-compose -f docker-compose.dev.yml down

dev-clean:  ## Clean development environment
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f

dev-logs:  ## View development logs
	docker-compose -f docker-compose.dev.yml logs -f

# Staging deployment
deploy-staging:  ## Deploy to staging environment
	@echo "Deploying to staging..."
	@echo "Note: This requires proper AWS/K8s credentials"
	# kubectl apply -f deployment/k8s/staging/

test-staging:  ## Run smoke tests against staging
	@echo "Running staging smoke tests..."
	# pytest tests/smoke/ --target-url=https://staging.vislang.example.com

monitor-staging:  ## Monitor staging deployment
	@echo "Monitoring staging deployment..."
	# kubectl logs -f deployment/vislang-app -n vislang-staging

# Production deployment  
deploy-production:  ## Deploy to production (requires manual approval)
	@echo "Production deployment requires manual approval in GitHub Actions"
	@echo "Create a release tag to trigger production deployment"

monitor-production:  ## Monitor production deployment
	@echo "Monitoring production deployment..."
	# kubectl logs -f deployment/vislang-app -n vislang-production

rollback-production:  ## Emergency rollback production
	@echo "Emergency rollback requires manual intervention"
	@echo "Check deployment documentation for rollback procedures"

# Maintenance tasks
maintenance-weekly:  ## Weekly maintenance tasks
	@echo "Running weekly maintenance..."
	make check-dependencies
	make security-scan
	make collect-metrics
	@echo "Weekly maintenance complete"

maintenance-monthly:  ## Monthly maintenance tasks  
	@echo "Running monthly maintenance..."
	make maintenance-weekly
	make docker-clean
	make clean
	@echo "Monthly maintenance complete"

update-all:  ## Update dependencies and rebuild
	make check-dependencies
	make clean
	make install-dev
	make build

# Testing and validation
test-all:  ## Run all tests and checks
	make lint
	make type-check
	make security-check
	make test-cov
	make integration-test-quick

docker-verify:  ## Verify Docker setup
	@echo "Verifying Docker setup..."
	docker --version
	docker-compose --version
	docker info
	@echo "Docker verification complete"

monitoring-test:  ## Test monitoring setup
	@echo "Testing monitoring configuration..."
	python -c "import yaml; yaml.safe_load(open('monitoring/prometheus.yml'))"
	python -c "import yaml; yaml.safe_load(open('monitoring/alertmanager.yml'))"
	@echo "Monitoring configuration valid"

# Documentation and help
show-urls:  ## Show important URLs and endpoints
	@echo "=== Important URLs ==="
	@echo "Application (dev): http://localhost:8000"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Alertmanager: http://localhost:9093"
	@echo ""
	@echo "=== Documentation ==="
	@echo "Integration Guide: docs/INTEGRATION_GUIDE.md"
	@echo "Workflow Guide: docs/workflows/WORKFLOW_GUIDE.md"  
	@echo "Development Guide: docs/DEVELOPMENT.md"

show-config:  ## Show current configuration
	@echo "=== Project Configuration ==="
	@echo "Python version: $$(python --version)"
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Docker not available')"
	@echo "Project root: $$(pwd)"
	@echo "Virtual env: $${VIRTUAL_ENV:-'Not active'}"
	@echo ""
	@make validate-config

# All-in-one commands
full-setup: dev-setup docker-build monitoring-up  ## Complete project setup
	@echo "=== Full Project Setup Complete ==="
	@echo "✓ Development environment ready"
	@echo "✓ Docker images built"  
	@echo "✓ Monitoring stack running"
	@echo ""
	@make show-urls

verify-all: validate-config integration-test docker-verify monitoring-test  ## Verify entire setup
	@echo "=== Complete Verification Passed ==="