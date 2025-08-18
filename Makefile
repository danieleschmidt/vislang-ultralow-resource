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

# Monitoring
monitoring-up:  ## Start monitoring stack
	docker-compose --profile monitoring up -d

# Database utilities
db-shell:  ## Open database shell
	docker-compose exec postgres psql -U vislang -d vislang_db