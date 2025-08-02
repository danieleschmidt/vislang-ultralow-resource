# Multi-stage build for VisLang-UltraLow-Resource
# Base stage with security best practices
FROM python:3.11-slim as base

# Security: Create non-root user
RUN groupadd -r vislang && useradd -r -g vislang vislang

# Install system dependencies required for OCR and ML libraries
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-amh \
    tesseract-ocr-swa \
    tesseract-ocr-fra \
    tesseract-ocr-spa \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libpoppler-cpp-dev \
    libmagic1 \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml pytest.ini ./
COPY requirements*.txt ./

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,ocr,training]"

# Copy source code
COPY . .

# Set ownership to non-root user
RUN chown -R vislang:vislang /app

USER vislang

EXPOSE 8000

CMD ["python", "-m", "vislang_ultralow.api"]

# Production stage
FROM base as production

WORKDIR /app

# Copy only production files
COPY pyproject.toml ./
COPY src/ ./src/
COPY requirements.txt ./

# Install production dependencies only
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir gunicorn uvicorn

# Create necessary directories
RUN mkdir -p /app/data /app/logs \
    && chown -R vislang:vislang /app

USER vislang

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "vislang_ultralow.api:app"]

# Final minimal stage for deployment
FROM production as minimal

# Remove unnecessary packages for minimal footprint
RUN apt-get update && apt-get remove -y \
    curl \
    git \
    pkg-config \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*