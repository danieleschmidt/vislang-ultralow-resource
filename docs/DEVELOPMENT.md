# Development Guide

This guide covers development setup, testing, and contribution workflows for VisLang-UltraLow-Resource.

## Quick Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/vislang-ultralow-resource.git
cd vislang-ultralow-resource
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Project Structure

```
vislang-ultralow-resource/
├── src/vislang_ultralow/          # Main package
│   ├── __init__.py               # Package initialization
│   ├── dataset.py                # Dataset building logic
│   ├── scraper.py                # Humanitarian report scraping
│   ├── trainer.py                # Model training utilities
│   └── utils/                    # Utility modules
├── tests/                        # Test suite
├── docs/                         # Documentation
├── pyproject.toml               # Project configuration
└── README.md                    # Project overview
```

## Development Workflow

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting and style checking
- **mypy**: Static type checking
- **pre-commit**: Automated pre-commit checks

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_dataset.py

# Run with verbose output
pytest -v
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture Overview

### Core Components

1. **HumanitarianScraper**: Extracts content from humanitarian reports
   - Web scraping from UN, WHO, NGO sources
   - PDF processing and text extraction
   - Multilingual content detection

2. **DatasetBuilder**: Creates vision-language datasets
   - OCR processing for infographics
   - Cross-lingual alignment
   - Quality assurance and filtering

3. **VisionLanguageTrainer**: Fine-tunes models
   - Support for popular VL models
   - Multi-language training strategies
   - Evaluation and metrics

### Key Design Principles

- **Ethical AI**: Privacy-preserving and culturally sensitive
- **Scalability**: Handle large-scale humanitarian document corpora  
- **Multilingual**: Support for 100+ languages including low-resource ones
- **Quality**: Robust validation and human-in-the-loop verification

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end workflow testing
3. **Data Tests**: Dataset quality and format validation
4. **Model Tests**: Training and inference validation

### Test Data

- Use synthetic or publicly available test data
- Never commit real humanitarian data
- Mock external API calls in tests
- Include multilingual test cases

## Documentation

### Documentation Types

- **API Documentation**: Auto-generated from docstrings
- **User Guides**: Setup and usage instructions
- **Developer Guides**: Architecture and contribution info
- **Examples**: Jupyter notebooks and scripts

### Writing Documentation

- Use clear, concise language
- Include code examples
- Consider non-native English speakers
- Test all code examples

## Debugging

### Common Issues

1. **OCR Failures**: Check image quality and language support
2. **Memory Issues**: Use batch processing for large datasets
3. **API Rate Limits**: Implement proper backoff strategies
4. **Unicode Issues**: Ensure proper encoding handling

### Debugging Tools

```bash
# Run with debugging
python -m pdb script.py

# Profile performance
python -m cProfile script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py
```

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Run full test suite
4. Create and push git tag
5. Build and upload to PyPI

```bash
# Build package
python -m build

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Performance Considerations

- **Memory Usage**: Large datasets require careful memory management
- **GPU Usage**: Optimize for both single and multi-GPU setups
- **Network I/O**: Implement caching for repeated downloads
- **Disk I/O**: Use efficient file formats (HDF5, Parquet)

## Security Guidelines

- Never commit API keys or credentials
- Sanitize all external inputs
- Use virtual environments
- Keep dependencies updated
- Audit data sources for sensitive information

---

For questions about development, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.