# Contributing to VisLang-UltraLow-Resource

We welcome contributions to this humanitarian AI project! This guide will help you get started with contributing to vision-language models for ultra-low-resource languages.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Setup

### Prerequisites
- Python 3.8+ 
- Git
- Basic familiarity with machine learning and NLP concepts

### Installation

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/vislang-ultralow-resource.git
cd vislang-ultralow-resource
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards
3. Add tests for new functionality
4. Run the test suite:
```bash
pytest
```

5. Run code formatting and linting:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Add comprehensive tests for new features

### Testing

- Write unit tests for all new functionality
- Ensure test coverage remains above 80%
- Include integration tests for complex workflows
- Test with multiple Python versions when possible

### Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Include examples in docstrings
- Update CHANGELOG.md for notable changes

## Ethical Considerations

### Humanitarian Focus
This project serves humanitarian organizations and vulnerable populations:

- **Data Privacy**: Ensure no sensitive information is logged or exposed
- **Cultural Sensitivity**: Be respectful of cultural contexts in all languages
- **Accessibility**: Design for users with limited technical resources
- **Transparency**: Make model limitations and biases clear

### Language Representation
- Prioritize truly under-resourced languages
- Avoid reinforcing linguistic biases
- Collaborate with native speakers when possible
- Respect linguistic diversity and cultural contexts

## Types of Contributions

### Welcome Contributions
- **New Language Support**: Adding support for additional low-resource languages
- **Data Sources**: Integrating new humanitarian report sources
- **OCR Improvements**: Enhancing multilingual text extraction
- **Model Training**: Improving fine-tuning approaches
- **Documentation**: Clarifying setup and usage instructions
- **Testing**: Adding test coverage and integration tests
- **Bug Fixes**: Resolving issues with existing functionality

### Review Process

1. **Automated Checks**: All PRs must pass CI/CD checks
2. **Code Review**: At least one maintainer review required
3. **Testing**: New features require comprehensive tests
4. **Documentation**: Updates must include relevant documentation
5. **Ethical Review**: Changes affecting data handling need ethical assessment

## Getting Help

- **Technical Questions**: Open a GitHub issue with the `question` label
- **Feature Requests**: Open a GitHub issue with the `enhancement` label  
- **Bug Reports**: Open a GitHub issue with the `bug` label
- **Security Issues**: Email maintainers directly (see SECURITY.md)

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- CHANGELOG.md for significant contributions
- Research publications when appropriate (with consent)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make AI more accessible for humanitarian applications and ultra-low-resource languages!