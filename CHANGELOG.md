# Changelog

All notable changes to the VisLang-UltraLow-Resource project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and configuration
- Core package skeleton with main components
- Comprehensive development tooling setup
- Documentation structure and guidelines
- Security and ethical AI policies
- Multi-language OCR engine support framework
- Vision-language model training infrastructure
- Humanitarian data source integration framework

### Security
- Security policy and vulnerability reporting process  
- Pre-commit hooks for secrets detection
- Dependency security scanning with Safety
- Code security analysis with Bandit

## [0.1.0] - 2025-01-31

### Added
- Initial release with foundational SDLC structure
- Project packaging and dependency management
- Code quality tools and pre-commit hooks
- Testing framework and CI/CD foundation
- Documentation templates and guidelines
- Security policies for humanitarian AI applications
- Multi-language support architecture
- Ethical AI guidelines and community standards

### Technical Details
- Python 3.8+ support
- PyTorch-based model training infrastructure
- HuggingFace integration for datasets and models
- Comprehensive testing with pytest
- Code formatting with Black and isort
- Type checking with mypy
- Linting with flake8
- Security scanning with multiple tools

---

## Contributing

When making changes, please:
1. Add entries to the [Unreleased] section
2. Follow the format: `### [Added/Changed/Deprecated/Removed/Fixed/Security]`
3. Include breaking changes in a separate `### Breaking Changes` section
4. Move items to a version section when releasing

## Release Process

1. Update version numbers in `pyproject.toml` and `src/vislang_ultralow/__init__.py`
2. Move [Unreleased] items to new version section
3. Add new [Unreleased] section
4. Create git tag and release
5. Update documentation as needed