# Contributing to FARA Document Processing Pipeline

Thank you for helping out with the FARA Document Processing Pipeline! This document provides guidelines for contributing to the project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Tesseract OCR
- Basic familiarity with web scraping and OCR concepts

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/abotolino/farascraper.git
cd farascraper
```

2. **Set up development environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

3. **Install development tools:**
```bash
pip install -r requirements-dev.txt
pre-commit install
```

4. **Verify setup:**
```bash
pytest tests/
python scripts/test_scraper.py --dry-run
```

## Development Process

### Workflow
1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages
Follow conventional commits format:
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(ocr): add support for multi-page documents`
- `fix(scraper): handle authentication timeout errors`
- `docs(readme): update installation instructions`


### Running Quality Checks
```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

## Testing

### Test Structure
```
tests/
├── unit/               # Unit tests for individual components
├── integration/        # Integration tests for component interactions
└── fixtures/          # Test data and mock objects
```

### Writing Tests
- Use pytest framework
- Write tests before implementing features (TDD preferred)
- Mock external dependencies (web requests, file system)
- Test both success and failure cases
- Include edge cases

Example test:
```python
import pytest
from unittest.mock import patch, Mock
from src.downloader.fara_scraper import FARAScraperSession

class TestFARAScraperSession:
    def test_authenticate_success(self):
        """Test successful authentication."""
        scraper = FARAScraperSession("user@test.com", "password")
        
        with patch('requests.Session.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.text = '<form>success</form>'
            
            result = scraper.authenticate()
            
            assert result is True
            assert scraper.is_authenticated
    
    def test_authenticate_failure(self):
        """Test authentication failure handling."""
        scraper = FARAScraperSession("user@test.com", "wrong_password")
        
        with patch('requests.Session.post') as mock_post:
            mock_post.return_value.status_code = 401
            
            with pytest.raises(AuthenticationError):
                scraper.authenticate()
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_fara_scraper.py

# Run with coverage
pytest --cov=src tests/

# Run tests with verbose output
pytest -v -s
```

## Submitting Changes

### Pull Request Process
1. **Update documentation** - Ensure README and relevant docs are updated
2. **Add tests** - New features must include tests
3. **Pass CI checks** - All automated checks must pass
4. **Request review** - Tag relevant maintainers for review

### Pull Request Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass

## Checklist
- [ ] Self-review of code completed
- [ ] Code is commented, particularly complex areas
- [ ] Corresponding changes to documentation made
- [ ] No new linting warnings
- [ ] Tests cover new functionality
```

### Review Process
- All CI checks must pass
- Address all feedback before merge
- Squash commits for clean history

## Issue Reporting

### Bug Reports
Include the following information:
- Python version
- Operating system
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
Include the following:
- Clear description of the feature
- Use case and motivation
- Potential implementation approach
- Breaking change considerations

### Issue Labels
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements or additions to docs
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed

## Development Areas

### High-Priority Areas
- Web automation component (Stage 3)
- OCR accuracy improvements
- Error handling and recovery
- Performance optimization

### Technical Debt
- Database integration for job persistence
- API development for external integration
- Docker containerization
- Enhanced logging and monitoring

### Documentation Needs
- API documentation
- Deployment guides
- Troubleshooting guides
- Video tutorials

## Getting Help

### Communication Channels
- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Email** - Direct contact for sensitive issues

### Resources
- [Python PEP 8](https://www.python.org/dev/peps/pep-0008/) - Style guide
- [pytest Documentation](https://docs.pytest.org/) - Testing framework
- [Tesseract Documentation](https://tesseract-ocr.github.io/) - OCR engine

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors graph

Thank you for contributing to making government data more accessible!