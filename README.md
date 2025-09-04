# farascraper
A tool to scrape FARA documents for OpenSecrets

# FARA Document Processing Pipeline

> **Automated processing system for Foreign Agents Registration Act (FARA) documents from fara.crphq.org**

A complete 3-stage automation pipeline that downloads FARA documents, extracts structured data using OCR, and prepares data for automated form submission.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/abotolino/farascraper.git
cd farascraper

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure credentials
cp .env.example .env
# Edit .env with your FARA credentials

# Run a test
python scripts/test_scraper.py
```

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features

### Stage 1: Document Download ✅ **COMPLETED**
- **Automated Authentication**: Handles Rails/Devise login with CSRF tokens
- **Document Discovery**: Finds documents waiting to be processed
- **S3 Integration**: Downloads PDFs from AWS S3 URLs
- **Rate Limiting**: Respectful scraping with configurable delays
- **Session Management**: Maintains authenticated sessions with auto-retry

### Stage 2: OCR Processing ✅ **COMPLETED**
- **High Accuracy OCR**: 80-93% accuracy on FARA forms using Tesseract
- **Structured Extraction**: Converts PDFs to structured JSON/CSV data
- **Form-Aware Processing**: Recognizes FARA Supplemental Statement layouts
- **Financial Data**: Extracts tables, dates, amounts, and foreign principals
- **Confidence Scoring**: Quality metrics for extracted data

### Stage 3: Web Automation 🔄 **PLANNED**
- **Automated Form Filling**: Selenium-based web form population
- **Data Validation**: Ensures extracted data matches form requirements
- **Submission Tracking**: Monitors form submission success/failure

### System Features
- **Professional Architecture**: Modular, testable, and maintainable code
- **Comprehensive Logging**: Structured JSON logging with rotation
- **Error Recovery**: Robust error handling with retry mechanisms
- **Pipeline Orchestration**: Job tracking and state management
- **Configuration Management**: Environment-based settings

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │      OCR        │    │      Web        │
│   Download      │───▶│   Processing    │───▶│   Automation    │
│                 │    │                 │    │   (Planned)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw PDFs      │    │ Structured Data │    │   Completed     │
│   from S3       │    │   JSON/CSV      │    │   Submissions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Project Structure

```
farascraper/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── .env.example                 # Configuration template
├── .gitignore                   # Git exclusions
├── pyproject.toml              # Modern Python packaging
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration
│
├── src/
│   ├── __init__.py
│   ├── common/                  # Shared utilities
│   │   ├── __init__.py
│   │   ├── logger.py            # Structured logging
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── utils.py             # Utility functions
│   │
│   ├── downloader/              # Stage 1: Document Download
│   │   ├── __init__.py
│   │   ├── fara_scraper.py      # Web scraper (400+ lines)
│   │   └── download_manager.py  # Download coordination
│   │
│   ├── ocr/                     # Stage 2: OCR Processing
│   │   ├── __init__.py
│   │   ├── processors/          # OCR engines
│   │   ├── extractors/          # Data extraction
│   │   └── validators/          # Data validation
│   │
│   ├── automation/              # Stage 3: Web Automation
│   │   ├── __init__.py
│   │   └── (planned)
│   │
│   └── pipeline/                # Pipeline Orchestration
│       ├── __init__.py
│       └── orchestrator.py      # Job management
│
├── data/                        # Data storage
│   ├── raw/fara_documents/      # Downloaded PDFs
│   ├── processed/               # OCR results
│   ├── logs/                    # Application logs
│   └── cache/                   # Temporary files
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test data
│
├── scripts/                     # Utility scripts
│   ├── run_pipeline.py          # Main execution
│   ├── test_scraper.py          # Test scraper
│   └── maintenance/             # Admin scripts
│
└── docs/                        # Documentation
    ├── api.md                   # API reference
    ├── setup.md                 # Detailed setup
    └── troubleshooting.md       # Common issues
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine
- Valid credentials for fara.crphq.org

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
1. Download Tesseract from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install poppler: `conda install poppler` or download from [poppler website](https://poppler.freedesktop.org/)

### Python Setup

```bash
# Clone repository
git clone https://github.com/abotolino/farascraper.git
cd fara-pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
# Test system dependencies
tesseract --version
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import PIL; print('Pillow: OK')"

# Test package installation
python -c "from src.downloader import fara_scraper; print('Package installed successfully')"
```

## ⚙️ Configuration

### Environment Variables

Copy the example configuration file and customize it:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# FARA Website Credentials
FARA_USERNAME=your_email@opensecrets.com
FARA_PASSWORD=your_secure_password
FARA_BASE_URL=http://fara.crphq.org

# Processing Configuration
BATCH_SIZE=10
RATE_LIMIT_DELAY=2.0
MAX_RETRIES=3

# OCR Configuration
OCR_PROVIDER=tesseract
OCR_CONFIDENCE_THRESHOLD=0.8
OCR_LANGUAGES=eng

# File Paths
DATA_DIR=./data
LOG_DIR=./data/logs
CACHE_DIR=./data/cache

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE_ROTATION=true
```

### Configuration Sections

| Section | Purpose | Key Settings |
|---------|---------|--------------|
| **FARA Settings** | Website access | Username, password, base URL |
| **Processing** | Pipeline behavior | Batch size, rate limits, retries |
| **OCR** | Text extraction | Provider, confidence threshold, languages |
| **Paths** | File organization | Data directory, logs, cache |
| **Logging** | Monitoring | Log level, format, rotation |

## 📖 Usage

### Basic Usage

#### 1. Test Your Setup

```bash
# Test FARA website connection
python scripts/test_scraper.py

# Expected output:
# Authentication successful
# Found X documents waiting to be processed
# Successfully downloaded sample PDF
```

#### 2. Run the Full Pipeline

```bash
# Process all available documents
python scripts/run_pipeline.py

# Process specific number of documents
python scripts/run_pipeline.py --batch-size 5

# Resume failed jobs
python scripts/run_pipeline.py --resume-failed
```

#### 3. Monitor Progress

```bash
# View logs in real-time
tail -f data/logs/fara_pipeline.log

# Check job status
python -c "
from src.pipeline.orchestrator import PipelineOrchestrator
orchestrator = PipelineOrchestrator()
print(orchestrator.get_pipeline_status())
"
```

### Advanced Usage

#### Custom OCR Processing

```python
from src.ocr.processors import TesseractProcessor
from src.ocr.extractors import FARAExtractor

# Initialize components
processor = TesseractProcessor()
extractor = FARAExtractor()

# Process single document
pdf_path = "data/raw/fara_documents/sample.pdf"
ocr_result = processor.process_pdf(pdf_path)
structured_data = extractor.extract_fara_data(ocr_result)

print(f"Extracted data: {structured_data}")
```

#### Pipeline Integration

```python
from src.pipeline.orchestrator import PipelineOrchestrator

# Create pipeline orchestrator
orchestrator = PipelineOrchestrator()

# Submit job
job_id = orchestrator.submit_job("batch_001")

# Monitor progress
status = orchestrator.get_job_status(job_id)
print(f"Job {job_id} status: {status.stage}")

# Get results
if status.stage == "completed":
    results = orchestrator.get_job_results(job_id)
    print(f"Processed {len(results.documents)} documents")
```

### Output Formats

#### JSON Output
```json
{
  "job_id": "batch_001_20250902_143022",
  "documents_processed": 15,
  "success_rate": 0.87,
  "extracted_data": [
    {
      "filename": "supplemental_statement_123.pdf",
      "registrant_name": "Example Corp",
      "registration_number": "12345",
      "foreign_principals": ["Foreign Entity A", "Foreign Entity B"],
      "financial_receipts": [
        {
          "foreign_principal": "Foreign Entity A",
          "date": "2024-Q3",
          "amount": 50000.00,
          "description": "Consulting services"
        }
      ],
      "filing_date": "2024-10-15",
      "confidence_score": 0.92
    }
  ]
}
```

#### CSV Output
```csv
filename,registrant_name,registration_number,foreign_principal,amount,date,confidence
supplemental_statement_123.pdf,Example Corp,12345,Foreign Entity A,50000.00,2024-Q3,0.92
supplemental_statement_123.pdf,Example Corp,12345,Foreign Entity B,25000.00,2024-Q3,0.89
```

## 🔍 API Reference

### Core Classes

#### `FARAScraperSession`
Main web scraping interface for FARA document download.

```python
from src.downloader.fara_scraper import FARAScraperSession

scraper = FARAScraperSession(
    username="your_email@example.com",
    password="your_password",
    rate_limit_delay=2.0
)

# Authenticate and download
scraper.authenticate()
documents = scraper.get_documents_waiting()
pdf_data = scraper.download_document(documents[0]['url'])
```

#### `PipelineOrchestrator`
Manages the complete processing pipeline.

```python
from src.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

# Submit batch job
job_id = orchestrator.submit_batch_job(batch_size=10)

# Monitor progress
status = orchestrator.get_job_status(job_id)
progress = orchestrator.get_pipeline_statistics()
```

#### `TesseractProcessor`
OCR processing engine for PDF text extraction.

```python
from src.ocr.processors import TesseractProcessor

processor = TesseractProcessor(
    confidence_threshold=0.8,
    languages=['eng']
)

# Process PDF
result = processor.process_pdf("path/to/document.pdf")
text = result.get_text()
confidence = result.get_confidence()
```

### Configuration Classes

All configuration is managed through environment variables and the `config.settings` module:

```python
from config.settings import get_settings

settings = get_settings()
print(f"FARA URL: {settings.fara.base_url}")
print(f"Batch size: {settings.processing.batch_size}")
print(f"OCR threshold: {settings.ocr.confidence_threshold}")
```

## 🛠 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Format code
black src/ tests/
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only

# Run tests with output
pytest -v -s

# Run specific test
pytest tests/unit/test_fara_scraper.py::TestAuthentication::test_login_success
```

### Code Quality

This project maintains high code quality standards:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing
- **Pre-commit hooks** for automated checks

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/your-feature-name`
2. **Write tests first**: Add tests in `tests/unit/` or `tests/integration/`
3. **Implement feature**: Add code in appropriate `src/` subdirectory
4. **Update documentation**: Modify README.md and relevant docs
5. **Run quality checks**: `pytest && black . && flake8 && mypy src/`
6. **Create pull request**: Include description and test results

## 🚨 Troubleshooting

### Common Issues

#### Authentication Problems
```bash
# Error: "Authentication failed"
# Solution: Verify credentials in .env file
python -c "from config.settings import get_settings; s=get_settings(); print(f'Username: {s.fara.username}')"
```

#### OCR Accuracy Issues
```bash
# Error: Low confidence scores
# Solution: Check image preprocessing
python scripts/debug_ocr.py path/to/problematic.pdf
```

#### Rate Limiting
```bash
# Error: "Too many requests"
# Solution: Increase delay in .env
RATE_LIMIT_DELAY=5.0  # Increase from 2.0 to 5.0 seconds
```

#### Missing Dependencies
```bash
# Error: "tesseract not found"
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
# Windows: Download from GitHub releases
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set debug level in .env
LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/run_pipeline.py --verbose

# Check logs
tail -f data/logs/fara_pipeline.log | grep ERROR
```

### Getting Help

1. **Check logs**: Look in `data/logs/fara_pipeline.log` for error details
2. **Run diagnostics**: Use `scripts/diagnose_system.py` to check setup
3. **Review configuration**: Verify all settings in `.env` file
4. **Test components**: Use individual test scripts in `scripts/` directory
5. **Create issue**: Open GitHub issue with log output and system details

### Performance Optimization

```bash
# For large batches
BATCH_SIZE=50
RATE_LIMIT_DELAY=1.0

# For high accuracy OCR
OCR_CONFIDENCE_THRESHOLD=0.9
OCR_PREPROCESSING=aggressive

# For faster processing
OCR_DPI=200  # Lower than default 300
PARALLEL_JOBS=4
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome!

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format code (`black .`)
7. Commit changes (`git commit -m 'Add AmazingFeature'`)
8. Push to branch (`git push origin feature/AmazingFeature`)
9. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include docstrings for public methods
- Maintain test coverage above 80%
- Update documentation for new features

## 📞 Support

- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Email**: [abotolino@opensecrets.com] for direct support

---

**Built with ❤️ for government transparency and automation**
