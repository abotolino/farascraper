"""Setup script for FARA Document Processing Pipeline."""

from setuptools import setup, find_packages
import os

# Read README for long description
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="fara-pipeline",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@opensecrets.com",
    description="Automated processing system for Foreign Agents Registration Act (FARA) documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abotolino/farascraper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Office/Business",
        "Topic :: Text Processing :: Markup :: HTML",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fara-pipeline=pipeline.orchestrator:main",
            "fara-scraper=downloader.fara_scraper:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords="fara, document processing, ocr, web scraping, government transparency",
    project_urls={
        "Bug Reports": "https://github.com/abotolino/farascraper/issues",
        "Source": "https://github.com/abotolino/farascraper",
        "Documentation": "https://github.com/abotolino/farascraper/blob/main/README.md",
    },
)