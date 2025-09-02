#!/usr/bin/env python3
"""
FARA Pipeline Project Setup Script

This script helps set up the FARA document processing pipeline project
by creating necessary directories, checking dependencies, and validating configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message: str, color: str) -> None:
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.END}")

def print_header(message: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"{message}")
    print(f"{'='*60}{Colors.END}\n")

def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_colored(f"Python {version.major}.{version.minor}.{version.micro} - OK", Colors.GREEN)
        return True
    else:
        print_colored(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", Colors.RED)
        return False

def check_system_dependency(command: str, package_name: str, install_hint: str = "") -> bool:
    """Check if a system dependency is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print_colored(f"{package_name} - OK", Colors.GREEN)
        if result.stdout.strip():
            print(f"   Version: {result.stdout.split()[0] if result.stdout.split() else 'Unknown'}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_colored(f"{package_name} - Not found", Colors.RED)
        if install_hint:
            print_colored(f"   Install with: {install_hint}", Colors.YELLOW)
        return False

def create_directory_structure() -> bool:
    """Create the required directory structure."""
    directories = [
        "data/raw/fara_documents",
        "data/processed/extracted_data",
        "data/processed/validated_data", 
        "data/logs",
        "data/cache",
        "data/backups",
        "tests/unit",
        "tests/integration", 
        "tests/fixtures",
        "scripts/maintenance",
        "docs"
    ]
    
    created_dirs = []
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            # Create .gitkeep file
            (path / '.gitkeep').touch()
            created_dirs.append(dir_path)
    
    if created_dirs:
        print_colored("Created missing directories:", Colors.GREEN)
        for dir_path in created_dirs:
            print(f"   {dir_path}")
    else:
        print_colored("All directories already exist", Colors.GREEN)
    
    return True

def check_env_file() -> bool:
    """Check if .env file exists and is configured."""
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if not env_path.exists():
        if env_example_path.exists():
            print_colored("⚠️  .env file not found, but .env.example exists", Colors.YELLOW)
            print_colored("   Please copy .env.example to .env and configure your credentials", Colors.YELLOW)
            print_colored("   Command: cp .env.example .env", Colors.BLUE)
        else:
            print_colored("Neither .env nor .env.example found", Colors.RED)
            return False
    else:
        print_colored(".env file exists", Colors.GREEN)
        
        # Check if basic configuration is present
        with open(env_path, 'r') as f:
            content = f.read()
            if 'FARA_USERNAME=your_email@example.com' in content:
                print_colored("⚠️  Please update .env with your actual FARA credentials", Colors.YELLOW)
            else:
                print_colored(".env appears to be configured", Colors.GREEN)
    
    return True

def install_python_dependencies() -> bool:
    """Install Python dependencies."""
    requirements_file = Path('requirements.txt')
    
    if not requirements_file.exists():
        print_colored("requirements.txt not found", Colors.RED)
        return False
    
    try:
        print_colored("Installing Python dependencies...", Colors.BLUE)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print_colored("Python dependencies installed", Colors.GREEN)
        
        # Install package in development mode
        print_colored("Installing package in development mode...", Colors.BLUE)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                      check=True)
        print_colored("Package installed in development mode", Colors.GREEN)
        
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"Failed to install dependencies: {e}", Colors.RED)
        return False

def run_basic_tests() -> bool:
    """Run basic tests to verify installation."""
    try:
        print_colored("Running basic import tests...", Colors.BLUE)
        
        # Test imports
        test_imports = [
            ('requests', 'Web scraping'),
            ('bs4', 'BeautifulSoup'),
            ('cv2', 'OpenCV'),
            ('PIL', 'Pillow'),
            ('pytesseract', 'Tesseract Python wrapper'),
        ]
        
        for module, description in test_imports:
            try:
                __import__(module)
                print_colored(f"   {description} - OK", Colors.GREEN)
            except ImportError:
                print_colored(f"   {description} - Import failed", Colors.RED)
                return False
        
        print_colored("All basic tests passed", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"Test failed: {e}", Colors.RED)
        return False

def print_next_steps():
    """Print next steps for the user."""
    print_header("NEXT STEPS")
    
    steps = [
        "1. Configure your FARA credentials in .env file:",
        "   - Set FARA_USERNAME to your email",
        "   - Set FARA_PASSWORD to your password",
        "",
        "2. Test the scraper connection:",
        "   python scripts/test_scraper.py",
        "",
        "3. Run the full pipeline:", 
        "   python scripts/run_pipeline.py",
        "",
        "4. Check logs for any issues:",
        "   tail -f data/logs/fara_pipeline.log",
        "",
        "5. Read the documentation:",
        "   - README.md for usage instructions",
        "   - CONTRIBUTING.md for development guidelines"
    ]
    
    for step in steps:
        if step.startswith(("   python", "   tail", "   cp")):
            print_colored(step, Colors.BLUE)
        elif step.startswith("   -"):
            print_colored(step, Colors.YELLOW)
        else:
            print(step)

def main():
    """Main setup function."""
    print_header("FARA DOCUMENT PROCESSING PIPELINE - SETUP")
    
    print_colored("Setting up your FARA document processing environment...", Colors.BLUE)
    
    # Track setup success
    checks_passed = 0
    total_ # type: ignore