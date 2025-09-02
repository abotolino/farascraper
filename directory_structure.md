# Directory Structure Preservation

To maintain the project directory structure in Git, create these empty `.gitkeep` files:

## Required .gitkeep Files

Create these empty files to preserve directory structure:

```bash
# Data directories
touch data/raw/fara_documents/.gitkeep
touch data/processed/extracted_data/.gitkeep  
touch data/processed/validated_data/.gitkeep
touch data/logs/.gitkeep
touch data/cache/.gitkeep
touch data/backups/.gitkeep

# Test directories
touch tests/unit/.gitkeep
touch tests/integration/.gitkeep
touch tests/fixtures/.gitkeep

# Scripts directories  
touch scripts/maintenance/.gitkeep

# Documentation
touch docs/.gitkeep
```

## Commands to Create All Directories

Run this script to create the complete directory structure:

```bash
#!/bin/bash
# Create all required directories with .gitkeep files

# Data directories
mkdir -p data/raw/fara_documents data/processed/extracted_data data/processed/validated_data
mkdir -p data/logs data/cache data/backups

# Source directories (should already exist)
mkdir -p src/common src/downloader src/ocr/processors src/ocr/extractors src/ocr/validators
mkdir -p src/automation src/pipeline

# Test directories
mkdir -p tests/unit tests/integration tests/fixtures

# Scripts directories
mkdir -p scripts/maintenance

# Documentation
mkdir -p docs

# Create .gitkeep files for empty directories
touch data/raw/fara_documents/.gitkeep
touch data/processed/extracted_data/.gitkeep  
touch data/processed/validated_data/.gitkeep
touch data/logs/.gitkeep
touch data/cache/.gitkeep
touch data/backups/.gitkeep
touch tests/unit/.gitkeep
touch tests/integration/.gitkeep
touch tests/fixtures/.gitkeep
touch scripts/maintenance/.gitkeep
touch docs/.gitkeep

echo "Directory structure created successfully!"
```

## Alternative: Manual Creation

If you prefer to create them manually:

1. **Data directories:**
   - `data/raw/fara_documents/.gitkeep`
   - `data/processed/extracted_data/.gitkeep`
   - `data/processed/validated_data/.gitkeep`
   - `data/logs/.gitkeep`
   - `data/cache/.gitkeep`
   - `data/backups/.gitkeep`

2. **Test directories:**
   - `tests/unit/.gitkeep`
   - `tests/integration/.gitkeep`
   - `tests/fixtures/.gitkeep`

3. **Other directories:**
   - `scripts/maintenance/.gitkeep`
   - `docs/.gitkeep`