#!/usr/bin/env python3
"""
Simple OCR Processing Script for FARA Documents

This script processes all PDF documents in the data/raw/fara_documents/ directory
and extracts text using OCR, saving results to data/processed/.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.pipeline.fara_ocr_clean import FARAOCRProcessor, FARAConfig
    from src.common.logger import get_logger
    from config.settings import settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

logger = get_logger(__name__)


def process_downloaded_documents():
    """Process all downloaded FARA documents with OCR"""
    
    # Check for downloaded documents
    pdf_dir = settings.raw_documents
    if not pdf_dir.exists():
        print(f"❌ Documents directory not found: {pdf_dir}")
        print("Run the test_scraper.py first to download documents")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("Run the test_scraper.py first to download documents")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF documents to process")
    
    # Initialize OCR processor with custom config
    try:
        # Create custom config that uses our project settings paths
        config = FARAConfig()
        config.input_dir = settings.raw_documents
        config.output_dir = settings.processed_data
        config.log_dir = settings.logs
        
        processor = FARAOCRProcessor(config)
        logger.info("OCR processor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OCR processor: {e}")
        return
    
    # Process each document
    processed_count = 0
    failed_count = 0
    
    for pdf_file in pdf_files:
        print(f"\n🔄 Processing: {pdf_file.name}")
        
        try:
            # Process the PDF - returns FARAData object or None
            result = processor.process_single_document(pdf_file)
            
            if result is not None:
                print(f"✅ Successfully processed: {pdf_file.name}")
                print(f"   Registrant: {result.registrant_name or 'Not found'}")
                print(f"   Confidence: {result.confidence_score:.1f}%")
                processed_count += 1
            else:
                print(f"❌ Failed to process: {pdf_file.name}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Exception processing {pdf_file.name}: {e}")
            logger.error("OCR processing failed", file=pdf_file.name, error=str(e))
            failed_count += 1
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {processed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Results saved to: {settings.processed_data}")
    
    if processed_count > 0:
        print(f"\n💡 Next steps:")
        print(f"   • Check extracted data in: {settings.processed_data}")
        print(f"   • Review processing logs in: {settings.logs}")


def main():
    """Main entry point"""
    print("🔍 FARA OCR Document Processing")
    print("=" * 50)
    
    try:
        process_downloaded_documents()
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error("OCR processing script failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()