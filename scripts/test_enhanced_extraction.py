#!/usr/bin/env python3
"""
Test script for enhanced FARA OCR Item 14(a) table extraction
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.pipeline.enhanced_fara_ocr import EnhancedFARAOCRProcessor
    from config.settings import settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def test_enhanced_extraction():
    """Test the enhanced extraction on existing documents"""
    
    # Check for input files
    input_dir = settings.raw_documents
    pdf_files = list(input_dir.glob("*.pdf")) if input_dir.exists() else []
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        print("Please ensure you have FARA documents to test with")
        return
    
    print("FARA Enhanced OCR - Test Run")
    print("=" * 50)
    
    # Initialize enhanced processor
    processor = EnhancedFARAOCRProcessor()
    
    # Test on first document
    test_file = pdf_files[0]
    print(f"Testing with: {test_file.name}")
    
    # Process the document
    result = processor.process_document_for_14a_tables(test_file)
    
    # Display results
    print(f"\nExtraction Results:")
    print(f"  Table detected: {result.table_detected}")
    print(f"  Total entries: {result.total_entries}")
    print(f"  Total amount: ${result.total_amount:,.2f}")
    print(f"  Extraction confidence: {result.extraction_confidence:.1f}%")
    
    if result.entries:
        print(f"\nSample entries:")
        for i, entry in enumerate(result.entries[:3]):
            print(f"  {i+1}. {entry.foreign_principal}")
            if entry.date_received:
                print(f"     Date: {entry.date_received}")
            if entry.amount:
                print(f"     Amount: ${entry.amount} (${entry.amount_numeric:,.2f})")
            if entry.purpose:
                print(f"     Purpose: {entry.purpose[:60]}...")
            print(f"     Confidence: {entry.row_confidence:.1f}%")
            print()
        
        if len(result.entries) > 3:
            print(f"  ... and {len(result.entries) - 3} more entries")
    
    # Save test results
    output_dir = settings.processed_data
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{test_file.stem}_test_enhanced.json"
    
    processor.save_14a_results(result, output_file)
    print(f"\nTest results saved to: {output_file.name}")
    
    return result.total_entries > 0


if __name__ == "__main__":
    success = test_enhanced_extraction()
    
    if success:
        print("\nTest PASSED: Enhanced extraction found table data")
        sys.exit(0)
    else:
        print("\nTest FAILED: No table data extracted")
        sys.exit(1)