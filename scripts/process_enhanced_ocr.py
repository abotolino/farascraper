#!/usr/bin/env python3
"""
Enhanced OCR Processing Script for FARA Item 14(a) Tables

This script uses the enhanced OCR system to extract tabular data from FARA documents,
specifically targeting Item 14(a) financial information tables.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.pipeline.enhanced_fara_ocr import EnhancedFARAOCRProcessor, Item14aData
    from src.common.logger import get_logger
    from config.settings import settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

logger = get_logger(__name__)


def process_documents_enhanced():
    """Process all downloaded FARA documents with enhanced Item 14(a) extraction"""
    
    # Check for downloaded documents
    pdf_dir = settings.raw_documents
    if not pdf_dir.exists():
        print(f"Documents directory not found: {pdf_dir}")
        print("Run the test_scraper.py first to download documents")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        print("Run the test_scraper.py first to download documents")
        return
    
    print(f"Found {len(pdf_files)} PDF documents to process with enhanced Item 14(a) extraction")
    
    # Initialize enhanced OCR processor
    try:
        processor = EnhancedFARAOCRProcessor()
        logger.info("Enhanced OCR processor initialized")
    except Exception as e:
        print(f"Failed to initialize enhanced OCR processor: {e}")
        return
    
    # Ensure output directory exists
    output_dir = settings.processed_data
    output_dir.mkdir(exist_ok=True)
    
    # Process each document
    processed_count = 0
    failed_count = 0
    total_entries = 0
    total_amount = 0.0
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            # Process with enhanced Item 14(a) extraction
            result = processor.process_document_for_14a_tables(pdf_file)
            
            if result.table_detected:
                print(f"Successfully processed: {pdf_file.name}")
                print(f"   Tables detected: Yes")
                print(f"   Entries extracted: {result.total_entries}")
                print(f"   Total amount: ${result.total_amount:,.2f}")
                print(f"   Confidence: {result.extraction_confidence:.1f}%")
                
                # Save detailed results
                output_file = output_dir / f"{pdf_file.stem}_14a_enhanced.json"
                processor.save_14a_results(result, output_file)
                print(f"   Detailed JSON: {output_file.name}")
                
                # Show sample entries
                if result.entries:
                    print(f"   Sample entries:")
                    for i, entry in enumerate(result.entries[:2]):  # Show first 2 entries
                        print(f"      {i+1}. {entry.foreign_principal}")
                        if entry.amount:
                            print(f"         Amount: ${entry.amount} ({entry.date_received})")
                        if entry.purpose and len(entry.purpose) > 0:
                            purpose_preview = entry.purpose[:80] + "..." if len(entry.purpose) > 80 else entry.purpose
                            print(f"         Purpose: {purpose_preview}")
                
                processed_count += 1
                total_entries += result.total_entries
                total_amount += result.total_amount
                
            else:
                print(f"No Item 14(a) tables detected in: {pdf_file.name}")
                print(f"   This document may not contain financial receipt tables")
                
                # Still save the result for completeness
                output_file = output_dir / f"{pdf_file.stem}_14a_enhanced.json"
                processor.save_14a_results(result, output_file)
                
                processed_count += 1
                
        except Exception as e:
            print(f"Exception processing {pdf_file.name}: {e}")
            logger.error("Enhanced OCR processing failed", file=pdf_file.name, error=str(e))
            failed_count += 1
    
    # Summary
    print(f"\nEnhanced Processing Summary:")
    print(f"   Successfully processed: {processed_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Total entries extracted: {total_entries}")
    print(f"   Total amount across all documents: ${total_amount:,.2f}")
    print(f"   Results saved to: {settings.processed_data}")
    
    if processed_count > 0:
        print(f"\nNext steps:")
        print(f"   • Review extracted data in: {settings.processed_data}")
        print(f"   • Check individual *_14a_enhanced.json files for detailed results")
        print(f"   • Files with table_detected: true contain Item 14(a) financial data")
        print(f"   • Use the structured JSON format for further processing or analysis")

    # Create a summary report
    create_summary_report(processed_count, total_entries, total_amount, output_dir)


def create_summary_report(processed_count: int, total_entries: int, total_amount: float, output_dir: Path):
    """Create a summary report of all processed documents"""
    
    summary = {
        "processing_summary": {
            "documents_processed": processed_count,
            "total_entries_extracted": total_entries,
            "total_amount_all_documents": total_amount,
            "processing_timestamp": "2025-01-01 12:00:00"  # Will be updated at runtime
        },
        "document_details": []
    }
    
    # Collect details from individual files
    for json_file in output_dir.glob("*_14a_enhanced.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            doc_summary = {
                "filename": json_file.stem.replace('_14a_enhanced', ''),
                "table_detected": data.get('table_detected', False),
                "entries_count": data.get('total_entries', 0),
                "total_amount": data.get('total_amount', 0.0),
                "confidence": data.get('extraction_confidence', 0.0)
            }
            summary["document_details"].append(doc_summary)
            
        except Exception as e:
            logger.warning(f"Could not include {json_file.name} in summary: {e}")
    
    # Save summary report
    summary_file = output_dir / "enhanced_processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"   Summary report: {summary_file.name}")


def main():
    """Main entry point"""
    print("FARA Enhanced OCR - Item 14(a) Table Extraction")
    print("=" * 60)
    print("This enhanced system specifically targets financial tables")
    print("in FARA documents for improved accuracy and structure.")
    print()
    
    try:
        process_documents_enhanced()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.error("Enhanced OCR processing script failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()