#!/usr/bin/env python3
"""
Main script to run the FARA document processing pipeline.
"""
import sys
import argparse
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.common.logger import get_logger
from src.common.exceptions import FARAError
from config.settings import settings

logger = get_logger(__name__)


def process_downloaded_documents():
    """Process all downloaded documents through the pipeline"""
    
    # Check for downloaded documents
    pdf_dir = settings.raw_documents
    if not pdf_dir.exists():
        print(f"❌ No documents directory found: {pdf_dir}")
        print("Run 'python3 scripts/test_scraper.py' first to download documents")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("Run 'python3 scripts/test_scraper.py' first to download documents")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF documents to process")
    
    orchestrator = PipelineOrchestrator()
    
    # Add jobs for all downloaded documents
    job_ids = []
    for pdf_file in pdf_files:
        job_id = orchestrator.add_job(str(pdf_file), {"filename": pdf_file.name})
        job_ids.append(job_id)
        logger.info("Added pipeline job", job_id=job_id, filename=pdf_file.name)
    
    print(f"✅ Added {len(job_ids)} documents to processing pipeline")
    
    # Show pipeline summary
    summary = orchestrator.get_pipeline_summary()
    logger.info("Pipeline summary", **summary)
    
    print("\n📊 Pipeline Status:")
    print(f"   📋 Total jobs: {summary['total_jobs']}")
    for stage, count in summary['stage_counts'].items():
        if count > 0:
            print(f"   📌 {stage.title()}: {count}")
    
    return job_ids


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run FARA document processing pipeline")
    parser.add_argument('--test', action='store_true', help='Run test mode with dummy data')
    parser.add_argument('--batch-size', type=int, help='Batch size (currently ignored)')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting FARA pipeline")
        
        if args.test:
            # Original test mode
            orchestrator = PipelineOrchestrator()
            job_id = orchestrator.add_job("https://example.com/test.pdf", {"test": True})
            logger.info("Test job added", job_id=job_id)
            
            summary = orchestrator.get_pipeline_summary()
            logger.info("Pipeline summary", **summary)
            
            print("FARA pipeline test completed successfully")
            print(f"Test job ID: {job_id}")
            print("Pipeline summary:", summary)
        else:
            # Process real documents
            print("🚀 FARA Document Processing Pipeline")
            print("=" * 50)
            process_downloaded_documents()
            
            print(f"\n💡 Next Steps:")
            print(f"   • Run 'python3 scripts/process_ocr.py' to extract text from PDFs")
            print(f"   • Check processing status and logs in: {settings.logs}")
        
    except FARAError as e:
        logger.error("FARA pipeline error", error=str(e))
        print(f"❌ FARA pipeline error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
