#!/usr/bin/env python3
"""
Main script to run the FARA document processing pipeline.
"""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.common.logger import get_logger
from src.common.exceptions import FARAError

logger = get_logger(__name__)


def main():
    """Main entry point."""
    try:
        logger.info("Starting FARA pipeline")
        
        orchestrator = PipelineOrchestrator()
        
        # Add a test job
        job_id = orchestrator.add_job("https://example.com/test.pdf", {"test": True})
        logger.info("Test job added", job_id=job_id)
        
        # Show pipeline summary
        summary = orchestrator.get_pipeline_summary()
        logger.info("Pipeline summary", **summary)
        
        print("🎉 FARA pipeline test completed successfully")
        print(f"Test job ID: {job_id}")
        print("Pipeline summary:", summary)
        
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
