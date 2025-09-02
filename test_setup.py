#!/usr/bin/env python3
"""Simple test to verify the setup is working."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    print("Testing FARA Pipeline Setup...")
    print()
    
    print("1. Testing settings import...")
    from config.settings import settings
    print("   Settings imported successfully")
    
    print("2. Testing logger import...")
    from src.common.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Test log message")
    print("   Logger working successfully")
    
    print("3. Testing exceptions import...")
    from src.common.exceptions import FARAError
    print("   Exceptions imported successfully")
    
    print("4. Testing orchestrator import...")
    from src.pipeline.orchestrator import PipelineOrchestrator
    print("   Orchestrator imported successfully")
    
    print("5. Testing orchestrator functionality...")
    orchestrator = PipelineOrchestrator()
    job_id = orchestrator.add_job("https://example.com/test.pdf")
    print(f"   Test job created: {job_id}")
    
    summary = orchestrator.get_pipeline_summary()
    print(f"   Pipeline summary: {summary}")
    
    print()
    print("All tests passed! Your FARA pipeline setup is working correctly.")
    print()
    print("Next steps:")
    print("1. Update your .env file with real FARA credentials")
    print("2. Add your existing scraper code to src/downloader/")
    print("3. Run: python scripts/run_pipeline.py")
    
except Exception as e:
    print(f"Error: {e}")
    print()
    print("Debugging information:")
    import traceback
    traceback.print_exc()
