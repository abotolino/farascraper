#!/usr/bin/env python3
"""Test the integrated FARA scraper."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.downloader.download_manager import DownloadManager
from src.common.logger import get_logger

logger = get_logger(__name__)

def main():
    """Test the scraper integration."""
    try:
        print("Testing FARA scraper integration...")
        
        downloader = DownloadManager()
        
        # First, let's try to get the list of available documents
        print("\nGetting list of available documents...")
        available_docs = downloader.get_available_documents()
        
        print(f"Found {len(available_docs)} documents waiting to be processed")
        
        if available_docs:
            # Show first few documents
            for i, doc in enumerate(available_docs[:5], 1):
                print(f"{i}. {doc['name']} (ID: {doc['document_id']})")
            
            if len(available_docs) > 5:
                print(f"... and {len(available_docs) - 5} more documents")
            
            # Ask if user wants to download all
            response = input("\nWould you like to download all waiting documents? (y/N): ")
            if response.lower().startswith('y'):
                print("\nStarting bulk download...")
                results = downloader.download_all_waiting_documents()
                
                print("\nDownload Results:")
                print(f"Total documents: {results['total_documents']}")
                print(f"Successful: {results['successful_downloads']}")
                print(f"Failed: {results['failed_downloads']}")
                print(f"Skipped: {results['skipped_downloads']}")
        else:
            print("No documents found. This might indicate:")
            print("1. Authentication failed")
            print("2. No documents are waiting to be processed")
            print("3. The website structure has changed")
            
    except Exception as e:
        logger.error("Test failed", error=str(e))
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
