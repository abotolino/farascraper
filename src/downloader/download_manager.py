from typing import Dict, Any, Optional, List
from pathlib import Path

from src.common.logger import get_logger
from src.common.exceptions import FARAError, AuthenticationError
from .fara_scraper import FARADocumentScraper
from config.settings import settings

logger = get_logger(__name__)


class DownloadManager:
    """Manages document downloads using the FARA scraper."""
    
    def __init__(self):
        self.scraper = FARADocumentScraper()
        logger.info("Download manager initialized")
    
    def download_document(self, document_url: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Download a single document - this method is called by the pipeline orchestrator"""
        try:
            # If this is a direct URL, handle it differently
            if document_url.startswith('http'):
                # This might be a direct S3 URL or document page URL
                return self._download_direct_url(document_url, metadata)
            else:
                # This might be a document ID or name
                return self._download_by_identifier(document_url, metadata)
                
        except Exception as e:
            logger.error("Download failed", url=document_url, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "url": document_url
            }
    
    def _download_direct_url(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle direct URL downloads"""
        # This is a simplified version - you might want to enhance this
        import requests
        from urllib.parse import urlparse
        
        try:
            response = self.scraper.session.get(url)
            response.raise_for_status()
            
            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "document.pdf"
            filepath = settings.raw_documents / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return {
                "success": True,
                "file_path": str(filepath),
                "file_size": len(response.content),
                "url": url,
                "metadata": metadata or {}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def _download_by_identifier(self, identifier: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Download by document identifier (name or ID)"""
        # Get list of waiting documents
        waiting_docs = self.scraper.get_waiting_documents_list()
        
        # Find the document by name or ID
        target_doc = None
        for doc in waiting_docs:
            if doc['name'] == identifier or doc['document_id'] == identifier:
                target_doc = doc
                break
        
        if not target_doc:
            return {
                "success": False,
                "error": f"Document not found: {identifier}"
            }
        
        # Download the found document
        return self.scraper.download_document(target_doc)
    
    def get_available_documents(self) -> List[Dict[str, Any]]:
        """Get list of all documents waiting to be processed"""
        try:
            return self.scraper.get_waiting_documents_list()
        except Exception as e:
            logger.error("Failed to get available documents", error=str(e))
            return []
    
    def download_waiting_documents(self, max_count: Optional[int] = None) -> Dict[str, Any]:
        """Download waiting documents with optional limit"""
        try:
            waiting_docs = self.scraper.get_waiting_documents_list()
            
            # Limit the number of documents if specified
            if max_count is not None and max_count > 0:
                waiting_docs = waiting_docs[:max_count]
                logger.info("Limited document list", requested=max_count, actual=len(waiting_docs))
            else:
                logger.info("Found waiting documents", count=len(waiting_docs))
            
            results = {
                "total_documents": len(waiting_docs),
                "successful_downloads": 0,
                "failed_downloads": 0,
                "skipped_downloads": 0,
                "results": []
            }
            
            for doc in waiting_docs:
                result = self.scraper.download_document(doc)
                results["results"].append(result)
                
                if result["success"]:
                    if result.get("skipped"):
                        results["skipped_downloads"] += 1
                    else:
                        results["successful_downloads"] += 1
                else:
                    results["failed_downloads"] += 1
                
                # Rate limiting
                import time
                time.sleep(settings.rate_limit_delay)
            
            return results
            
        except Exception as e:
            logger.error("Bulk download failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_all_waiting_documents(self) -> Dict[str, Any]:
        """Download all documents waiting to be processed (backward compatibility)"""
        return self.download_waiting_documents()
