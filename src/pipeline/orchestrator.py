from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.logger import get_logger
from src.common.exceptions import FARAError
from config.settings import settings

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline stage enumeration."""
    DOWNLOAD = "download"
    OCR = "ocr"
    VALIDATION = "validation"
    AUTOMATION = "automation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineJob:
    """Represents a single pipeline job."""
    job_id: str
    document_url: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stage: PipelineStage = PipelineStage.DOWNLOAD
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None


class PipelineOrchestrator:
    """Main pipeline orchestrator that manages the complete workflow."""
    
    def __init__(self):
        self.state_file = settings.data_root / "pipeline_state.json"
        self.jobs: Dict[str, PipelineJob] = {}
        logger.info("Pipeline orchestrator initialized")
    
    def add_job(self, document_url: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new job to the pipeline."""
        job_id = str(uuid.uuid4())
        job = PipelineJob(
            job_id=job_id,
            document_url=document_url,
            metadata=metadata or {}
        )
        
        self.jobs[job_id] = job
        logger.info("New pipeline job added", job_id=job_id, document_url=document_url)
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        job = self.jobs.get(job_id)
        if job:
            return {
                "job_id": job.job_id,
                "document_url": job.document_url,
                "stage": job.stage.value,
                "created_at": job.created_at.isoformat(),
                "metadata": job.metadata,
                "results": job.results,
                "errors": job.errors
            }
        return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of all jobs in the pipeline."""
        stage_counts = {}
        for stage in PipelineStage:
            stage_counts[stage.value] = sum(1 for job in self.jobs.values() if job.stage == stage)
        
        return {
            "total_jobs": len(self.jobs),
            "stage_counts": stage_counts,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
