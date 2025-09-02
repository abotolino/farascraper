import os
from pathlib import Path
from decouple import config


class SimpleSettings:
    """Simple settings without Pydantic dependencies"""
    
    def __init__(self):
        # FARA settings
        self.fara_username = config("FARA_USERNAME", default="your_username_here")
        self.fara_password = config("FARA_PASSWORD", default="your_password_here")
        self.fara_base_url = config("FARA_BASE_URL", default="http://fara.crphq.org")
        
        # OCR settings
        self.ocr_provider = config("OCR_PROVIDER", default="textract")
        self.ocr_confidence_threshold = config("OCR_CONFIDENCE_THRESHOLD", default=0.8, cast=float)
        
        # AWS settings
        self.aws_access_key_id = config("AWS_ACCESS_KEY_ID", default=None)
        self.aws_secret_access_key = config("AWS_SECRET_ACCESS_KEY", default=None)
        self.aws_region = config("AWS_DEFAULT_REGION", default="us-east-1")
        
        # Processing settings
        self.batch_size = config("BATCH_SIZE", default=10, cast=int)
        self.rate_limit_delay = config("RATE_LIMIT_DELAY", default=2.0, cast=float)
        self.max_retries = config("MAX_RETRIES", default=3, cast=int)
        
        # Logging settings
        self.log_level = config("LOG_LEVEL", default="INFO")
        self.log_format = config("LOG_FORMAT", default="json")
        self.log_file = Path(config("LOG_FILE", default="data/logs/pipeline.log"))
        
        # Paths
        self.data_root = Path("data")
        self.raw_documents = Path("data/raw/fara_documents")
        self.processed_data = Path("data/processed")
        self.logs = Path("data/logs")
        self.cache = Path("data/cache")
        self.backups = Path("data/backups")
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories"""
        for path in [self.data_root, self.raw_documents, self.processed_data, 
                     self.logs, self.cache, self.backups]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = SimpleSettings()
