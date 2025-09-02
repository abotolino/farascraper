import logging
import structlog
import sys
from pathlib import Path

# Add project root to path to find config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def configure_logging() -> None:
    """Configure structured logging for the application."""
    try:
        from config.settings import settings
        log_file = settings.log_file
        log_level = settings.log_level
        log_format = settings.log_format
    except ImportError:
        # Fallback if settings not available
        log_file = Path("data/logs/pipeline.log")
        log_level = "INFO"
        log_format = "text"
    
    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def get_logger(name: str):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Initialize logging when module is imported
configure_logging()
