# Try to import the full settings, fall back to simple if needed
try:
    from .settings import settings
except ImportError:
    from .simple_settings import settings

__all__ = ['settings']
