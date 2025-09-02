"""Custom exceptions for the FARA pipeline."""


class FARAError(Exception):
    """Base exception for FARA pipeline errors."""
    pass


class AuthenticationError(FARAError):
    """Raised when authentication fails."""
    pass


class DocumentNotFoundError(FARAError):
    """Raised when a document cannot be found or accessed."""
    pass


class OCRError(FARAError):
    """Raised when OCR processing fails."""
    pass


class ValidationError(FARAError):
    """Raised when data validation fails."""
    pass


class WebAutomationError(FARAError):
    """Raised when web automation encounters an error."""
    pass


class ConfigurationError(FARAError):
    """Raised when configuration is invalid or missing."""
    pass
