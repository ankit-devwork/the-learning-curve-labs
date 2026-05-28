class AppException(Exception):
    """Base class for all custom exceptions."""
    def __init__(self, message: str, status_code: int = 400, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ValidationException(AppException):
    def __init__(self, message="Validation failed", details=None):
        super().__init__(message, 422, details)


class NotFoundException(AppException):
    def __init__(self, message="Resource not found", details=None):
        super().__init__(message, 404, details)


class ExternalServiceException(AppException):
    def __init__(self, message="External service error", details=None):
        super().__init__(message, 502, details)

class DatabaseConnectionException(AppException):
    """Raised when PostgreSQL connection drops or fails."""
    def __init__(self, message="Database layer is currently unavailable", details=None):
        # Maps to a 503 Service Unavailable status code
        super().__init__(message, 503, details)


class RedisConnectionException(AppException):
    """Raised when Redis cache connection drops or fails."""
    def __init__(self, message="Cache service layer is currently unavailable", details=None):
        # Maps to a 503 Service Unavailable status code
        super().__init__(message, 503, details)

class Neo4jConnectionException(AppException):
    """Raised when the Neo4j driver cannot establish a socket connection or authenticate."""
    def __init__(self, message="Graph database service is currently unavailable", details=None):
        # Maps to a 503 Service Unavailable status code
        super().__init__(message, 503, details)


class GraphQueryException(AppException):
    """Raised when a Cypher query execution fails due to constraints or bad syntax."""
    def __init__(self, message="Failed to execute graph database transaction", details=None):
        # Maps to a 400 Bad Request if it's a query structural issue
        super().__init__(message, 400, details)