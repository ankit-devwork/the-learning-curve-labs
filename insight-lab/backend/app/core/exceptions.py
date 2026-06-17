from pycorekit.exceptions.base import AppException


class UnauthorizedException(AppException):
    def __init__(self, message: str = "Invalid or missing authentication token"):
        super().__init__(message, status_code=401, error_type="UNAUTHORIZED")


class ForbiddenException(AppException):
    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, status_code=403, error_type="FORBIDDEN")


class ServiceUnavailableException(AppException):
    def __init__(self, message: str = "Service unavailable"):
        super().__init__(message, status_code=503, error_type="SERVICE_UNAVAILABLE")


class NotFoundException(AppException):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404, error_type="NOT_FOUND")


class RateLimitException(AppException):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message, status_code=429, error_type="RATE_LIMIT")
        self.retry_after = retry_after
