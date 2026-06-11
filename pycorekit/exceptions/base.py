class AppException(Exception):
    """
    Base application exception with HTTP status code.
    Optional error_type for structured error responses.
    """
    def __init__(self, message, status_code=400, error_type="APP_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)
