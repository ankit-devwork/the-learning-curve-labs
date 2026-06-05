class AppException(Exception):
    """
    Base application exception with HTTP status code.
    """
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)
