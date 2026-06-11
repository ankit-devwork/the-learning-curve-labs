from pycorekit.exceptions.base import AppException


class FileException(AppException):
    """
    Exception for file-related errors:
    - invalid file type
    - file too large
    - upload failure
    - missing file
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="FILE_ERROR"
        )
