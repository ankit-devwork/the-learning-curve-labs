import os
from typing import Dict, Optional


class PropertyFileReader:
    """A clean utility to parse and read property configurations securely."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._properties: Dict[str, str] = {}
        self._load_properties()

    def _load_properties(self) -> None:
        """Parses the property file and loads key-value pairs into memory."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Property file not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                clean_line = line.strip()
                
                # Skip empty lines and comment structures (# or !)
                if not clean_line or clean_line.startswith(("#", "!")):
                    continue
                
                if "=" not in clean_line:
                    raise ValueError(f"Malformed configuration line {line_num}: missing '=' separator")

                key, value = clean_line.split("=", 1)
                
                # Strip spaces and optional wrapping quotes
                clean_key = key.strip()
                clean_value = value.strip().strip("'\"")
                
                self._properties[clean_key] = clean_value

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves a string property value, returning a default if not found."""
        return self._properties.get(key, default)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Retrieves a property value cast to an Integer."""
        val = self.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            raise TypeError(f"Property '{key}' value '{val}' cannot be converted to int.")

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Retrieves a property value cast to a Boolean."""
        val = self.get(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes", "on")