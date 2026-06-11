from pathlib import Path
from pycorekit.utils.yaml_loader import load_yaml
from pycorekit.logging.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    def __init__(self, path: Path, base_dir: Path | None = None):
        self.path = Path(path)
        self.base_dir = Path(base_dir) if base_dir is not None else None

    def load(self) -> dict:
        """
        Load raw YAML from a path into a Python dictionary.

        Returns:
            dict: Parsed YAML data.

        Example:
            config = ConfigLoader(Path("config.yaml"), base_dir=Path(".")).load()
        """
        logger.info(f"Loading config from: {self.path}")
        data = load_yaml(self.path) or {}

        # Inject base_dir into paths if provided
        if self.base_dir is not None:
            paths = data.get("paths", {})
            paths["base_dir"] = str(self.base_dir)
            data["paths"] = paths

        return data

    def load_typed(self, model: type):
        """
        Convert YAML dict into a typed Pydantic model.

        Args:
            model: Pydantic v2 model class.

        Returns:
            model: Validated typed config instance.

        Example:
            settings = ConfigLoader(Path("config.yaml"), base_dir=Path(".")).load_typed(Settings)
        """
        data = self.load()
        logger.info("Config loaded, validating schema...")
        return model.model_validate(data)
