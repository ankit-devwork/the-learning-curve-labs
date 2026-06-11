from pathlib import Path

from dotenv import load_dotenv

from pycorekit.utils.env_overrides import apply_env_overrides
from pycorekit.utils.yaml_loader import load_yaml
from pycorekit.core_logging.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    Load YAML config with optional .env file support and env-var overrides.

    Load order (later wins for env overrides):
    1. YAML file
    2. Injected base_dir (if provided)
    3. Environment variables matching the override convention

    .env is loaded before overrides so secrets like GROQ_API_KEY are available
    via os.environ even when they are not part of config.yaml.
    """

    def __init__(
        self,
        path: Path,
        base_dir: Path | None = None,
        env_file: Path | None = None,
        env_prefix: str | None = None,
        dotenv_override: bool = False,
    ):
        self.path = Path(path)
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.env_file = Path(env_file) if env_file is not None else None
        self.env_prefix = env_prefix
        self.dotenv_override = dotenv_override

    def _load_dotenv(self) -> None:
        if self.env_file is None:
            return
        if not self.env_file.exists():
            logger.info(f"No .env file found at: {self.env_file}")
            return
        load_dotenv(self.env_file, override=self.dotenv_override)
        logger.info(f"Loaded environment from: {self.env_file}")

    def load(self) -> dict:
        """
        Load raw YAML from a path into a Python dictionary.

        Returns:
            dict: Parsed YAML data with optional env overrides applied.

        Example:
            config = ConfigLoader(
                Path("config.yaml"),
                base_dir=Path("."),
                env_file=Path(".env"),
            ).load()
        """
        self._load_dotenv()

        logger.info(f"Loading config from: {self.path}")
        data = load_yaml(self.path) or {}

        if self.base_dir is not None:
            paths = data.get("paths", {})
            paths["base_dir"] = str(self.base_dir)
            data["paths"] = paths

        data = apply_env_overrides(data, prefix=self.env_prefix)
        return data

    def load_typed(self, model: type):
        """
        Convert YAML dict into a typed Pydantic model.

        Args:
            model: Pydantic v2 model class.

        Returns:
            model: Validated typed config instance.

        Example:
            settings = ConfigLoader(
                Path("config.yaml"),
                base_dir=Path("."),
                env_file=Path(".env"),
                env_prefix="APP",
            ).load_typed(Settings)
        """
        data = self.load()
        logger.info("Config loaded, validating schema...")
        return model.model_validate(data)
