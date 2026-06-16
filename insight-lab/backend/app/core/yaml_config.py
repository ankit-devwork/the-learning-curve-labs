from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pycorekit.utils.config_loader import ConfigLoader

BACKEND_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BACKEND_DIR / "config.yaml"
ENV_PATH = BACKEND_DIR / ".env"


class AppSection(BaseModel):
    env: str = "development"
    debug: bool = True
    name: str = "InsightLab API"


class LoggingSection(BaseModel):
    dir: str = "logs"


class UploadTypeConfig(BaseModel):
    extensions: list[str]
    mime_types: list[str] = Field(default_factory=list)
    signatures: dict[str, str] = Field(default_factory=dict)


class UploadSection(BaseModel):
    storage_bucket: str
    max_bytes: int
    rate_limit_per_hour: int
    filename_max_length: int
    default_workspace_name: str
    documents_list_default_limit: int = 20
    excel: UploadTypeConfig
    document: UploadTypeConfig

    def all_extensions(self) -> set[str]:
        return {ext.lower() for ext in self.excel.extensions + self.document.extensions}

    def file_type_for_extension(self, extension: str) -> str | None:
        ext = extension.lower()
        if ext in {e.lower() for e in self.excel.extensions}:
            return "excel"
        if ext in {e.lower() for e in self.document.extensions}:
            return "document"
        return None

    def type_config(self, file_type: str) -> UploadTypeConfig:
        if file_type == "excel":
            return self.excel
        if file_type == "document":
            return self.document
        raise ValueError(f"Unknown file type: {file_type}")

    def accept_attribute(self) -> str:
        return ",".join(sorted(self.all_extensions()))


class YamlConfig(BaseModel):
    app: AppSection
    logging: LoggingSection
    upload: UploadSection


@lru_cache(maxsize=1)
def get_yaml_config() -> YamlConfig:
    loader = ConfigLoader(
        CONFIG_PATH,
        base_dir=BACKEND_DIR,
        env_file=ENV_PATH,
        env_prefix="APP",
    )
    return loader.load_typed(YamlConfig)
