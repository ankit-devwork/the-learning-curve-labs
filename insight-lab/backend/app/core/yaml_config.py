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


class CacheSection(BaseModel):
    key_prefix: str = "insightlab"
    summary_ttl: int = 604800
    chat_ttl: int = 86400


class DocumentsSection(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_context_chunks: int = 6
    process_rate_limit_per_hour: int = 10
    chat_rate_limit_per_min: int = 20


class LlmSection(BaseModel):
    model: str = "groq/llama-3.3-70b-versatile"
    summary_max_tokens: int = 800
    chat_max_tokens: int = 600


class EmbeddingsSection(BaseModel):
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    batch_size: int = 32
    similarity_threshold: float = 0.35


class YamlConfig(BaseModel):
    app: AppSection
    logging: LoggingSection
    upload: UploadSection
    cache: CacheSection
    documents: DocumentsSection
    llm: LlmSection
    embeddings: EmbeddingsSection


@lru_cache(maxsize=1)
def get_yaml_config() -> YamlConfig:
    loader = ConfigLoader(
        CONFIG_PATH,
        base_dir=BACKEND_DIR,
        env_file=ENV_PATH,
        env_prefix="APP",
    )
    return loader.load_typed(YamlConfig)
