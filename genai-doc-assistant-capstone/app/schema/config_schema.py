from pathlib import Path
from pydantic import BaseModel, field_validator, model_validator


class PathsConfig(BaseModel):
    base_dir: Path
    upload_dir: Path
    vector_db_path: Path
    log_dir: Path

    @field_validator("base_dir", "upload_dir", "vector_db_path", "log_dir", mode="before")
    def to_path(cls, v):
        return Path(v)

    @model_validator(mode="after")
    def resolve_absolute_paths(self):
        """
        Convert all relative paths into absolute paths using base_dir.
        """
        if not self.upload_dir.is_absolute():
            self.upload_dir = self.base_dir / self.upload_dir

        if not self.vector_db_path.is_absolute():
            self.vector_db_path = self.base_dir / self.vector_db_path

        if not self.log_dir.is_absolute():
            self.log_dir = self.base_dir / self.log_dir

        return self


class RagConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    splitter: str = "hybrid"  # options: "sliding", "recursive", "hybrid"
    quality_threshold:float = 0.4
    semantic_dedupe: bool = True
    document_selection_margin: float = 0.15


class FileUploadConfig(BaseModel):
    max_file_size_mb: int
    allowed_file_types: list[str]

class ModelConfig(BaseModel):
    embedding_model: str
    llm_model: str


class AppConfig(BaseModel):
    env: str
    paths: PathsConfig
    rag: RagConfig
    file_upload: FileUploadConfig
    models: ModelConfig
