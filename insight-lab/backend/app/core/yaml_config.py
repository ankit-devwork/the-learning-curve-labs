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


class UploadGuidanceSection(BaseModel):
    summary: str = (
        "Only upload study materials you are comfortable sharing with everyone on this study sheet."
    )
    points: list[str] = Field(
        default_factory=lambda: [
            "Do not upload medical records, financial statements, government IDs, or passwords.",
            "Do not upload copyrighted or confidential content you do not have permission to share.",
            "Uploaded files can be viewed by study sheet members and are stored until you delete them.",
        ]
    )
    require_acknowledgment: bool = True


class UploadSection(BaseModel):
    storage_bucket: str
    max_bytes: int
    rate_limit_per_hour: int
    filename_max_length: int
    default_workspace_name: str
    documents_list_default_limit: int = 20
    delete_rate_limit_per_hour: int = 20
    excel: UploadTypeConfig
    document: UploadTypeConfig
    guidance: UploadGuidanceSection = Field(default_factory=UploadGuidanceSection)

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
    excel_ttl: int = 86400
    quiz_ttl: int = 604800
    artifact_ttl: int = 604800
    semantic_threshold: float = 0.85
    semantic_max_entries: int = 50


class CoursePackSection(BaseModel):
    generate_rate_limit_per_hour: int = 3


class SharingSection(BaseModel):
    invite_preview_rate_limit_per_min: int = 15
    invite_accept_rate_limit_per_min: int = 5
    invite_create_rate_limit_per_min: int = 10
    member_change_rate_limit_per_min: int = 20


class TeamChatSection(BaseModel):
    message_max_length: int = 2000
    list_page_size: int = 50
    post_rate_limit_per_min: int = 30
    list_rate_limit_per_min: int = 120
    delete_rate_limit_per_min: int = 30
    inbox_rate_limit_per_min: int = 60
    mark_read_rate_limit_per_min: int = 120
    typing_rate_limit_per_min: int = 120
    typing_ttl_seconds: int = 5


class EmailSection(BaseModel):
    enabled: bool = False
    from_address: str = "InsightLab <onboarding@resend.dev>"
    frontend_base_url: str = "http://localhost:3000"
    resend_api_key_env: str = "RESEND_API_KEY"
    notify_artifact_ready: bool = False


class ExcelSection(BaseModel):
    analyze_rate_limit_per_min: int = 10
    chat_rate_limit_per_min: int = 20
    max_rows: int = 50000
    max_columns: int = 100
    sample_rows_for_llm: int = 50
    max_charts: int = 6
    chart_plan_max_tokens: int = 1200
    summary_max_tokens: int = 800
    chat_max_tokens: int = 600
    chart_context_points: int = 25


class QuizzesSection(BaseModel):
    generate_rate_limit_per_min: int = 5
    submit_rate_limit_per_min: int = 30
    public_get_rate_limit_per_min: int = 30
    public_submit_rate_limit_per_min: int = 10
    public_max_answers_bytes: int = 4096
    default_num_questions: int = 5
    max_questions: int = 10
    max_context_chunks: int = 8
    quiz_max_tokens: int = 2000


class GraphSection(BaseModel):
    sync_rate_limit_per_hour: int = 10
    max_concepts_per_document: int = 40
    concept_extract_max_tokens: int = 1500
    cache_ttl: int = 604800


class AdaptiveQuizSection(BaseModel):
    generate_rate_limit_per_min: int = 5
    weak_threshold_percent: int = 60
    min_attempts_before_adaptive: int = 1
    max_weak_concepts: int = 5


class MultiDocSection(BaseModel):
    chat_rate_limit_per_min: int = 20
    max_documents: int = 10
    max_context_chunks: int = 6
    max_chunks_per_document: int = 3
    chat_max_tokens: int = 800


class ResilienceSection(BaseModel):
    retry_max_attempts: int = 4
    retry_base_delay_sec: float = 1.0
    retry_max_delay_sec: float = 30.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_sec: float = 60.0


class DocumentsSection(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_context_chunks: int = 6
    process_rate_limit_per_hour: int = 10
    chat_rate_limit_per_min: int = 20
    delete_rate_limit_per_hour: int = 20


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


class ArtifactsSection(BaseModel):
    generate_rate_limit_per_min: int = 5
    review_rate_limit_per_min: int = 120
    max_flashcards: int = 20
    max_context_chunks: int = 8
    max_slides: int = 12
    flashcard_max_tokens: int = 2000
    study_guide_max_tokens: int = 2500
    infographic_max_tokens: int = 2500
    slide_deck_max_tokens: int = 2800
    suggested_questions_max_tokens: int = 400
    chunk_preview_max_chars: int = 1200


class ExplainSection(BaseModel):
    rate_limit_per_min: int = 15
    max_tokens: int = 900


class HomeworkSection(BaseModel):
    rate_limit_per_min: int = 10
    max_tokens: int = 1800


class AudioOverviewSection(BaseModel):
    generate_rate_limit_per_min: int = 5
    max_tokens: int = 1200
    target_words: int = 450
    tts_enabled: bool = True


class ExcelPreviewSection(BaseModel):
    default_limit: int = 50
    max_limit: int = 200


class YamlConfig(BaseModel):
    app: AppSection
    logging: LoggingSection
    upload: UploadSection
    cache: CacheSection
    documents: DocumentsSection
    llm: LlmSection
    embeddings: EmbeddingsSection
    excel: ExcelSection
    quizzes: QuizzesSection
    graph: GraphSection
    adaptive_quiz: AdaptiveQuizSection
    multi_doc: MultiDocSection
    artifacts: ArtifactsSection
    audio_overview: AudioOverviewSection
    explain: ExplainSection
    homework: HomeworkSection
    excel_preview: ExcelPreviewSection
    course_pack: CoursePackSection
    sharing: SharingSection
    team_chat: TeamChatSection
    email: EmailSection
    resilience: ResilienceSection


@lru_cache(maxsize=1)
def get_yaml_config() -> YamlConfig:
    loader = ConfigLoader(
        CONFIG_PATH,
        base_dir=BACKEND_DIR,
        env_file=ENV_PATH,
        env_prefix="APP",
    )
    return loader.load_typed(YamlConfig)
