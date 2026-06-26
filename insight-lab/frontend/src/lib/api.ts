const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === "production" ? "/api-backend" : "http://localhost:8000");

export function getApiUrl(path: string): string {
  const base = API_URL.replace(/\/$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  // Relative proxy path (e.g. /api-backend) — same-origin on Vercel HTTPS
  if (base.startsWith("/")) {
    return `${base}${suffix}`;
  }
  return `${base}${suffix}`;
}

/** One ID per user action — send on every related API call so logs grep together. */
export function generateTrackingId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export type ApiFetchInit = RequestInit & {
  /** Propagate to backend as X-Tracking-ID (reuse for multi-call user actions). */
  trackingId?: string;
};

export async function apiFetch(
  path: string,
  accessToken: string,
  init: ApiFetchInit = {},
): Promise<Response> {
  const { trackingId, ...fetchInit } = init;
  const headers = new Headers(fetchInit.headers);
  headers.set("Authorization", `Bearer ${accessToken}`);
  if (trackingId) {
    headers.set("X-Tracking-ID", trackingId);
  }
  return fetch(getApiUrl(path), { ...fetchInit, headers });
}

/** Read X-Tracking-ID (or legacy x-correlation-id) from an API response. */
export function getTrackingIdFromResponse(response: Response): string | null {
  return (
    response.headers.get("X-Tracking-ID") ||
    response.headers.get("x-tracking-id") ||
    response.headers.get("X-Correlation-Id") ||
    response.headers.get("x-correlation-id")
  );
}

/** Extract a user-facing message from FastAPI / backend JSON error bodies. */
export function parseApiError(
  body: unknown,
  fallback = "Request failed",
  trackingId?: string | null,
): string {
  let message = fallback;
  if (body && typeof body === "object") {
    const record = body as Record<string, unknown>;
    if (typeof record.error === "string" && record.error.trim()) {
      message = record.error;
    } else if (typeof record.detail === "string" && record.detail.trim()) {
      message = record.detail;
    } else if (Array.isArray(record.detail)) {
      const parts = record.detail
        .map((item) => {
          if (typeof item === "string") {
            return item;
          }
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return null;
        })
        .filter(Boolean);
      if (parts.length > 0) {
        message = parts.join(". ");
      }
    } else if (typeof record.message === "string" && record.message.trim()) {
      message = record.message;
    }
  }

  const id =
    trackingId ||
    (body && typeof body === "object"
      ? (() => {
          const record = body as Record<string, unknown>;
          if (typeof record.tracking_id === "string" && record.tracking_id.trim()) {
            return record.tracking_id;
          }
          if (typeof record.correlation_id === "string" && record.correlation_id.trim()) {
            return record.correlation_id;
          }
          return null;
        })()
      : null);

  if (id) {
    return `${message} (Tracking ID: ${id})`;
  }
  return message;
}


export type DocumentSummary = {
  id: string;
  filename: string;
  file_type: "excel" | "document";
  mime_type: string | null;
  status: string;
  created_at: string;
};

export type DocumentDetail = DocumentSummary & {
  workspace_id?: string;
  summary?: string | null;
  error_message?: string | null;
  processed_at?: string | null;
};

export type SummaryResponse = {
  document_id: string;
  summary: string;
  status: string;
  cached?: boolean;
  correlation_id?: string;
};

export type AskResponse = {
  document_id: string;
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cited_chunks?: number[];
  retrieval_method?: "vector" | "keyword";
  chunk_similarities?: number[];
  cached?: boolean;
  cache_match?: string;
  similarity?: number;
  correlation_id?: string;
};

export type SourceCitation = {
  id: number;
  document_id: string;
  filename: string;
  chunk_index?: number;
  preview: string;
  similarity?: number | null;
  selected?: boolean;
};

export type QuizQuestion = {
  id: string;
  question_text: string;
  options: string[];
  sort_order: number;
};

export type QuizResponse = {
  quiz_id: string;
  document_id: string;
  title: string;
  question_type: "scq" | "mcq" | "true_false";
  difficulty: string;
  published?: boolean;
  public_share_token?: string;
  questions: QuizQuestion[];
  cached?: boolean;
  target_concepts?: ConceptMasteryItem[];
  correlation_id?: string;
};

export type QuizResultItem = {
  question_id: string;
  question_text: string;
  selected_option_index: number;
  correct_option_index: number;
  correct: boolean;
  explanation?: string | null;
  source_preview?: string | null;
  source_chunk_index?: number | null;
};

export type QuizSubmitResponse = {
  attempt_id?: string | null;
  quiz_id: string;
  document_id: string;
  score: number;
  total: number;
  percent: number;
  results: QuizResultItem[];
  correlation_id?: string;
};

export type ConceptMasteryItem = {
  concept_id: string;
  name: string;
  topic?: string | null;
  attempts: number;
  correct: number;
  percent?: number | null;
  last_attempt_at?: string | null;
  document_id?: string;
  document_filename?: string;
};

export type ConceptMasteryResponse = {
  document_id: string;
  concepts: ConceptMasteryItem[];
  migration_required?: boolean;
  notice?: string;
  correlation_id?: string;
};

export type GraphNodeMastery = {
  attempts: number;
  percent: number | null;
  status: "untested" | "needs_practice" | "strong";
};

export type GraphNode = {
  id: string;
  label: string;
  topic?: string | null;
  chunk_indexes?: number[];
  concept_id?: string;
  document_id?: string;
  document_filename?: string;
  mastery?: GraphNodeMastery;
};

export type GraphEdge = {
  source: string;
  target: string;
  type: string;
  document_id?: string;
};

export type DocumentGraphResponse = {
  document_id: string;
  filename: string;
  neo4j_synced_at?: string | null;
  nodes: GraphNode[];
  edges: GraphEdge[];
  migration_required?: boolean;
  notice?: string;
  correlation_id?: string;
};

export type WorkspaceGraphDocument = {
  document_id: string;
  filename: string;
  concept_count: number;
  relationship_count: number;
  neo4j_synced_at?: string | null;
};

export type WorkspaceGraphResponse = {
  workspace_id: string;
  documents: WorkspaceGraphDocument[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    document_count: number;
    node_count: number;
    edge_count: number;
    topic_count: number;
  };
  migration_required?: boolean;
  notice?: string;
  correlation_id?: string;
};

export type GraphSyncResponse = {
  document_id: string;
  concept_count: number;
  relationship_count: number;
  neo4j_synced: boolean;
  synced_at: string;
  cached?: boolean;
  correlation_id?: string;
};

export type TeamChatMessage = {
  id: string;
  workspace_id: string;
  author_id: string;
  author_name: string;
  body: string;
  created_at: string;
  is_own: boolean;
};

export type WorkspaceMessagesResponse = {
  workspace_id: string;
  messages: TeamChatMessage[];
  has_more: boolean;
  next_before?: string | null;
  migration_required?: boolean;
  notice?: string;
  correlation_id?: string;
};

export type DocumentReviewOption = {
  document_id: string;
  filename: string;
  summary: string;
  selected?: boolean;
};

export type MultiRetrieveResponse = {
  document_ids: string[];
  question: string;
  documents: DocumentReviewOption[];
  hitl_required?: boolean;
  correlation_id?: string;
};

export type MultiAskResponse = {
  document_ids: string[];
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cited_documents?: string[];
  retrieval_method?: "vector" | "keyword";
  cached?: boolean;
  cache_match?: string;
  similarity?: number;
  correlation_id?: string;
};

export type GenerateQuizRequest = {
  question_type?: "scq" | "mcq" | "true_false";
  difficulty?: "easy" | "medium" | "hard";
  num_questions?: number;
};

export type ExcelChart = {
  id: string;
  title: string;
  chart_type: "bar" | "line" | "pie" | "scatter";
  x_column: string;
  y_column?: string | null;
  aggregation?: string;
  labels: string[];
  values: number[];
  custom?: boolean;
};

export type CustomChartRequest = {
  chart_type: ExcelChart["chart_type"];
  x_column: string;
  y_column?: string | null;
  aggregation?: "sum" | "mean" | "count" | "none";
  title?: string | null;
};

export type CustomChartResponse = {
  chart: ExcelChart;
  correlation_id?: string;
};

export type ExcelAnalysisResponse = {
  document_id: string;
  status: string;
  profile: {
    row_count: number;
    column_count: number;
    columns: Array<{
      name: string;
      dtype: string;
      null_pct: number;
      unique_count: number;
      sample_values?: string[];
    }>;
  };
  charts: ExcelChart[];
  summary: string;
  cached?: boolean;
  correlation_id?: string;
};

export type ExcelAskResponse = {
  document_id: string;
  question: string;
  answer: string;
  sources?: string[];
  document_citations?: SourceCitation[];
  cached?: boolean;
  cache_match?: string;
  similarity?: number;
  correlation_id?: string;
};

export type UploadResponse = DocumentSummary & {
  correlation_id?: string;
};

export type DocumentsResponse = {
  documents: DocumentSummary[];
  count: number;
  correlation_id?: string;
};

export type UploadGuidanceConfig = {
  summary: string;
  points: string[];
  require_acknowledgment: boolean;
};

export type UploadConfigResponse = {
  max_bytes: number;
  max_mb: number;
  accept: string;
  allowed_extensions: string[];
  excel_extensions: string[];
  document_extensions: string[];
  guidance?: UploadGuidanceConfig;
  storage_encrypted?: boolean;
};

export async function fetchUploadConfig(): Promise<UploadConfigResponse> {
  const response = await fetch(getApiUrl("/upload/config"));
  if (!response.ok) {
    throw new Error(`Failed to load upload config (${response.status})`);
  }
  return response.json();
}

export type WorkspaceSummary = {
  id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  access_role?: "owner" | "editor" | "viewer" | null;
  is_owner?: boolean;
  shared?: boolean;
};

export type QuizQuestionEditable = QuizQuestion & {
  correct_option_index?: number;
  explanation?: string | null;
};

export type CoursePackDocumentResult = {
  document_id: string;
  filename: string;
  file_type?: "document" | "excel";
  artifacts: {
    summary?: string;
    quiz_id?: string;
    flashcard_set_id?: string;
    study_guide_id?: string;
    audio_title?: string;
    audio_script?: string;
    infographic_id?: string;
    slide_deck_id?: string;
    homework_solution_id?: string;
    homework_question?: string;
    chart_count?: number;
  };
  errors: string[];
};

export type CoursePackResponse = {
  workspace_id: string;
  document_count: number;
  documents: CoursePackDocumentResult[];
  correlation_id?: string;
};

export type WorkspaceStats = {
  workspace_id: string;
  document_count: number;
  ready_count: number;
  document_files: number;
  excel_files: number;
  quiz_attempts: number;
  avg_quiz_percent: number | null;
};

export type WorkspaceProgress = WorkspaceStats & {
  flashcard_reviews: number;
  mastery_avg_percent: number | null;
  weak_concepts: ConceptMasteryItem[];
  study_next: { action: string; label: string };
};

export type ClassroomMemberStats = {
  user_id: string;
  email?: string | null;
  full_name?: string | null;
  role: string;
  joined_at?: string;
  quiz_attempts: number;
  avg_quiz_percent: number | null;
  flashcard_reviews: number;
  flashcard_knew_percent: number | null;
  mastery_avg_percent: number | null;
  weak_topic_count: number;
  last_activity_at: string | null;
};

export type ClassroomAnalytics = {
  workspace_id: string;
  member_count: number;
  ready_documents: number;
  class_avg_quiz_percent: number | null;
  public_quiz_attempts: number;
  public_quiz_avg_percent: number | null;
  members: ClassroomMemberStats[];
  correlation_id?: string;
};

export type StudySessionStep = {
  step: "brief" | "flashcards" | "quiz";
  label: string;
  ready: boolean;
  duration_min: number;
  set_id?: string | null;
  quiz_id?: string | null;
  published?: boolean;
  generate_if_missing?: boolean;
};

export type StudySessionPlan = {
  document_id: string;
  filename: string;
  steps: StudySessionStep[];
  weak_concepts: ConceptMasteryItem[];
  focus_topic: string | null;
  estimated_minutes: number;
};

export type WorkspaceStudySessionStep =
  | {
      step: "focus";
      label: string;
      ready: boolean;
      duration_min: number;
      weak_concepts: ConceptMasteryItem[];
      focus_topic: string | null;
    }
  | {
      step: "brief";
      label: string;
      ready: boolean;
      duration_min: number;
      document_id: string;
      filename: string;
    }
  | {
      step: "flashcards";
      label: string;
      ready: boolean;
      duration_min: number;
      document_id: string;
      filename: string;
      set_id?: string | null;
      generate_if_missing?: boolean;
    }
  | {
      step: "adaptive_quiz";
      label: string;
      ready: boolean;
      duration_min: number;
      weak_count: number;
    }
  | {
      step: "set_quiz";
      label: string;
      ready: boolean;
      duration_min: number;
      hint?: string;
    };

export type WorkspaceStudySessionPlan = {
  workspace_id: string;
  document_count: number;
  steps: WorkspaceStudySessionStep[];
  weak_concepts: ConceptMasteryItem[];
  focus_topic: string | null;
  estimated_minutes: number;
  correlation_id?: string;
};

export type StudySessionStepProgress = {
  id: string;
  step_index: number;
  step_type: string;
  label: string;
  payload: Record<string, unknown>;
  status: "pending" | "in_progress" | "completed" | "skipped";
  started_at?: string | null;
  completed_at?: string | null;
};

export type StudySessionRecord = {
  session_id: string;
  session_type: "document" | "workspace";
  workspace_id?: string | null;
  document_id?: string | null;
  learning_path_id?: string | null;
  status: "active" | "completed" | "abandoned";
  current_step_index: number;
  progress: {
    completed_steps: number;
    total_steps: number;
    percent: number;
  };
  plan: WorkspaceStudySessionPlan | StudySessionPlan | Record<string, unknown>;
  steps: StudySessionStepProgress[];
};

export type LearningPathNode = {
  id: string;
  sort_order: number;
  node_kind: "concept" | "document";
  document_id?: string;
  document_filename?: string;
  concept_id?: string;
  concept_name?: string;
  topic?: string | null;
  status: "available" | "needs_practice" | "completed" | "locked";
  mastery_percent?: number | null;
};

export type LearningPathResponse = {
  workspace_id: string;
  path_id: string | null;
  title: string;
  node_count?: number;
  nodes: LearningPathNode[];
  migration_required?: boolean;
  notice?: string;
  correlation_id?: string;
};

export type SourceLink = {
  id: string;
  excel_document_id: string;
  document_id: string;
  label?: string | null;
  excel_filename?: string;
  document_filename?: string;
  created_at?: string;
};

export type PublicQuizResponse = QuizResponse & {
  source_filename?: string | null;
  public_share_token?: string;
};

export type ProcessingStatus = {
  document_id: string;
  status: string;
  stage: string;
  progress_pct: number;
  message: string;
};

export type FlashcardItem = {
  id: string;
  front: string;
  back: string;
  sort_order: number;
  source_chunk_index?: number | null;
};

export type FlashcardSetResponse = {
  set_id: string;
  document_id: string;
  title: string;
  cards: FlashcardItem[];
  card_count: number;
  due_count?: number;
  due_ids?: string[];
};

export type ExplainResponse = {
  explanation: string;
  sources?: SourceCitation[];
  kind?: string;
};

export type ChatHistoryMessage = {
  id?: string;
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cached?: boolean;
  created_at?: string;
};

export type HomeworkStep = {
  title: string;
  detail: string;
};

export type HomeworkSolutionResponse = {
  solution_id?: string;
  document_id: string;
  question: string;
  steps: HomeworkStep[];
  summary?: string;
  sources?: SourceCitation[];
  disclaimer?: string;
};

export type SlideDeckSlide = {
  slide_number: number;
  title: string;
  bullets: string[];
  speaker_notes?: string;
};

export type SlideDeckContent = {
  title: string;
  slides: SlideDeckSlide[];
};

export type SlideDeckResponse = {
  slide_deck_id: string;
  document_id: string;
  title: string;
  content: SlideDeckContent;
  created_at?: string;
};

export type StudyGuideContent = {
  title: string;
  overview: string;
  key_terms: Array<{ term: string; definition: string }>;
  sections: Array<{ heading: string; bullets: string[] }>;
  sample_questions: string[];
};

export type StudyGuideResponse = {
  guide_id: string;
  document_id: string;
  title: string;
  content: StudyGuideContent;
  created_at: string;
};

export type InfographicStatBlock = {
  type: "stat";
  label: string;
  value: string;
  caption?: string;
};

export type InfographicBulletsBlock = {
  type: "bullets";
  heading: string;
  items: string[];
};

export type InfographicComparisonBlock = {
  type: "comparison";
  heading: string;
  left_title: string;
  left_items: string[];
  right_title: string;
  right_items: string[];
};

export type InfographicQuoteBlock = {
  type: "quote";
  text: string;
  attribution?: string;
};

export type InfographicBlock =
  | InfographicStatBlock
  | InfographicBulletsBlock
  | InfographicComparisonBlock
  | InfographicQuoteBlock;

export type InfographicContent = {
  title: string;
  subtitle: string;
  theme: "blue" | "violet" | "emerald" | "amber" | "rose" | "cyan";
  blocks: InfographicBlock[];
};

export type InfographicResponse = {
  infographic_id: string;
  document_id: string;
  title: string;
  content: InfographicContent;
  created_at: string;
};

export type DocumentChunkResponse = {
  document_id: string;
  chunk_index: number;
  content: string;
  preview: string;
  truncated: boolean;
};

export type AudioOverviewResponse = {
  document_id: string;
  overview_id?: string;
  title: string;
  script: string;
  estimated_minutes?: number;
  has_audio?: boolean;
  cached?: boolean;
  correlation_id?: string;
};

export type ExcelPreviewResponse = {
  document_id: string;
  columns: string[];
  rows: string[][];
  preview_rows: number;
  total_rows: number;
  total_columns: number;
  correlation_id?: string;
};
