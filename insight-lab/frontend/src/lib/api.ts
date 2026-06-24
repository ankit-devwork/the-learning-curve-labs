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

export async function apiFetch(
  path: string,
  accessToken: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${accessToken}`);
  return fetch(getApiUrl(path), { ...init, headers });
}

/** Extract a user-facing message from FastAPI / backend JSON error bodies. */
export function parseApiError(body: unknown, fallback = "Request failed"): string {
  if (!body || typeof body !== "object") {
    return fallback;
  }
  const record = body as Record<string, unknown>;
  if (typeof record.error === "string" && record.error.trim()) {
    return record.error;
  }
  if (typeof record.detail === "string" && record.detail.trim()) {
    return record.detail;
  }
  if (Array.isArray(record.detail)) {
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
      return parts.join(". ");
    }
  }
  if (record.message && typeof record.message === "string") {
    return record.message;
  }
  return fallback;
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

export type GraphNode = {
  id: string;
  label: string;
  topic?: string | null;
  chunk_indexes?: number[];
};

export type GraphEdge = {
  source: string;
  target: string;
  type: string;
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

export type GraphSyncResponse = {
  document_id: string;
  concept_count: number;
  relationship_count: number;
  neo4j_synced: boolean;
  synced_at: string;
  cached?: boolean;
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

export type UploadConfigResponse = {
  max_bytes: number;
  max_mb: number;
  accept: string;
  allowed_extensions: string[];
  excel_extensions: string[];
  document_extensions: string[];
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
  artifacts: {
    summary?: string;
    quiz_id?: string;
    flashcard_set_id?: string;
    study_guide_id?: string;
    audio_title?: string;
    audio_script?: string;
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

export type DocumentChunkResponse = {
  document_id: string;
  chunk_index: number;
  content: string;
  preview: string;
  truncated: boolean;
};

export type AudioOverviewResponse = {
  document_id: string;
  title: string;
  script: string;
  estimated_minutes?: number;
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
