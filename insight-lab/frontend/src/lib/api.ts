const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function getApiUrl(path: string): string {
  return `${API_URL.replace(/\/$/, "")}${path.startsWith("/") ? path : `/${path}`}`;
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

export type DocumentSummary = {
  id: string;
  filename: string;
  file_type: "excel" | "document";
  mime_type: string | null;
  status: string;
  created_at: string;
};

export type DocumentDetail = DocumentSummary & {
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
  cited_chunks: number[];
  retrieval_method?: "vector" | "keyword";
  chunk_similarities?: number[];
  cached?: boolean;
  correlation_id?: string;
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
};

export type ConceptMasteryResponse = {
  document_id: string;
  concepts: ConceptMasteryItem[];
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

export type MultiAskResponse = {
  document_ids: string[];
  question: string;
  answer: string;
  cited_documents?: string[];
  retrieval_method?: "vector" | "keyword";
  cached?: boolean;
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
