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

export type UploadResponse = DocumentSummary & {
  storage_path: string;
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
