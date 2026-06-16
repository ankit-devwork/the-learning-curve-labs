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
