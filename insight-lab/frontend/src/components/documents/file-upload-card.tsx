"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type DocumentSummary,
  type DocumentsResponse,
  type UploadResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const ACCEPT = ".xlsx,.xls,.csv,.pdf,.txt,.docx,.doc";

function formatDate(value: string): string {
  return new Date(value).toLocaleString();
}

function statusLabel(status: string): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

export function FileUploadCard() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    setError(null);
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (!session?.access_token) {
      setError("Sign in required to upload files");
      setLoading(false);
      return;
    }

    const response = await apiFetch("/documents", session.access_token);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Failed to load documents (${response.status})`);
      setLoading(false);
      return;
    }

    const data = (await response.json()) as DocumentsResponse;
    setDocuments(data.documents);
    setLoading(false);
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!session?.access_token) {
        setError("Sign in required to upload files");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      const response = await apiFetch("/upload", session.access_token, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || `Upload failed (${response.status})`);
        return;
      }

      const uploaded = (await response.json()) as UploadResponse;
      setSuccess(`Uploaded ${uploaded.filename}`);
      setDocuments((prev) => [
        {
          id: uploaded.id,
          filename: uploaded.filename,
          file_type: uploaded.file_type,
          mime_type: uploaded.mime_type,
          status: uploaded.status,
          created_at: uploaded.created_at,
        },
        ...prev.filter((doc) => doc.id !== uploaded.id),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    }
  }

  function onFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      void handleUpload(file);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload files</CardTitle>
        <CardDescription>
          Excel (.xlsx, .csv) or documents (.pdf, .txt, .docx) — up to 20 MB
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={onFileChange}
            disabled={uploading}
          />
          <Button type="button" disabled={uploading} onClick={() => inputRef.current?.click()}>
            {uploading ? "Uploading..." : "Choose file"}
          </Button>
          <Button type="button" variant="outline" disabled={loading} onClick={() => loadDocuments()}>
            Refresh
          </Button>
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}
        {success && <p className="text-sm text-green-600 dark:text-green-400">{success}</p>}

        {loading ? (
          <p className="text-sm text-muted-foreground">Loading your files...</p>
        ) : documents.length === 0 ? (
          <p className="text-sm text-muted-foreground">No uploads yet. Choose a file to get started.</p>
        ) : (
          <ul className="divide-y rounded-md border">
            {documents.map((doc) => (
              <li key={doc.id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 text-sm">
                <div>
                  <p className="font-medium">{doc.filename}</p>
                  <p className="text-muted-foreground">
                    {doc.file_type} · {formatDate(doc.created_at)}
                  </p>
                </div>
                <span className="rounded-full bg-muted px-2 py-1 text-xs capitalize">
                  {statusLabel(doc.status)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
