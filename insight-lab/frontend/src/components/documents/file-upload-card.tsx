"use client";

import Link from "next/link";
import { FileSpreadsheet, FileText, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  fetchUploadConfig,
  type DocumentSummary,
  type DocumentsResponse,
  type UploadConfigResponse,
  type UploadResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import { MultiDocChatPanel } from "@/components/documents/multi-doc-chat-panel";
import { FileListSkeleton } from "@/components/ui/loading-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

function formatDate(value: string): string {
  return new Date(value).toLocaleString();
}

function FileTypeIcon({ fileType }: { fileType: string }) {
  const isExcel = fileType === "excel";
  const Icon = isExcel ? FileSpreadsheet : FileText;
  return (
    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
      <Icon className="h-5 w-5" aria-hidden />
    </span>
  );
}

export function FileUploadCard() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploadConfig, setUploadConfig] = useState<UploadConfigResponse | null>(null);
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
    async function init() {
      try {
        const config = await fetchUploadConfig();
        setUploadConfig(config);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load upload settings");
        setLoading(false);
        return;
      }
      await loadDocuments();
    }
    void init();
  }, [loadDocuments]);

  async function handleUpload(file: File) {
    if (!uploadConfig) {
      setError("Upload settings not loaded yet");
      return;
    }

    if (file.size > uploadConfig.max_bytes) {
      setError(`File too large. Maximum size is ${uploadConfig.max_mb} MB`);
      return;
    }

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

  const description = uploadConfig
    ? `Allowed: ${uploadConfig.allowed_extensions.join(", ")} — up to ${uploadConfig.max_mb} MB`
    : null;

  return (
    <div className="space-y-6">
    <Card className="shadow-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-xl">Your files</CardTitle>
        <CardDescription>
          {description ? (
            <>
              {description}. Processing usually takes about a minute after upload.
            </>
          ) : (
            <span className="inline-flex items-center gap-2">
              <Skeleton className="inline-block h-4 w-48" />
            </span>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={inputRef}
            type="file"
            accept={uploadConfig?.accept}
            className="hidden"
            onChange={onFileChange}
            disabled={uploading || !uploadConfig}
          />
          <Button
            type="button"
            disabled={uploading || !uploadConfig}
            onClick={() => inputRef.current?.click()}
            className="gap-2"
            data-tour="upload-btn"
          >
            <Upload className="h-4 w-4" aria-hidden />
            {uploading ? "Uploading..." : "Upload file"}
          </Button>
          <Button type="button" variant="outline" disabled={loading} onClick={() => loadDocuments()}>
            Refresh
          </Button>
        </div>

        {error && (
          <p className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}
        {success && (
          <p className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-200">
            {success}
          </p>
        )}

        {loading ? (
          <div data-tour="file-list">
            <FileListSkeleton />
          </div>
        ) : documents.length === 0 ? (
          <div
            className="rounded-xl border border-dashed bg-muted/30 px-6 py-10 text-center"
            data-tour="file-list"
          >
            <Upload className="mx-auto h-8 w-8 text-muted-foreground/60" aria-hidden />
            <p className="mt-3 font-medium">No files yet</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Upload a PDF lecture note, Word doc, or spreadsheet to get started.
            </p>
          </div>
        ) : (
          <ul
            className="divide-y overflow-hidden rounded-xl border bg-card shadow-sm"
            data-tour="file-list"
          >
            {documents.map((doc) => (
              <li
                key={doc.id}
                className="flex flex-wrap items-center justify-between gap-3 px-4 py-3.5 transition-colors hover:bg-muted/40"
              >
                <div className="flex min-w-0 items-center gap-3">
                  <FileTypeIcon fileType={doc.file_type} />
                  <div className="min-w-0">
                    {doc.file_type === "document" ? (
                      <Link
                        href={`/dashboard/documents/${doc.id}`}
                        className="truncate font-medium text-foreground hover:text-primary hover:underline"
                      >
                        {doc.filename}
                      </Link>
                    ) : (
                      <Link
                        href={`/dashboard/excel/${doc.id}`}
                        className="truncate font-medium text-foreground hover:text-primary hover:underline"
                      >
                        {doc.filename}
                      </Link>
                    )}
                    <p className="text-xs text-muted-foreground">
                      {doc.file_type} · {formatDate(doc.created_at)}
                    </p>
                  </div>
                </div>
                <StatusBadge status={doc.status} />
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
    <MultiDocChatPanel documents={documents} />
    </div>
  );
}
