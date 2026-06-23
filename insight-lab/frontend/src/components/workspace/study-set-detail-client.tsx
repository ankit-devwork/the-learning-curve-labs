"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type DocumentSummary,
  type WorkspaceStats,
  type WorkspaceSummary,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import { MultiDocChatPanel } from "@/components/documents/multi-doc-chat-panel";
import { UploadDropzone } from "@/components/workspace/upload-dropzone";
import { WorkspaceStatsPanel } from "@/components/workspace/workspace-stats-panel";
import { CoursePackPanel } from "@/components/workspace/course-pack-panel";
import { ShareWorkspacePanel } from "@/components/workspace/share-workspace-panel";
import { SetQuizPanel } from "@/components/workspace/set-quiz-panel";
import { useToast } from "@/components/ui/toast";
import { fetchUploadConfig, type UploadConfigResponse } from "@/lib/api";
import { FileSpreadsheet, FileText } from "lucide-react";
import { canEditWorkspace, workspaceRoleLabel } from "@/lib/workspace-roles";

function FileTypeIcon({ fileType }: { fileType: string }) {
  const Icon = fileType === "excel" ? FileSpreadsheet : FileText;
  return (
    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
      <Icon className="h-4 w-4" aria-hidden />
    </span>
  );
}

export function StudySetDetailClient({ setId }: { setId: string }) {
  const { toast } = useToast();
  const [workspace, setWorkspace] = useState<WorkspaceSummary | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [uploadConfig, setUploadConfig] = useState<UploadConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    setError(null);
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setError("Sign in required");
      setLoading(false);
      return;
    }
    setAccessToken(session.access_token);

    const [workspaceRes, docsRes, statsRes] = await Promise.all([
      apiFetch(`/workspaces/${setId}`, session.access_token),
      apiFetch(`/workspaces/${setId}/documents`, session.access_token),
      apiFetch(`/workspaces/${setId}/stats`, session.access_token),
    ]);

    if (!workspaceRes.ok || !docsRes.ok || !statsRes.ok) {
      setError("Failed to load study set");
      setLoading(false);
      return;
    }

    const workspaceData = await workspaceRes.json();
    const docsData = await docsRes.json();
    const statsData = await statsRes.json();
    setWorkspace(workspaceData);
    setDocuments(docsData.documents ?? []);
    setStats(statsData);
    setLoading(false);
  }, [setId]);

  useEffect(() => {
    async function init() {
      try {
        const config = await fetchUploadConfig();
        setUploadConfig(config);
      } catch {
        setError("Failed to load upload settings");
        setLoading(false);
        return;
      }
      await loadAll();
    }
    void init();
  }, [loadAll]);

  async function handleUpload(file: File) {
    if (!uploadConfig) {
      return;
    }
    if (file.size > uploadConfig.max_bytes) {
      toast({
        title: "File too large",
        description: `Maximum size is ${uploadConfig.max_mb} MB`,
        variant: "error",
      });
      return;
    }

    setUploading(true);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        return;
      }

      const formData = new FormData();
      formData.append("file", file);
      const response = await apiFetch(
        `/upload?workspace_id=${encodeURIComponent(setId)}`,
        session.access_token,
        { method: "POST", body: formData },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Upload failed",
          description: body.error || `Error ${response.status}`,
          variant: "error",
        });
        return;
      }
      toast({ title: "File uploaded", description: file.name, variant: "success" });
      await loadAll();
    } finally {
      setUploading(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading study set…</p>;
  }

  if (!workspace) {
    return <p className="text-sm text-destructive">{error || "Study set not found"}</p>;
  }

  const canEdit = canEditWorkspace(workspace.access_role);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/dashboard/sets" className="text-sm text-muted-foreground hover:text-primary">
            ← All study sets
          </Link>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">{workspace.name}</h1>
          {workspace.access_role ? (
            <p className="mt-1 text-sm text-muted-foreground">
              Your role: {workspaceRoleLabel(workspace.access_role)}
              {workspace.shared ? " · Shared with you" : ""}
            </p>
          ) : null}
          {workspace.description ? (
            <p className="mt-1 max-w-2xl text-muted-foreground">{workspace.description}</p>
          ) : null}
        </div>
        <Button type="button" variant="outline" onClick={() => void loadAll()}>
          Refresh
        </Button>
      </div>

      {stats ? <WorkspaceStatsPanel stats={stats} /> : null}

      <CoursePackPanel setId={setId} canEdit={canEdit} />

      <ShareWorkspacePanel
        setId={setId}
        canManage={canEdit}
        isOwner={workspace.access_role === "owner" || Boolean(workspace.is_owner)}
      />

      {canEdit ? (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Upload materials</CardTitle>
          <CardDescription>
            {uploadConfig
              ? `Allowed: ${uploadConfig.allowed_extensions.join(", ")} — up to ${uploadConfig.max_mb} MB`
              : "Loading upload settings…"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <UploadDropzone
            accept={uploadConfig?.accept}
            disabled={!uploadConfig}
            uploading={uploading}
            onFileSelected={(file) => void handleUpload(file)}
          />
        </CardContent>
      </Card>
      ) : (
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle>Upload materials</CardTitle>
            <CardDescription>Viewer access — you can read and study files but not upload.</CardDescription>
          </CardHeader>
        </Card>
      )}

      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Files in this set</CardTitle>
          <CardDescription>Open a ready file to use chat, quiz, flashcards, and the studio tools.</CardDescription>
        </CardHeader>
        <CardContent>
          {documents.length === 0 ? (
            <p className="text-sm text-muted-foreground">No files yet — upload your first material above.</p>
          ) : (
            <ul className="divide-y overflow-hidden rounded-xl border" data-tour="file-list">
              {documents.map((doc) => (
                <li key={doc.id} className="flex items-center justify-between gap-3 px-4 py-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <FileTypeIcon fileType={doc.file_type} />
                    <div className="min-w-0">
                      <Link
                        href={
                          doc.file_type === "document"
                            ? `/dashboard/sets/${setId}/documents/${doc.id}`
                            : `/dashboard/sets/${setId}/excel/${doc.id}`
                        }
                        className="truncate font-medium hover:text-primary hover:underline"
                      >
                        {doc.filename}
                      </Link>
                      <p className="text-xs text-muted-foreground capitalize">
                        {doc.file_type} · {new Date(doc.created_at).toLocaleString()}
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

      <SetQuizPanel
        setId={setId}
        accessToken={accessToken}
        canEdit={canEdit}
        hasReadyDocuments={documents.some(
          (doc) => doc.file_type === "document" && doc.status === "ready",
        )}
      />

      <MultiDocChatPanel documents={documents} workspaceId={setId} />
    </div>
  );
}
