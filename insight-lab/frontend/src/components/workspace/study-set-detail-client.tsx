"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
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
import { ContextBreadcrumb } from "@/components/layout/context-breadcrumb";
import { ShareWorkspacePanel } from "@/components/workspace/share-workspace-panel";
import { SetQuizPanel } from "@/components/workspace/set-quiz-panel";
import { useToast } from "@/components/ui/toast";
import { fetchUploadConfig, type UploadConfigResponse } from "@/lib/api";
import { FileSpreadsheet, FileText } from "lucide-react";
import { canEditWorkspace, workspaceRoleLabel } from "@/lib/workspace-roles";
import { cn } from "@/lib/utils";

const MATERIAL_ROW_HEIGHT_PX = 64;
const MATERIAL_MAX_VISIBLE_ROWS = 5;

function FileTypeIcon({ fileType }: { fileType: string }) {
  const Icon = fileType === "excel" ? FileSpreadsheet : FileText;
  return (
    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
      <Icon className="h-4 w-4" aria-hidden />
    </span>
  );
}

export function StudySetDetailClient({ setId }: { setId: string }) {
  const router = useRouter();
  const { toast } = useToast();
  const [workspace, setWorkspace] = useState<WorkspaceSummary | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [uploadConfig, setUploadConfig] = useState<UploadConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

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
      setError("Failed to load study sheet");
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

  async function handleDeleteStudySet() {
    if (!workspace?.is_owner && workspace?.access_role !== "owner") {
      return;
    }
    if (
      !window.confirm(
        `Delete "${workspace?.name}" permanently? All files, quizzes, and sharing settings will be removed.`,
      )
    ) {
      return;
    }

    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }

    setDeleting(true);
    try {
      const response = await apiFetch(`/workspaces/${setId}`, session.access_token, {
        method: "DELETE",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Could not delete study sheet",
          description: body.error || body.detail,
          variant: "error",
        });
        return;
      }
      toast({ title: "Study sheet deleted", variant: "success" });
      router.push("/dashboard/sets");
    } finally {
      setDeleting(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading study sheet…</p>;
  }

  if (!workspace) {
    return <p className="text-sm text-destructive">{error || "Study sheet not found"}</p>;
  }

  const canEdit = canEditWorkspace(workspace.access_role);
  const isOwner = workspace.access_role === "owner" || Boolean(workspace.is_owner);
  const readyCount = documents.filter((doc) => doc.status === "ready").length;
  const materialsScrollable = documents.length > MATERIAL_MAX_VISIBLE_ROWS;

  return (
    <div className="space-y-6 pb-8">
      <ContextBreadcrumb
        items={[
          { label: "Study sheets", href: "/dashboard/sets" },
          { label: workspace.name },
        ]}
      />

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-semibold tracking-tight">{workspace.name}</h1>
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
        <div className="flex flex-wrap items-center gap-2">
          {isOwner ? (
            <Button
              type="button"
              variant="destructive"
              size="sm"
              disabled={deleting}
              data-tour="delete-set"
              onClick={() => void handleDeleteStudySet()}
            >
              {deleting ? "Deleting…" : "Delete study sheet"}
            </Button>
          ) : null}
          <Button type="button" variant="outline" onClick={() => void loadAll()}>
            Refresh
          </Button>
        </div>
      </div>

      {stats ? <WorkspaceStatsPanel stats={stats} /> : null}

      <CoursePackPanel setId={setId} canEdit={canEdit} />

      <ShareWorkspacePanel
        setId={setId}
        canManage={canEdit}
        isOwner={isOwner}
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

      <Card className="shadow-sm" data-tour="sources-strip">
        <CardHeader>
          <CardTitle>Materials</CardTitle>
          <CardDescription>
            {documents.length === 0
              ? "Upload PDFs, Word docs, or spreadsheets to start studying."
              : `${readyCount} of ${documents.length} ready — open a file for chat, quiz, and study tools.`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {documents.length === 0 ? (
            <p className="text-sm text-muted-foreground">No materials yet — upload your first file above.</p>
          ) : (
            <>
              <ul
                className={cn(
                  "divide-y overflow-hidden rounded-xl border",
                  materialsScrollable && "overflow-y-auto pr-1",
                )}
                style={
                  materialsScrollable
                    ? { maxHeight: MATERIAL_MAX_VISIBLE_ROWS * MATERIAL_ROW_HEIGHT_PX }
                    : undefined
                }
                data-tour="file-list"
              >
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
              {materialsScrollable ? (
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Scroll to see all {documents.length} materials.
                </p>
              ) : null}
            </>
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
