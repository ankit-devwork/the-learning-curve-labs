"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type DocumentSummary,
  type WorkspaceProgress,
  type WorkspaceSummary,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import { MultiDocChatPanel } from "@/components/documents/multi-doc-chat-panel";
import { UploadDropzone } from "@/components/workspace/upload-dropzone";
import { UploadGuidance, UploadGuidanceNotice } from "@/components/workspace/upload-guidance";
import { ProgressDashboardPanel } from "@/components/workspace/progress-dashboard-panel";
import { ClassroomAnalyticsPanel } from "@/components/workspace/classroom-analytics-panel";
import { SourceLinksPanel, shouldShowSourceLinks } from "@/components/workspace/source-links-panel";
import { CoursePackPanel } from "@/components/workspace/course-pack-panel";
import { ContextBreadcrumb } from "@/components/layout/context-breadcrumb";
import { ShareWorkspacePanel } from "@/components/workspace/share-workspace-panel";
import { SetQuizPanel } from "@/components/workspace/set-quiz-panel";
import { LearningPathPanel } from "@/components/workspace/learning-path-panel";
import { TeamChatDock } from "@/components/workspace/team-chat-dock";
import { WorkspaceConceptGraphPanel } from "@/components/workspace/workspace-concept-graph-panel";
import { WorkspaceStudySessionPanel } from "@/components/workspace/workspace-study-session-panel";
import { useToast } from "@/components/ui/toast";
import { fetchUploadConfig, type UploadConfigResponse } from "@/lib/api";
import { FileSpreadsheet, FileText, Package, Share2, Upload } from "lucide-react";
import { canEditWorkspace, workspaceRoleLabel } from "@/lib/workspace-roles";
import { cn } from "@/lib/utils";
import { SlideOverDrawer } from "@/components/ui/slide-over-drawer";
import { CollapsibleSection } from "@/components/ui/collapsible-section";

type SheetDrawerPanel = "share" | "upload" | "course-pack" | "links";

const MATERIAL_ROW_HEIGHT_PX = 52;
const MATERIAL_MAX_VISIBLE_ROWS = 4;

function FileTypeIcon({ fileType }: { fileType: string }) {
  const Icon = fileType === "excel" ? FileSpreadsheet : FileText;
  return (
    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
      <Icon className="h-3.5 w-3.5" aria-hidden />
    </span>
  );
}

export function StudySetDetailClient({ setId }: { setId: string }) {
  const router = useRouter();
  const { toast } = useToast();
  const [workspace, setWorkspace] = useState<WorkspaceSummary | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [progress, setProgress] = useState<WorkspaceProgress | null>(null);
  const [uploadConfig, setUploadConfig] = useState<UploadConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [uploadAcknowledged, setUploadAcknowledged] = useState(false);
  const [drawerPanel, setDrawerPanel] = useState<SheetDrawerPanel | null>(null);
  const [learningPathId, setLearningPathId] = useState<string | null>(null);
  const [sourceLinkCount, setSourceLinkCount] = useState(0);

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

    const [workspaceRes, docsRes, statsRes, linksRes] = await Promise.all([
      apiFetch(`/workspaces/${setId}`, session.access_token),
      apiFetch(`/workspaces/${setId}/documents`, session.access_token),
      apiFetch(`/workspaces/${setId}/progress`, session.access_token),
      apiFetch(`/workspaces/${setId}/source-links`, session.access_token),
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
    setProgress(statsData as WorkspaceProgress);
    if (linksRes.ok) {
      const linksData = await linksRes.json();
      setSourceLinkCount((linksData.links ?? []).length);
    } else {
      setSourceLinkCount(0);
    }
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
    if (uploadConfig.guidance?.require_acknowledgment && !uploadAcknowledged) {
      toast({
        title: "Confirm upload guidance",
        description: "Please acknowledge the privacy notice before uploading.",
        variant: "error",
      });
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
      setUploadAcknowledged(false);
      await loadAll();
    } finally {
      setUploading(false);
    }
  }

  async function handleDeleteDocument(doc: DocumentSummary) {
    if (
      !window.confirm(
        `Delete "${doc.filename}" permanently? This removes the file, chat history, quizzes, and other generated content.`,
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

    setDeletingDocumentId(doc.id);
    try {
      const response = await apiFetch(`/documents/${doc.id}`, session.access_token, {
        method: "DELETE",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Could not delete file",
          description: body.error || body.detail,
          variant: "error",
        });
        return;
      }
      toast({ title: "File deleted", description: doc.filename, variant: "success" });
      await loadAll();
    } finally {
      setDeletingDocumentId(null);
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
  const readyDocumentCount = documents.filter(
    (doc) => doc.file_type === "document" && doc.status === "ready",
  ).length;
  const hasReadyDocuments = readyDocumentCount > 0;
  const materialsScrollable = documents.length > MATERIAL_MAX_VISIBLE_ROWS;
  const showSourceLinks = shouldShowSourceLinks(documents, sourceLinkCount);

  function openDrawer(panel: SheetDrawerPanel) {
    setDrawerPanel(panel);
  }

  function tryOpenUploadDrawer() {
    if (
      uploadConfig?.guidance?.require_acknowledgment &&
      !uploadAcknowledged &&
      documents.length === 0
    ) {
      toast({
        title: "Confirm upload notice",
        description: "Please acknowledge the privacy notice in the Materials section first.",
        variant: "error",
      });
      return;
    }
    openDrawer("upload");
  }

  function closeDrawer() {
    setDrawerPanel(null);
  }

  const drawerCopy: Record<
    SheetDrawerPanel,
    { title: string; description: string }
  > = {
    share: {
      title: "Share study sheet",
      description: "Invite classmates and manage access.",
    },
    upload: {
      title: "Upload materials",
      description: uploadConfig
        ? `Allowed: ${uploadConfig.allowed_extensions.join(", ")} — up to ${uploadConfig.max_mb} MB`
        : "Add PDFs, Word docs, or spreadsheets to this sheet.",
    },
    "course-pack": {
      title: "Course pack",
      description: "Generate learning outputs for every ready document or analyzed spreadsheet.",
    },
    links: {
      title: "Source links",
      description: showSourceLinks
        ? "Connect spreadsheets to related PDF or Word files for richer Excel Q&A."
        : "Only applies when this sheet includes spreadsheets.",
    },
  };

  return (
    <div className="space-y-4 pb-6">
      <ContextBreadcrumb
        items={[
          { label: "Study sheets", href: "/dashboard/sets" },
          { label: workspace.name },
        ]}
      />

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight">{workspace.name}</h1>
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
          <Button
            type="button"
            variant="outline"
            size="sm"
            data-tour="share-btn"
            onClick={() => openDrawer("share")}
          >
            <Share2 className="mr-1.5 h-4 w-4" aria-hidden />
            Share
          </Button>
          {canEdit ? (
            <>
              <Button
                type="button"
                variant="outline"
                size="sm"
                data-tour="upload-btn"
                onClick={() => tryOpenUploadDrawer()}
              >
                <Upload className="mr-1.5 h-4 w-4" aria-hidden />
                Upload
              </Button>
              {showSourceLinks ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => openDrawer("links")}
                >
                  Source links
                </Button>
              ) : null}
              <Button
                type="button"
                variant="outline"
                size="sm"
                data-tour="course-pack-btn"
                onClick={() => openDrawer("course-pack")}
              >
                <Package className="mr-1.5 h-4 w-4" aria-hidden />
                Course pack
              </Button>
            </>
          ) : null}
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

      <Card className="shadow-sm" data-tour="sources-strip">
        <CardHeader className="space-y-1 pb-2 pt-4">
          <CardTitle className="text-base">Materials</CardTitle>
          <CardDescription className="text-xs">
            {documents.length === 0
              ? "Upload PDFs, Word docs, or spreadsheets to start studying."
              : `${readyCount} of ${documents.length} ready — open a file for chat, quiz, and study tools.`}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 pt-0">
          {canEdit && uploadConfig?.guidance ? (
            documents.length === 0 ? (
              <UploadGuidance
                guidance={uploadConfig.guidance}
                acknowledged={uploadAcknowledged}
                onAcknowledgedChange={setUploadAcknowledged}
              />
            ) : (
              <UploadGuidanceNotice guidance={uploadConfig.guidance} compact />
            )
          ) : null}
          {documents.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-2 text-center">
              <p className="text-sm text-muted-foreground">
                {canEdit
                  ? "No materials yet — upload your first file to start studying."
                  : "No materials in this sheet yet."}
              </p>
              {canEdit ? (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={
                    Boolean(uploadConfig?.guidance?.require_acknowledgment) && !uploadAcknowledged
                  }
                  onClick={() => tryOpenUploadDrawer()}
                >
                  <Upload className="mr-1.5 h-4 w-4" aria-hidden />
                  Upload materials
                </Button>
              ) : null}
            </div>
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
                  <li key={doc.id} className="flex items-center justify-between gap-2 px-3 py-2">
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
                    <div className="flex shrink-0 items-center gap-2">
                      <StatusBadge status={doc.status} />
                      {canEdit ? (
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          disabled={deletingDocumentId === doc.id}
                          onClick={() => void handleDeleteDocument(doc)}
                        >
                          {deletingDocumentId === doc.id ? "Deleting…" : "Delete"}
                        </Button>
                      ) : null}
                    </div>
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

      {progress ? <ProgressDashboardPanel progress={progress} compact /> : null}

      {hasReadyDocuments ? (
        <div className="space-y-2">
          <CollapsibleSection
            title="Guided study plan"
            description="Optional step-by-step checklist with saved progress"
            tourId="study-session"
            compact
          >
            <WorkspaceStudySessionPanel
              setId={setId}
              accessToken={accessToken}
              hasReadyDocuments={hasReadyDocuments}
              learningPathId={learningPathId}
              embedded
            />
          </CollapsibleSection>

          <CollapsibleSection
            title="Learning path"
            description="Concepts ordered by prerequisites and quiz mastery"
            tourId="learning-path"
            compact
          >
            <LearningPathPanel
              setId={setId}
              accessToken={accessToken}
              hasReadyDocuments={hasReadyDocuments}
              onPathGenerated={setLearningPathId}
              embedded
            />
          </CollapsibleSection>

          <CollapsibleSection
            title="Concept graph"
            description="Topics across all documents in this sheet"
            tourId="concept-graph"
            compact
          >
            <WorkspaceConceptGraphPanel
              setId={setId}
              accessToken={accessToken}
              hasReadyDocuments={hasReadyDocuments}
              embedded
            />
          </CollapsibleSection>
        </div>
      ) : null}

      <div className="space-y-2">
        {canEdit ? (
          <CollapsibleSection
            title="Classroom analytics"
            description="Per-member quiz scores, flashcards, and weak topics"
            tourId="classroom-analytics"
            compact
          >
            <ClassroomAnalyticsPanel
              setId={setId}
              accessToken={accessToken}
              canManage={canEdit}
              embedded
            />
          </CollapsibleSection>
        ) : null}

        <CollapsibleSection
          id="set-quiz"
          title="Practice quiz (whole sheet)"
          description="Adaptive quiz from weak topics across all files"
          tourId="set-quiz"
          compact
        >
          <SetQuizPanel
            setId={setId}
            accessToken={accessToken}
            canEdit={canEdit}
            hasReadyDocuments={hasReadyDocuments}
            embedded
          />
        </CollapsibleSection>

        <CollapsibleSection
          title="Compare documents"
          description="Ask one question across multiple files"
          tourId="compare-docs"
          compact
        >
          <MultiDocChatPanel documents={documents} workspaceId={setId} embedded />
        </CollapsibleSection>
      </div>

      <SlideOverDrawer
        open={drawerPanel !== null}
        title={drawerPanel ? drawerCopy[drawerPanel].title : ""}
        description={drawerPanel ? drawerCopy[drawerPanel].description : undefined}
        onClose={closeDrawer}
        widthClassName="w-[min(520px,100vw)]"
      >
        {drawerPanel === "share" ? (
          <ShareWorkspacePanel
            setId={setId}
            canManage={canEdit}
            isOwner={isOwner}
            embedded
          />
        ) : null}
        {drawerPanel === "upload" ? (
          canEdit ? (
            <UploadDropzone
              accept={uploadConfig?.accept}
              disabled={!uploadConfig}
              uploading={uploading}
              guidance={uploadConfig?.guidance ?? null}
              acknowledged={uploadAcknowledged}
              onAcknowledgedChange={setUploadAcknowledged}
              onFileSelected={(file) => void handleUpload(file)}
            />
          ) : (
            <p className="text-sm text-muted-foreground">
              Viewer access — you can read and study files but not upload.
            </p>
          )
        ) : null}
        {drawerPanel === "course-pack" ? (
          <CoursePackPanel setId={setId} documents={documents} canEdit={canEdit} embedded />
        ) : null}
        {drawerPanel === "links" ? (
          <SourceLinksPanel
            setId={setId}
            documents={documents}
            accessToken={accessToken}
            canEdit={canEdit}
          />
        ) : null}
      </SlideOverDrawer>

      <TeamChatDock setId={setId} accessToken={accessToken} isOwner={isOwner} />
    </div>
  );
}
