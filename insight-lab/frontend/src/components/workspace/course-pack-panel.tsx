"use client";

import { useMemo, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type CoursePackResponse, type DocumentSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CoursePackResults } from "@/components/workspace/course-pack-results";
import { useToast } from "@/components/ui/toast";

async function readExportError(response: Response): Promise<string> {
  const body = await response.json().catch(() => ({}));
  return typeof body.error === "string" ? body.error : `Export failed (${response.status})`;
}

export function CoursePackPanel({
  setId,
  documents = [],
  canEdit = true,
  embedded = false,
}: {
  setId: string;
  documents?: DocumentSummary[];
  canEdit?: boolean;
  embedded?: boolean;
}) {
  const { toast } = useToast();
  const [generating, setGenerating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportingCanvas, setExportingCanvas] = useState(false);
  const [exportingBundle, setExportingBundle] = useState(false);
  const [pack, setPack] = useState<CoursePackResponse | null>(null);

  const readyDocuments = useMemo(
    () => documents.filter((doc) => doc.file_type === "document" && doc.status === "ready"),
    [documents],
  );
  const readyExcel = useMemo(
    () => documents.filter((doc) => doc.file_type === "excel" && doc.status === "ready"),
    [documents],
  );
  const hasReadyMaterials = readyDocuments.length + readyExcel.length > 0;
  const excelOnly = readyExcel.length > 0 && readyDocuments.length === 0;

  async function handleGenerate() {
    setGenerating(true);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        return;
      }
      const response = await apiFetch(`/workspaces/${setId}/course-pack/generate`, session.access_token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({ title: "Course pack failed", description: body.error, variant: "error" });
        return;
      }
      const data = (await response.json()) as CoursePackResponse;
      setPack(data);
      toast({ title: "Course pack generated", description: "Open artifacts from the cards below.", variant: "success" });
    } finally {
      setGenerating(false);
    }
  }

  async function handleExportMarkdown() {
    setExporting(true);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        return;
      }
      const response = await apiFetch(
        `/workspaces/${setId}/course-pack/export/markdown`,
        session.access_token,
      );
      if (!response.ok) {
        toast({ title: "Export failed", description: await readExportError(response), variant: "error" });
        return;
      }
      const markdown = await response.text();
      const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download = `course-pack-${setId}.md`;
      anchor.click();
      URL.revokeObjectURL(url);
      toast({ title: "Course pack exported", description: "Markdown file downloaded.", variant: "success" });
    } catch {
      toast({ title: "Export failed", variant: "error" });
    } finally {
      setExporting(false);
    }
  }

  async function handleExportCanvasCartridge() {
    setExportingCanvas(true);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        return;
      }
      const response = await apiFetch(`/workspaces/${setId}/export/canvas-cartridge`, session.access_token);
      if (!response.ok) {
        toast({ title: "Canvas export failed", description: await readExportError(response), variant: "error" });
        return;
      }
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const match = /filename="([^"]+)"/.exec(disposition);
      const filename = match?.[1] ?? "course-canvas.imscc";
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
      toast({
        title: "Canvas cartridge downloaded",
        description: "Import in Canvas via Settings → Import course content.",
        variant: "success",
      });
    } catch {
      toast({ title: "Canvas export failed", variant: "error" });
    } finally {
      setExportingCanvas(false);
    }
  }

  async function handleExportLmsBundle() {
    setExportingBundle(true);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        return;
      }
      const response = await apiFetch(`/workspaces/${setId}/export/lms-bundle`, session.access_token);
      if (!response.ok) {
        toast({ title: "LMS bundle export failed", description: await readExportError(response), variant: "error" });
        return;
      }
      const disposition = response.headers.get("Content-Disposition") ?? "";
      const match = /filename="([^"]+)"/.exec(disposition);
      const filename = match?.[1] ?? "lms-bundle.zip";
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
      toast({ title: "LMS bundle downloaded", variant: "success" });
    } catch {
      toast({ title: "LMS bundle export failed", variant: "error" });
    } finally {
      setExportingBundle(false);
    }
  }

  const introCopy = excelOnly
    ? "This sheet has spreadsheets only. Generate packs spreadsheet analysis summaries. Quizzes, flashcards, and study guides require PDF or Word uploads."
    : "Generate summary, quiz, flashcards, study guide, and audio for every ready document — plus spreadsheet summaries. Export Markdown, Canvas (.imscc), or an LMS zip.";

  const content = (
    <div className="space-y-4">
      {!hasReadyMaterials ? (
        <p className="text-sm text-muted-foreground">
          No ready materials yet. Upload files and wait for processing (or run Analyze on spreadsheets) before
          generating a course pack.
        </p>
      ) : null}

      <div className="flex flex-wrap gap-2">
        {canEdit ? (
          <Button
            type="button"
            disabled={generating || !hasReadyMaterials}
            onClick={() => void handleGenerate()}
          >
            {generating ? "Generating pack…" : "Generate course pack"}
          </Button>
        ) : (
          <p className="text-sm text-muted-foreground">
            Course pack generation requires editor or owner access.
          </p>
        )}
        {canEdit ? (
          <>
            <Button
              type="button"
              variant="outline"
              disabled={exporting || !hasReadyMaterials}
              onClick={() => void handleExportMarkdown()}
            >
              {exporting ? "Exporting…" : "Export Markdown"}
            </Button>
            <Button
              type="button"
              variant="outline"
              disabled={exportingCanvas || !hasReadyMaterials}
              onClick={() => void handleExportCanvasCartridge()}
            >
              {exportingCanvas ? "Exporting…" : "Export Canvas (.imscc)"}
            </Button>
            <Button
              type="button"
              variant="outline"
              disabled={exportingBundle || !hasReadyMaterials}
              onClick={() => void handleExportLmsBundle()}
            >
              {exportingBundle ? "Exporting…" : "Export LMS zip"}
            </Button>
          </>
        ) : null}
      </div>

      {pack ? <CoursePackResults setId={setId} documents={pack.documents} /> : null}
    </div>
  );

  if (embedded) {
    return (
      <div data-tour="course-pack">
        <p className="mb-4 text-sm text-muted-foreground">{introCopy}</p>
        {content}
      </div>
    );
  }

  return (
    <Card className="notebook-surface border-0 shadow-none" data-tour="course-pack">
      <CardHeader>
        <CardTitle>Course pack</CardTitle>
        <CardDescription>{introCopy}</CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}
