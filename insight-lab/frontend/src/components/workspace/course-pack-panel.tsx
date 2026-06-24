"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type CoursePackResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CoursePackResults } from "@/components/workspace/course-pack-results";
import { useToast } from "@/components/ui/toast";
import { downloadAuthenticatedText, downloadAuthenticatedBlob } from "@/lib/export-utils";

export function CoursePackPanel({
  setId,
  canEdit = true,
  embedded = false,
}: {
  setId: string;
  canEdit?: boolean;
  embedded?: boolean;
}) {
  const { toast } = useToast();
  const [generating, setGenerating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportingCanvas, setExportingCanvas] = useState(false);
  const [exportingBundle, setExportingBundle] = useState(false);
  const [pack, setPack] = useState<CoursePackResponse | null>(null);

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
      await downloadAuthenticatedText(
        `/workspaces/${setId}/course-pack/export/markdown`,
        session.access_token,
        `course-pack-${setId}.md`,
      );
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
      await downloadAuthenticatedBlob(
        `/workspaces/${setId}/export/canvas-cartridge`,
        session.access_token,
        `course-canvas.imscc`,
      );
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
      await downloadAuthenticatedBlob(
        `/workspaces/${setId}/export/lms-bundle`,
        session.access_token,
        `lms-bundle.zip`,
      );
      toast({ title: "LMS bundle downloaded", variant: "success" });
    } catch {
      toast({ title: "LMS bundle export failed", variant: "error" });
    } finally {
      setExportingBundle(false);
    }
  }

  const content = (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {canEdit ? (
          <Button type="button" disabled={generating} onClick={() => void handleGenerate()}>
            {generating ? "Generating pack…" : "Generate course pack"}
          </Button>
        ) : (
          <p className="text-sm text-muted-foreground">
            Course pack generation requires editor or owner access.
          </p>
        )}
        <Button type="button" variant="outline" disabled={exporting} onClick={() => void handleExportMarkdown()}>
          {exporting ? "Exporting…" : "Export Markdown"}
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={exportingCanvas}
          onClick={() => void handleExportCanvasCartridge()}
        >
          {exportingCanvas ? "Exporting…" : "Export Canvas (.imscc)"}
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={exportingBundle}
          onClick={() => void handleExportLmsBundle()}
        >
          {exportingBundle ? "Exporting…" : "Export LMS zip"}
        </Button>
      </div>

      {pack ? <CoursePackResults setId={setId} documents={pack.documents} /> : null}
    </div>
  );

  if (embedded) {
    return (
      <div data-tour="course-pack">
        <p className="mb-4 text-sm text-muted-foreground">
          Generate summary, quiz, flashcards, study guide, and audio for every ready document — then open
          each from the gallery. Export Markdown, a Canvas Common Cartridge (.imscc), or a full LMS zip
          with QTI quizzes and flashcards.
        </p>
        {content}
      </div>
    );
  }

  return (
    <Card className="notebook-surface border-0 shadow-none" data-tour="course-pack">
      <CardHeader>
        <CardTitle>Course pack</CardTitle>
        <CardDescription>
          Generate summary, quiz, flashcards, study guide, and audio for every ready document — then open each from
          the gallery.
        </CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}
