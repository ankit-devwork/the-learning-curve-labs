"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type CoursePackResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CoursePackResults } from "@/components/workspace/course-pack-results";
import { useToast } from "@/components/ui/toast";

export function CoursePackPanel({ setId, canEdit = true }: { setId: string; canEdit?: boolean }) {
  const { toast } = useToast();
  const [generating, setGenerating] = useState(false);
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

  return (
    <Card className="notebook-surface border-0 shadow-none" data-tour="course-pack">
      <CardHeader>
        <CardTitle>Course pack</CardTitle>
        <CardDescription>
          Generate summary, quiz, flashcards, study guide, and audio for every ready document — then open each from
          the gallery.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {canEdit ? (
          <Button type="button" disabled={generating} onClick={() => void handleGenerate()}>
            {generating ? "Generating pack…" : "Generate course pack"}
          </Button>
        ) : (
          <p className="text-sm text-muted-foreground">
            Course pack generation requires editor or owner access.
          </p>
        )}

        {pack ? <CoursePackResults setId={setId} documents={pack.documents} /> : null}
      </CardContent>
    </Card>
  );
}
