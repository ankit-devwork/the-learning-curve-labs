"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type CoursePackResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
      toast({ title: "Course pack generated", variant: "success" });
    } finally {
      setGenerating(false);
    }
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Course pack</CardTitle>
        <CardDescription>
          One click to generate summary, quiz, flashcards, study guide, and audio overview for every
          ready document in this set.
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

        {pack ? (
          <ul className="space-y-3 text-sm">
            {pack.documents.map((item) => (
              <li key={item.document_id} className="rounded-md border p-3">
                <p className="font-medium">{item.filename}</p>
                <ul className="mt-2 list-inside list-disc text-xs text-muted-foreground">
                  {item.artifacts.quiz_id ? <li>Quiz ready</li> : null}
                  {item.artifacts.flashcard_set_id ? <li>Flashcards ready</li> : null}
                  {item.artifacts.study_guide_id ? <li>Study guide ready</li> : null}
                  {item.artifacts.audio_script ? <li>Audio script ready</li> : null}
                </ul>
                {item.errors.length > 0 ? (
                  <p className="mt-2 text-xs text-destructive">{item.errors.join("; ")}</p>
                ) : null}
              </li>
            ))}
          </ul>
        ) : null}
      </CardContent>
    </Card>
  );
}
