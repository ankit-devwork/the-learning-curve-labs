"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, type StudySessionPlan, type StudySessionRecord } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { StudioTab } from "@/lib/notebook-utils";

type StudySessionPanelProps = {
  documentId: string;
  accessToken: string | null;
  ready: boolean;
  busy?: boolean;
  onSelectTab: (tab: StudioTab) => void;
  onGenerateFlashcards: () => void;
  onGenerateQuiz: () => void;
  onSessionChange?: (session: StudySessionRecord | null) => void;
};

function stepDescription(step: StudySessionPlan["steps"][number]): string {
  if (step.step === "brief") {
    return "Read the AI summary before quizzing";
  }
  if (step.step === "flashcards") {
    return step.ready ? "Flashcard deck ready" : "Generate flashcards to study";
  }
  if (step.step === "quiz") {
    return step.ready ? "Quiz ready" : "Generate a quiz to test mastery";
  }
  return "";
}

export function StudySessionPanel({
  documentId,
  accessToken,
  ready,
  busy,
  onSelectTab,
  onGenerateFlashcards,
  onGenerateQuiz,
  onSessionChange,
}: StudySessionPanelProps) {
  const [session, setSession] = useState<StudySessionRecord | null>(null);
  const [previewPlan, setPreviewPlan] = useState<StudySessionPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [migrationNotice, setMigrationNotice] = useState<string | null>(null);

  const updateSession = useCallback(
    (next: StudySessionRecord | null) => {
      setSession(next);
      onSessionChange?.(next);
    },
    [onSessionChange],
  );

  const loadActiveSession = useCallback(async () => {
    if (!accessToken || !ready) {
      return;
    }
    setLoading(true);
    setError(null);
    const [activeRes, planRes] = await Promise.all([
      apiFetch(`/documents/${documentId}/study-session/active`, accessToken),
      apiFetch(`/documents/${documentId}/study-session/plan`, accessToken),
    ]);
    setLoading(false);
    if (planRes.ok) {
      setPreviewPlan((await planRes.json()) as StudySessionPlan);
    }
    if (activeRes.ok) {
      const data = await activeRes.json();
      if (data.migration_required && data.notice) {
        setMigrationNotice(data.notice);
        updateSession(null);
      } else {
        setMigrationNotice(null);
        updateSession((data.session as StudySessionRecord | null) ?? null);
      }
    }
  }, [accessToken, documentId, ready, updateSession]);

  useEffect(() => {
    void loadActiveSession();
  }, [loadActiveSession]);

  async function startSession() {
    if (!accessToken) {
      return;
    }
    setStarting(true);
    setError(null);
    const response = await apiFetch(`/documents/${documentId}/study-session/start`, accessToken, {
      method: "POST",
    });
    setStarting(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Could not start study session");
      return;
    }
    updateSession((await response.json()) as StudySessionRecord);
  }

  async function advanceStep(stepIndex: number, status: "in_progress" | "completed" | "skipped") {
    if (!accessToken || !session) {
      return;
    }
    const response = await apiFetch(
      `/study-sessions/${session.session_id}/steps/${stepIndex}/advance`,
      accessToken,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      },
    );
    if (response.ok) {
      updateSession((await response.json()) as StudySessionRecord);
    }
  }

  async function runStep(stepIndex: number) {
    const plan = (session?.plan as StudySessionPlan | undefined) ?? previewPlan;
    const step = plan?.steps[stepIndex];
    if (!step) {
      return;
    }

    if (session) {
      await advanceStep(stepIndex, "in_progress");
    }

    if (step.step === "brief") {
      onSelectTab("brief");
      return;
    }
    if (step.step === "flashcards") {
      onSelectTab("flashcards");
      if (!step.ready) {
        onGenerateFlashcards();
      }
      return;
    }
    if (step.step === "quiz") {
      onSelectTab("quiz");
      if (!step.ready) {
        onGenerateQuiz();
      }
    }
  }

  async function markComplete(stepIndex: number) {
    await advanceStep(stepIndex, "completed");
  }

  const plan = session?.plan as StudySessionPlan | undefined;
  const displayPlan = plan ?? previewPlan;
  const progress = session?.progress;
  const steps = session?.steps ?? [];

  if (!ready) {
    return (
      <Card className="notebook-surface border-0 shadow-none">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Study session</CardTitle>
          <CardDescription>Guided flow through brief, flashcards, and quiz.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Process this document to start a tracked session.</p>
        </CardContent>
      </Card>
    );
  }

  if (loading && !displayPlan) {
    return <p className="text-sm text-muted-foreground">Loading study session…</p>;
  }

  if (!displayPlan) {
    return null;
  }

  return (
    <Card className="notebook-surface border-0 shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Study session</CardTitle>
        <CardDescription>
          ~{displayPlan.estimated_minutes} min guided flow
          {displayPlan.focus_topic ? ` · focus: ${displayPlan.focus_topic}` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {progress ? (
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {progress.completed_steps} / {progress.total_steps} steps
              </span>
              <span>{progress.percent}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          </div>
        ) : null}

        {!session ? (
          <Button type="button" disabled={starting || loading || busy} onClick={() => void startSession()}>
            {starting ? "Starting…" : "Start tracked session"}
          </Button>
        ) : (
          <p className="text-sm text-muted-foreground">
            Session active — complete steps below or mark them done as you go.
          </p>
        )}

        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        {migrationNotice ? (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100">
            {migrationNotice}
          </p>
        ) : null}

        {displayPlan.weak_concepts.length > 0 ? (
          <div className="rounded-lg border bg-muted/20 px-3 py-2 text-sm">
            <p className="font-medium">Focus topics</p>
            <ul className="mt-1 space-y-1 text-muted-foreground">
              {displayPlan.weak_concepts.map((concept) => (
                <li key={concept.concept_id}>
                  {concept.name}
                  {concept.percent != null ? ` (${concept.percent}%)` : ""}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <ol className="space-y-2">
          {displayPlan.steps.map((step, index) => {
            const progressRow = steps.find((row) => row.step_index === index);
            const status = progressRow?.status ?? "pending";
            return (
              <li
                key={step.step}
                className={cn(
                  "flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-sm",
                  status === "completed" && "border-emerald-300/50 bg-emerald-50/40 dark:bg-emerald-950/20",
                  status === "in_progress" && "border-primary/40 bg-primary/5",
                )}
              >
                <div className="min-w-0">
                  <p className="font-medium">
                    {index + 1}. {step.label}
                    {status === "completed" ? " ✓" : ""}
                  </p>
                  <p className="text-xs text-muted-foreground">{stepDescription(step)}</p>
                </div>
                <div className="flex shrink-0 gap-1">
                  {session && status !== "completed" ? (
                    <>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={busy}
                        onClick={() => void runStep(index)}
                      >
                        Start
                      </Button>
                      <Button type="button" size="sm" variant="ghost" onClick={() => void markComplete(index)}>
                        Done
                      </Button>
                    </>
                  ) : session ? null : (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={busy}
                      onClick={() => void runStep(index)}
                    >
                      Start
                    </Button>
                  )}
                </div>
              </li>
            );
          })}
        </ol>

        <div className="flex flex-wrap gap-2">
          {session ? (
            <Button type="button" disabled={busy} onClick={() => void runStep(session.current_step_index)}>
              Continue session
            </Button>
          ) : (
            <Button type="button" disabled={busy} onClick={() => void runStep(0)}>
              Start full session
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
