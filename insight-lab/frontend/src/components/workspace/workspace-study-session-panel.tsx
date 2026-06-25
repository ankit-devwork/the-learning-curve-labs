"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { apiFetch, type StudySessionRecord, type WorkspaceStudySessionPlan } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type WorkspaceStudySessionPanelProps = {
  setId: string;
  accessToken: string | null;
  hasReadyDocuments: boolean;
  learningPathId?: string | null;
};

function stepDescription(step: WorkspaceStudySessionPlan["steps"][number]): string {
  if (step.step === "focus") {
    return step.focus_topic ? `Focus: ${step.focus_topic}` : "Review weak topics from your quizzes";
  }
  if (step.step === "adaptive_quiz") {
    return `${step.weak_count} weak topic${step.weak_count === 1 ? "" : "s"} to practice`;
  }
  if (step.step === "set_quiz") {
    return step.hint ?? "Generate a quiz across this study sheet";
  }
  if (step.step === "brief" || step.step === "flashcards") {
    return step.filename;
  }
  return "";
}

export function WorkspaceStudySessionPanel({
  setId,
  accessToken,
  hasReadyDocuments,
  learningPathId = null,
}: WorkspaceStudySessionPanelProps) {
  const [session, setSession] = useState<StudySessionRecord | null>(null);
  const [previewPlan, setPreviewPlan] = useState<WorkspaceStudySessionPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [migrationNotice, setMigrationNotice] = useState<string | null>(null);

  const loadActiveSession = useCallback(async () => {
    if (!accessToken || !hasReadyDocuments) {
      return;
    }
    setLoading(true);
    setError(null);
    const [activeRes, planRes] = await Promise.all([
      apiFetch(`/workspaces/${setId}/study-session/active`, accessToken),
      apiFetch(`/workspaces/${setId}/study-session/plan`, accessToken),
    ]);
    setLoading(false);
    if (planRes.ok) {
      setPreviewPlan((await planRes.json()) as WorkspaceStudySessionPlan);
    }
    if (activeRes.ok) {
      const data = await activeRes.json();
      if (data.migration_required && data.notice) {
        setMigrationNotice(data.notice);
        setSession(null);
      } else {
        setMigrationNotice(null);
        setSession(data.session ?? null);
      }
    }
  }, [accessToken, hasReadyDocuments, setId]);

  useEffect(() => {
    void loadActiveSession();
  }, [loadActiveSession]);

  async function startSession() {
    if (!accessToken) {
      return;
    }
    setStarting(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/study-session/start`, accessToken, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ learning_path_id: learningPathId ?? undefined }),
    });
    setStarting(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Could not start study session");
      return;
    }
    setSession((await response.json()) as StudySessionRecord);
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
      setSession((await response.json()) as StudySessionRecord);
    }
  }

  function scrollToSetQuiz() {
    window.document.getElementById("set-quiz")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function runStep(stepIndex: number) {
    const steps = session?.steps ?? [];
    const stepProgress = steps.find((row) => row.step_index === stepIndex);
    const payload = stepProgress?.payload as WorkspaceStudySessionPlan["steps"][number] | undefined;
    if (!payload) {
      return;
    }

    await advanceStep(stepIndex, "in_progress");

    if (payload.step === "brief" && "document_id" in payload && payload.document_id) {
      window.location.href = `/dashboard/sets/${setId}/documents/${payload.document_id}#brief`;
      return;
    }
    if (payload.step === "flashcards" && "document_id" in payload && payload.document_id) {
      window.location.href = `/dashboard/sets/${setId}/documents/${payload.document_id}#flashcards`;
      return;
    }
    if (payload.step === "adaptive_quiz" || payload.step === "set_quiz") {
      scrollToSetQuiz();
    }
  }

  async function markComplete(stepIndex: number) {
    await advanceStep(stepIndex, "completed");
  }

  const plan = session?.plan as WorkspaceStudySessionPlan | undefined;
  const displayPlan = plan ?? previewPlan;
  const progress = session?.progress;
  const steps = session?.steps ?? [];

  if (!hasReadyDocuments) {
    return (
      <Card className="shadow-sm" data-tour="study-session">
        <CardHeader>
          <CardTitle>Study session</CardTitle>
          <CardDescription>Guided flow across every ready document in this sheet.</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Upload and process documents to start a set-wide study session.
          </p>
        </CardContent>
      </Card>
    );
  }

  const focusStep = displayPlan?.steps.find((step) => step.step === "focus");

  return (
    <Card className="shadow-sm" data-tour="study-session">
      <CardHeader>
        <CardTitle>Study session</CardTitle>
        <CardDescription>
          {displayPlan
            ? `~${displayPlan.estimated_minutes} min guided flow across ${displayPlan.document_count} document${displayPlan.document_count === 1 ? "" : "s"}`
            : "Track progress through your study sheet"}
          {displayPlan?.focus_topic ? ` · focus: ${displayPlan.focus_topic}` : ""}
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
          <Button type="button" disabled={starting || loading} onClick={() => void startSession()}>
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

        {focusStep?.step === "focus" && focusStep.weak_concepts.length > 0 ? (
          <div className="rounded-lg border bg-muted/20 px-3 py-2 text-sm">
            <p className="font-medium">Focus topics</p>
            <ul className="mt-1 space-y-1 text-muted-foreground">
              {focusStep.weak_concepts.map((concept) => (
                <li key={`${concept.document_id ?? "doc"}-${concept.concept_id}`}>
                  {concept.name}
                  {concept.document_filename ? ` · ${concept.document_filename}` : ""}
                  {concept.percent != null ? ` (${concept.percent}%)` : ""}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <ol className="space-y-2">
          {(displayPlan?.steps ?? [])
            .filter((step) => step.step !== "focus")
            .map((step, index) => {
              const stepIndex = displayPlan?.steps.indexOf(step) ?? index;
              const progressRow = steps.find((row) => row.step_index === stepIndex);
              const status = progressRow?.status ?? "pending";
              return (
                <li
                  key={`${step.step}-${"document_id" in step ? step.document_id : index}`}
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
                        <Button type="button" size="sm" variant="outline" onClick={() => void runStep(stepIndex)}>
                          Start
                        </Button>
                        <Button type="button" size="sm" variant="ghost" onClick={() => void markComplete(stepIndex)}>
                          Done
                        </Button>
                      </>
                    ) : session ? null : (
                      <span className="text-xs text-muted-foreground capitalize">{status.replace("_", " ")}</span>
                    )}
                  </div>
                </li>
              );
            })}
        </ol>

        <div className="flex flex-wrap gap-2">
          {session ? (
            <Button type="button" onClick={() => void runStep(session.current_step_index)}>
              Continue session
            </Button>
          ) : null}
          <Button type="button" variant="outline" asChild>
            <Link href="#set-quiz">Open set quiz</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
