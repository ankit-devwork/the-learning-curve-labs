"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { apiFetch, type WorkspaceStudySessionPlan } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type WorkspaceStudySessionPanelProps = {
  setId: string;
  accessToken: string | null;
  hasReadyDocuments: boolean;
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
}: WorkspaceStudySessionPanelProps) {
  const [plan, setPlan] = useState<WorkspaceStudySessionPlan | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadPlan = useCallback(async () => {
    if (!accessToken || !hasReadyDocuments) {
      return;
    }
    setLoading(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/study-session/plan`, accessToken);
    setLoading(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Could not load study session");
      return;
    }
    setPlan((await response.json()) as WorkspaceStudySessionPlan);
  }, [accessToken, hasReadyDocuments, setId]);

  useEffect(() => {
    void loadPlan();
  }, [loadPlan]);

  function scrollToSetQuiz() {
    document.getElementById("set-quiz")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function runStep(index: number) {
    if (!plan) {
      return;
    }
    const step = plan.steps[index];
    setActiveStep(index);

    if (step.step === "focus") {
      return;
    }
    if (step.step === "brief" && step.document_id) {
      window.location.href = `/dashboard/sets/${setId}/documents/${step.document_id}#brief`;
      return;
    }
    if (step.step === "flashcards" && step.document_id) {
      window.location.href = `/dashboard/sets/${setId}/documents/${step.document_id}#flashcards`;
      return;
    }
    if (step.step === "adaptive_quiz" || step.step === "set_quiz") {
      scrollToSetQuiz();
    }
  }

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

  if (loading && !plan) {
    return (
      <Card className="shadow-sm" data-tour="study-session">
        <CardContent className="py-6">
          <p className="text-sm text-muted-foreground">Loading study session…</p>
        </CardContent>
      </Card>
    );
  }

  if (!plan) {
    return error ? (
      <Card className="shadow-sm" data-tour="study-session">
        <CardContent className="py-6">
          <p className="text-sm text-destructive">{error}</p>
        </CardContent>
      </Card>
    ) : null;
  }

  const focusStep = plan.steps.find((step) => step.step === "focus");

  return (
    <Card className="shadow-sm" data-tour="study-session">
      <CardHeader>
        <CardTitle>Study session</CardTitle>
        <CardDescription>
          ~{plan.estimated_minutes} min guided flow across {plan.document_count} document
          {plan.document_count === 1 ? "" : "s"}
          {plan.focus_topic ? ` · focus: ${plan.focus_topic}` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
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
          {plan.steps
            .filter((step) => step.step !== "focus")
            .map((step, index) => {
              const stepIndex = plan.steps.indexOf(step);
              return (
                <li
                  key={`${step.step}-${"document_id" in step ? step.document_id : index}`}
                  className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-sm ${
                    activeStep === stepIndex ? "border-primary/40 bg-primary/5" : ""
                  }`}
                >
                  <div className="min-w-0">
                    <p className="font-medium">
                      {index + 1}. {step.label}
                    </p>
                    <p className="text-xs text-muted-foreground">{stepDescription(step)}</p>
                    {"ready" in step && step.step !== "set_quiz" && step.step !== "adaptive_quiz" ? (
                      <p className="text-xs text-muted-foreground">
                        {step.ready ? "Ready" : "Generate in document studio"}
                      </p>
                    ) : null}
                  </div>
                  <Button type="button" size="sm" variant="outline" onClick={() => runStep(stepIndex)}>
                    {step.step === "adaptive_quiz" || step.step === "set_quiz" ? "Go" : "Start"}
                  </Button>
                </li>
              );
            })}
        </ol>

        <div className="flex flex-wrap gap-2">
          <Button type="button" onClick={() => runStep(0)}>
            Start full session
          </Button>
          <Button type="button" variant="outline" asChild>
            <Link href="#set-quiz">Open set quiz</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
