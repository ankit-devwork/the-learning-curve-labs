"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, type StudySessionPlan } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { StudioTab } from "@/lib/notebook-utils";

type StudySessionPanelProps = {
  documentId: string;
  accessToken: string | null;
  ready: boolean;
  busy?: boolean;
  onSelectTab: (tab: StudioTab) => void;
  onGenerateFlashcards: () => void;
  onGenerateQuiz: () => void;
};

export function StudySessionPanel({
  documentId,
  accessToken,
  ready,
  busy,
  onSelectTab,
  onGenerateFlashcards,
  onGenerateQuiz,
}: StudySessionPanelProps) {
  const [plan, setPlan] = useState<StudySessionPlan | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);

  const loadPlan = useCallback(async () => {
    if (!accessToken || !ready) {
      return;
    }
    setLoading(true);
    const response = await apiFetch(`/documents/${documentId}/study-session/plan`, accessToken);
    setLoading(false);
    if (response.ok) {
      setPlan((await response.json()) as StudySessionPlan);
    }
  }, [accessToken, documentId, ready]);

  useEffect(() => {
    void loadPlan();
  }, [loadPlan]);

  function runStep(index: number) {
    if (!plan) {
      return;
    }
    const step = plan.steps[index];
    setActiveStep(index);
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

  if (loading && !plan) {
    return <p className="text-sm text-muted-foreground">Loading study session…</p>;
  }

  if (!plan) {
    return null;
  }

  return (
    <Card className="notebook-surface border-0 shadow-none">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Study session</CardTitle>
        <CardDescription>
          ~{plan.estimated_minutes} min guided flow
          {plan.focus_topic ? ` · focus: ${plan.focus_topic}` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <ol className="space-y-2">
          {plan.steps.map((step, index) => (
            <li
              key={step.step}
              className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-sm ${
                activeStep === index ? "border-primary/40 bg-primary/5" : ""
              }`}
            >
              <span>
                <span className="font-medium">{index + 1}. {step.label}</span>
                <span className="ml-2 text-xs text-muted-foreground">
                  {step.ready ? "Ready" : "Generate"}
                </span>
              </span>
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={!ready || busy}
                onClick={() => runStep(index)}
              >
                Start
              </Button>
            </li>
          ))}
        </ol>
        <Button type="button" disabled={!ready || busy} onClick={() => runStep(0)}>
          Start full session
        </Button>
      </CardContent>
    </Card>
  );
}
