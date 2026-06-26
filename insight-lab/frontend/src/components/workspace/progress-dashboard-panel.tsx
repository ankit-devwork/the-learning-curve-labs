"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { WorkspaceProgress } from "@/lib/api";

type ProgressDashboardPanelProps = {
  progress: WorkspaceProgress;
  compact?: boolean;
};

export function ProgressDashboardPanel({ progress, compact = false }: ProgressDashboardPanelProps) {
  const items = [
    { label: "Ready", value: `${progress.ready_count}/${progress.document_count}` },
    { label: "Quizzes", value: progress.quiz_attempts },
    {
      label: "Avg score",
      value: progress.avg_quiz_percent != null ? `${progress.avg_quiz_percent}%` : "—",
    },
    {
      label: "Mastery",
      value: progress.mastery_avg_percent != null ? `${progress.mastery_avg_percent}%` : "—",
    },
    { label: "Cards", value: progress.flashcard_reviews },
    { label: "Sheets", value: progress.excel_files },
  ];

  if (compact) {
    return (
      <div
        className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border bg-card px-4 py-3 text-sm shadow-sm"
        data-tour="progress-dashboard"
      >
        <span className="font-medium text-muted-foreground">Progress</span>
        {items.map((item) => (
          <span key={item.label} className="tabular-nums">
            <span className="text-muted-foreground">{item.label}</span>{" "}
            <span className="font-semibold">{item.value}</span>
          </span>
        ))}
        <span className="hidden h-4 w-px bg-border sm:inline" aria-hidden />
        <span className="min-w-0 flex-1 text-muted-foreground">
          <span className="font-medium text-foreground">Next:</span> {progress.study_next.label}
        </span>
        {progress.study_next.action === "adaptive_quiz" ? (
          <Button type="button" size="sm" variant="outline" asChild>
            <Link href="#set-quiz">Practice quiz</Link>
          </Button>
        ) : null}
      </div>
    );
  }

  return (
    <Card className="shadow-sm" data-tour="progress-dashboard">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Progress</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {items.map((item) => (
            <div key={item.label} className="rounded-lg border bg-muted/20 px-3 py-2">
              <dt className="text-xs text-muted-foreground">{item.label}</dt>
              <dd className="mt-1 text-lg font-semibold">{item.value}</dd>
            </div>
          ))}
        </dl>

        {progress.weak_concepts.length > 0 ? (
          <div>
            <p className="text-sm font-medium">Topics to review</p>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {progress.weak_concepts.map((concept) => (
                <li key={concept.concept_id}>
                  {concept.name || concept.concept_id}
                  {concept.percent != null ? ` · ${concept.percent}%` : ""}
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-primary/5 px-3 py-2">
          <p className="text-sm">
            <span className="font-medium">Study next:</span> {progress.study_next.label}
          </p>
          {progress.study_next.action === "adaptive_quiz" ? (
            <Button type="button" size="sm" variant="outline" asChild>
              <Link href="#set-quiz">Go to practice quiz</Link>
            </Button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}
