"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { WorkspaceProgress } from "@/lib/api";

type ProgressDashboardPanelProps = {
  progress: WorkspaceProgress;
};

export function ProgressDashboardPanel({ progress }: ProgressDashboardPanelProps) {
  const items = [
    { label: "Ready files", value: `${progress.ready_count}/${progress.document_count}` },
    { label: "Quiz attempts", value: progress.quiz_attempts },
    {
      label: "Avg quiz score",
      value: progress.avg_quiz_percent != null ? `${progress.avg_quiz_percent}%` : "—",
    },
    {
      label: "Mastery avg",
      value: progress.mastery_avg_percent != null ? `${progress.mastery_avg_percent}%` : "—",
    },
    { label: "Flashcard reviews", value: progress.flashcard_reviews },
    { label: "Spreadsheets", value: progress.excel_files },
  ];

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
