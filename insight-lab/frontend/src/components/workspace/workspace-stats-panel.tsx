"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { WorkspaceStats } from "@/lib/api";

type WorkspaceStatsPanelProps = {
  stats: WorkspaceStats;
};

export function WorkspaceStatsPanel({ stats }: WorkspaceStatsPanelProps) {
  const items = [
    { label: "Files", value: stats.document_count },
    { label: "Ready", value: stats.ready_count },
    { label: "Documents", value: stats.document_files },
    { label: "Spreadsheets", value: stats.excel_files },
    { label: "Quiz attempts", value: stats.quiz_attempts },
    {
      label: "Avg quiz score",
      value: stats.avg_quiz_percent != null ? `${stats.avg_quiz_percent}%` : "—",
    },
  ];

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {items.map((item) => (
            <div key={item.label} className="rounded-lg border bg-muted/20 px-3 py-2">
              <dt className="text-xs text-muted-foreground">{item.label}</dt>
              <dd className="mt-1 text-lg font-semibold">{item.value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}
