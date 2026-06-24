"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { apiFetch, type LearningPathResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type LearningPathPanelProps = {
  setId: string;
  accessToken: string | null;
  hasReadyDocuments: boolean;
  onPathGenerated?: (pathId: string) => void;
};

function statusLabel(status: string): string {
  switch (status) {
    case "completed":
      return "Strong";
    case "needs_practice":
      return "Needs practice";
    case "locked":
      return "Locked";
    default:
      return "Next up";
  }
}

export function LearningPathPanel({
  setId,
  accessToken,
  hasReadyDocuments,
  onPathGenerated,
}: LearningPathPanelProps) {
  const [path, setPath] = useState<LearningPathResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadLatest = useCallback(async () => {
    if (!accessToken || !hasReadyDocuments) {
      return;
    }
    setLoading(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/learning-paths/latest`, accessToken);
    setLoading(false);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    setPath(data.path ?? null);
  }, [accessToken, hasReadyDocuments, setId]);

  useEffect(() => {
    void loadLatest();
  }, [loadLatest]);

  async function handleGenerate() {
    if (!accessToken) {
      return;
    }
    setGenerating(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/learning-paths/generate`, accessToken, {
      method: "POST",
    });
    setGenerating(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Could not generate learning path");
      return;
    }
    const data = (await response.json()) as LearningPathResponse;
    setPath(data);
    if (data.path_id) {
      onPathGenerated?.(data.path_id);
    }
  }

  if (!hasReadyDocuments) {
    return null;
  }

  return (
    <Card className="shadow-sm" data-tour="learning-path">
      <CardHeader>
        <CardTitle>Learning path</CardTitle>
        <CardDescription>
          Concepts ordered by prerequisites and your quiz mastery — study foundations before advanced topics.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Button type="button" disabled={generating} onClick={() => void handleGenerate()}>
            {generating ? "Building path…" : path ? "Refresh path" : "Generate learning path"}
          </Button>
          {path?.path_id ? (
            <Button type="button" variant="outline" disabled={loading} onClick={() => void loadLatest()}>
              Reload saved
            </Button>
          ) : null}
        </div>

        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        {path?.migration_required && path.notice ? (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100">
            {path.notice}
          </p>
        ) : null}

        {path && path.nodes.length > 0 ? (
          <ol className="space-y-2">
            {path.nodes.map((node, index) => (
              <li
                key={node.id}
                className={cn(
                  "flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-sm",
                  node.status === "needs_practice" && "border-rose-300/60 bg-rose-50/50 dark:bg-rose-950/20",
                  node.status === "completed" && "opacity-80",
                )}
              >
                <div className="min-w-0">
                  <p className="font-medium">
                    {index + 1}. {node.concept_name ?? node.concept_id}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {node.document_filename}
                    {node.topic ? ` · ${node.topic}` : ""} · {statusLabel(node.status)}
                    {node.mastery_percent != null ? ` (${node.mastery_percent}%)` : ""}
                  </p>
                </div>
                {node.document_id ? (
                  <Button type="button" size="sm" variant="ghost" asChild>
                    <Link href={`/dashboard/sets/${setId}/documents/${node.document_id}#mindmap`}>
                      Open
                    </Link>
                  </Button>
                ) : null}
              </li>
            ))}
          </ol>
        ) : (
          !loading &&
          !generating && (
            <p className="text-sm text-muted-foreground">
              Generate a path to see prerequisite-ordered concepts across your documents.
            </p>
          )
        )}
      </CardContent>
    </Card>
  );
}
