"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, type DocumentGraphResponse } from "@/lib/api";
import { ConceptGraphFlow } from "@/components/documents/concept-graph-flow";
import { countWeakConcepts, filterConceptGraph } from "@/lib/concept-graph-utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ProcessingContentSkeleton } from "@/components/ui/loading-skeletons";

type DocumentConceptGraphPanelProps = {
  documentId: string;
  ready: boolean;
  accessToken: string | null;
  variant?: "document" | "excel";
  syncPath?: string;
};

export function DocumentConceptGraphPanel({
  documentId,
  ready,
  accessToken,
  variant = "document",
  syncPath,
}: DocumentConceptGraphPanelProps) {
  const isExcel = variant === "excel";
  const resolvedSyncPath =
    syncPath ?? (isExcel ? `/documents/${documentId}/excel/graph/sync` : `/documents/${documentId}/graph/sync`);

  const [graph, setGraph] = useState<DocumentGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [weakOnly, setWeakOnly] = useState(false);

  const loadGraph = useCallback(async () => {
    if (!accessToken) {
      return;
    }
    setLoading(true);
    setError(null);
    const response = await apiFetch(`/documents/${documentId}/graph`, accessToken);
    setLoading(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Could not load concept graph (${response.status})`);
      return;
    }
    setGraph((await response.json()) as DocumentGraphResponse);
  }, [accessToken, documentId]);

  useEffect(() => {
    if (ready && accessToken) {
      void loadGraph();
    }
  }, [ready, accessToken, loadGraph]);

  async function handleSync() {
    if (!accessToken) {
      return;
    }
    setSyncing(true);
    setError(null);
    const response = await apiFetch(resolvedSyncPath, accessToken, {
      method: "POST",
    });
    setSyncing(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Could not update concept graph (${response.status})`);
      return;
    }
    await loadGraph();
  }

  const { nodes, edges } = useMemo(
    () => filterConceptGraph(graph?.nodes ?? [], graph?.edges ?? [], weakOnly),
    [graph?.edges, graph?.nodes, weakOnly],
  );

  const weakCount = useMemo(() => countWeakConcepts(graph?.nodes ?? []), [graph?.nodes]);
  const nodeCount = nodes.length;
  const connectionCount = edges.length;

  const emptyCopy = isExcel
    ? "No concepts yet. Analyze the spreadsheet, then update the graph from insights and charts."
    : "No concepts yet. They appear automatically after the document is processed.";

  const notReadyCopy = isExcel
    ? "Analyze the spreadsheet to build its concept graph."
    : "Process the document to see its concept graph.";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Concept graph</CardTitle>
        <CardDescription>
          {isExcel
            ? "Topics extracted from spreadsheet insights and charts, colored by quiz mastery when available."
            : "Topics from this document, colored by your quiz mastery. Focus on weak areas to study smarter."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" disabled={!ready || syncing} onClick={() => void handleSync()}>
            {syncing ? "Updating..." : "Update graph"}
          </Button>
          <Button type="button" variant="ghost" disabled={!ready || loading} onClick={() => void loadGraph()}>
            Refresh
          </Button>
          {weakCount > 0 ? (
            <Button
              type="button"
              size="sm"
              variant={weakOnly ? "default" : "outline"}
              onClick={() => setWeakOnly((current) => !current)}
            >
              {weakOnly ? "Show all concepts" : `Weak concepts (${weakCount})`}
            </Button>
          ) : null}
        </div>

        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        {loading && !graph ? <ProcessingContentSkeleton lines={4} /> : null}

        {graph?.migration_required && graph.notice ? (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100">
            {graph.notice}
          </p>
        ) : null}

        {graph && graph.nodes.length === 0 && !loading && !graph.migration_required ? (
          <p className="text-sm text-muted-foreground">{emptyCopy}</p>
        ) : null}

        {graph && graph.nodes.length > 0 && weakOnly && nodeCount === 0 ? (
          <p className="text-sm text-muted-foreground">
            No weak concepts right now. Take a quiz to track mastery, or show all concepts.
          </p>
        ) : null}

        {graph && nodeCount > 0 ? (
          <>
            <p className="text-sm text-muted-foreground">
              {nodeCount} concept{nodeCount === 1 ? "" : "s"}
              {connectionCount > 0
                ? ` · ${connectionCount} connection${connectionCount === 1 ? "" : "s"}`
                : ""}
            </p>
            <ConceptGraphFlow nodes={nodes} edges={edges} heightClassName="h-[420px]" />
          </>
        ) : null}

        {!ready ? <p className="text-sm text-muted-foreground">{notReadyCopy}</p> : null}
      </CardContent>
    </Card>
  );
}

/** @deprecated Use DocumentConceptGraphPanel */
export const MindMapPanel = DocumentConceptGraphPanel;
