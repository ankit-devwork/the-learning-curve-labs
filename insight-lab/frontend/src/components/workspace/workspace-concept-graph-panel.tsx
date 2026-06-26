"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, type WorkspaceGraphResponse } from "@/lib/api";
import { ConceptGraphFlow } from "@/components/documents/concept-graph-flow";
import { countWeakConcepts, filterConceptGraph } from "@/lib/concept-graph-utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ProcessingContentSkeleton } from "@/components/ui/loading-skeletons";
import { Label } from "@/components/ui/label";

type WorkspaceConceptGraphPanelProps = {
  setId: string;
  accessToken: string | null;
  hasReadyDocuments: boolean;
  embedded?: boolean;
};

export function WorkspaceConceptGraphPanel({
  setId,
  accessToken,
  hasReadyDocuments,
  embedded = false,
}: WorkspaceConceptGraphPanelProps) {
  const [graph, setGraph] = useState<WorkspaceGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [documentFilter, setDocumentFilter] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"topics" | "documents">("topics");
  const [weakOnly, setWeakOnly] = useState(false);

  const loadGraph = useCallback(async () => {
    if (!accessToken || !hasReadyDocuments) {
      return;
    }
    setLoading(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/graph`, accessToken);
    setLoading(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Could not load concept graph (${response.status})`);
      return;
    }
    setGraph((await response.json()) as WorkspaceGraphResponse);
  }, [accessToken, hasReadyDocuments, setId]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  const filteredNodes = useMemo(() => {
    if (!graph) {
      return [];
    }
    if (documentFilter === "all") {
      return graph.nodes;
    }
    return graph.nodes.filter((node) => node.document_id === documentFilter);
  }, [documentFilter, graph]);

  const filteredEdges = useMemo(() => {
    if (!graph) {
      return [];
    }
    if (documentFilter === "all") {
      return graph.edges;
    }
    return graph.edges.filter((edge) => edge.document_id === documentFilter);
  }, [documentFilter, graph]);

  const weakCount = useMemo(() => countWeakConcepts(filteredNodes), [filteredNodes]);

  const { nodes: displayNodes, edges: displayEdges } = useMemo(
    () => filterConceptGraph(filteredNodes, filteredEdges, weakOnly),
    [filteredEdges, filteredNodes, weakOnly],
  );

  const nodeCount = displayNodes.length;
  const edgeCount = displayEdges.length;

  const content = (
    <div className="space-y-4">
      {!hasReadyDocuments ? (
        <p className="text-sm text-muted-foreground">
          Upload and process PDF or Word documents to build a concept graph.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <Label htmlFor="graph-doc-filter">Document</Label>
              <select
                id="graph-doc-filter"
                className="flex h-10 min-w-[200px] rounded-md border bg-background px-3 text-sm"
                value={documentFilter}
                onChange={(event) => setDocumentFilter(event.target.value)}
              >
                <option value="all">All documents</option>
                {(graph?.documents ?? []).map((doc) => (
                  <option key={doc.document_id} value={doc.document_id}>
                    {doc.filename} ({doc.concept_count})
                  </option>
                ))}
              </select>
            </div>
            <div className="flex gap-2">
              <Button
                type="button"
                size="sm"
                variant={viewMode === "topics" ? "default" : "outline"}
                onClick={() => setViewMode("topics")}
              >
                By topic
              </Button>
              <Button
                type="button"
                size="sm"
                variant={viewMode === "documents" ? "default" : "outline"}
                onClick={() => setViewMode("documents")}
              >
                By document
              </Button>
            </div>
            <Button type="button" variant="ghost" size="sm" disabled={loading} onClick={() => void loadGraph()}>
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

          {graph && nodeCount === 0 && !loading && !graph.migration_required ? (
            <p className="text-sm text-muted-foreground">
              No concepts yet. They are extracted automatically when documents finish processing.
            </p>
          ) : null}

          {graph && filteredNodes.length > 0 && weakOnly && nodeCount === 0 ? (
            <p className="text-sm text-muted-foreground">
              No weak concepts in this view. Try another document filter or show all concepts.
            </p>
          ) : null}

          {graph && nodeCount > 0 ? (
            <>
              <p className="text-sm text-muted-foreground">
                {nodeCount} concept{nodeCount === 1 ? "" : "s"}
                {edgeCount > 0 ? ` · ${edgeCount} connection${edgeCount === 1 ? "" : "s"}` : ""}
                {graph.stats ? ` · ${graph.stats.document_count} document${graph.stats.document_count === 1 ? "" : "s"}` : ""}
              </p>
              <ConceptGraphFlow
                nodes={displayNodes}
                edges={displayEdges}
                groupByDocument={viewMode === "documents" && documentFilter === "all"}
                getNodeHref={(node) =>
                  node.document_id
                    ? `/dashboard/sets/${setId}/documents/${node.document_id}#concepts`
                    : undefined
                }
                heightClassName={embedded ? "h-[360px]" : "h-[480px]"}
              />
            </>
          ) : null}
        </>
      )}
    </div>
  );

  if (embedded) {
    return content;
  }

  return (
    <Card className="shadow-sm" data-tour="concept-graph">
      <CardHeader>
        <CardTitle>Concept graph</CardTitle>
        <CardDescription>
          Topics extracted across your documents, colored by quiz mastery. Click a concept to open its source.
        </CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}
