"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, type DocumentGraphResponse, type GraphEdge, type GraphNode } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ProcessingContentSkeleton } from "@/components/ui/loading-skeletons";

type ConceptGraphPanelProps = {
  documentId: string;
  ready: boolean;
  accessToken: string | null;
};

function humanizeConceptId(id: string): string {
  return id
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function conceptLabel(id: string, labels: Map<string, string>): string {
  return labels.get(id) ?? humanizeConceptId(id);
}

function formatRelationship(edge: GraphEdge, labels: Map<string, string>): string {
  const source = conceptLabel(edge.source, labels);
  const target = conceptLabel(edge.target, labels);

  switch (edge.type) {
    case "belongs_to":
      return `${source} is part of ${target}`;
    case "prerequisite_for":
      return `${target} builds on ${source}`;
    case "related_to":
      return `${source} is related to ${target}`;
    default:
      return `${source} connects to ${target}`;
  }
}

function groupNodesByTopic(nodes: GraphNode[]): Array<{ topic: string; nodes: GraphNode[] }> {
  const groups = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const topic = node.topic?.trim() || "General";
    const existing = groups.get(topic);
    if (existing) {
      existing.push(node);
    } else {
      groups.set(topic, [node]);
    }
  }

  return Array.from(groups.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([topic, topicNodes]) => ({
      topic,
      nodes: topicNodes.sort((left, right) => left.label.localeCompare(right.label)),
    }));
}

export function ConceptGraphPanel({ documentId, ready, accessToken }: ConceptGraphPanelProps) {
  const [graph, setGraph] = useState<DocumentGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const labelById = useMemo(() => {
    const labels = new Map<string, string>();
    for (const node of graph?.nodes ?? []) {
      labels.set(node.id, node.label);
    }
    return labels;
  }, [graph?.nodes]);

  const topicGroups = useMemo(
    () => (graph ? groupNodesByTopic(graph.nodes) : []),
    [graph],
  );

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
      setError(body.error || `Could not load topics (${response.status})`);
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
    const response = await apiFetch(`/documents/${documentId}/graph/sync`, accessToken, {
      method: "POST",
    });
    setSyncing(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Could not update topics (${response.status})`);
      return;
    }
    await loadGraph();
  }

  const topicCount = graph?.nodes.length ?? 0;
  const connectionCount = graph?.edges.length ?? 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Key topics</CardTitle>
        <CardDescription>
          Main ideas from this document and how they connect to each other.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" disabled={!ready || syncing} onClick={() => void handleSync()}>
            {syncing ? "Updating..." : "Update topics"}
          </Button>
          <Button type="button" variant="ghost" disabled={!ready || loading} onClick={() => void loadGraph()}>
            Refresh
          </Button>
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}
        {loading && <ProcessingContentSkeleton lines={4} />}

        {graph?.migration_required && graph.notice && (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100">
            {graph.notice}
          </p>
        )}

        {graph && graph.nodes.length === 0 && !loading && !graph.migration_required && (
          <p className="text-sm text-muted-foreground">
            No topics yet. They appear automatically after the document is processed.
          </p>
        )}

        {graph && graph.nodes.length > 0 && (
          <div className="space-y-5">
            <p className="text-sm text-muted-foreground">
              {topicCount} topic{topicCount === 1 ? "" : "s"}
              {connectionCount > 0
                ? ` with ${connectionCount} connection${connectionCount === 1 ? "" : "s"}`
                : ""}
            </p>

            <div className="space-y-4">
              {topicGroups.map(({ topic, nodes }) => (
                <div key={topic} className="space-y-2">
                  <p className="text-sm font-medium">{topic}</p>
                  <ul className="grid gap-2 sm:grid-cols-2">
                    {nodes.map((node) => (
                      <li key={node.id} className="rounded-md border bg-muted/20 px-3 py-2 text-sm">
                        {node.label}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            {graph.edges.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">How topics connect</p>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  {graph.edges.map((edge, index) => (
                    <li key={`${edge.source}-${edge.target}-${index}`} className="rounded-md border px-3 py-2">
                      {formatRelationship(edge, labelById)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {!ready && (
          <p className="text-sm text-muted-foreground">Process the document to see its key topics.</p>
        )}
      </CardContent>
    </Card>
  );
}
