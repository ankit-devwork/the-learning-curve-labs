"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, type DocumentGraphResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type ConceptGraphPanelProps = {
  documentId: string;
  ready: boolean;
  accessToken: string | null;
};

export function ConceptGraphPanel({ documentId, ready, accessToken }: ConceptGraphPanelProps) {
  const [graph, setGraph] = useState<DocumentGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      setError(body.error || `Failed to load graph (${response.status})`);
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
      setError(body.error || `Graph sync failed (${response.status})`);
      return;
    }
    await loadGraph();
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Concept graph</CardTitle>
        <CardDescription>
          Key concepts extracted from this document. Syncs to Neo4j when configured.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" disabled={!ready || syncing} onClick={() => void handleSync()}>
            {syncing ? "Syncing..." : "Sync concepts"}
          </Button>
          <Button type="button" variant="ghost" disabled={!ready || loading} onClick={() => void loadGraph()}>
            Refresh
          </Button>
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}
        {loading && <p className="text-sm text-muted-foreground">Loading concept graph...</p>}

        {graph && graph.nodes.length === 0 && !loading && (
          <p className="text-sm text-muted-foreground">
            No concepts yet. Processing auto-syncs concepts; use Sync if needed.
          </p>
        )}

        {graph && graph.nodes.length > 0 && (
          <div className="space-y-4">
            <p className="text-xs text-muted-foreground">
              {graph.nodes.length} concepts · {graph.edges.length} relationships
              {graph.neo4j_synced_at ? " · Neo4j synced" : ""}
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              {graph.nodes.map((node) => (
                <div key={node.id} className="rounded-md border p-3 text-sm">
                  <p className="font-medium">{node.label}</p>
                  {node.topic && <p className="text-xs text-muted-foreground">Topic: {node.topic}</p>}
                </div>
              ))}
            </div>
            {graph.edges.length > 0 && (
              <div className="space-y-1 text-xs text-muted-foreground">
                <p className="font-medium text-foreground">Relationships</p>
                {graph.edges.map((edge, index) => (
                  <p key={`${edge.source}-${edge.target}-${index}`}>
                    {edge.source} → {edge.target} ({edge.type.replace(/_/g, " ")})
                  </p>
                ))}
              </div>
            )}
          </div>
        )}

        {!ready && (
          <p className="text-sm text-muted-foreground">Process the document to extract concepts.</p>
        )}
      </CardContent>
    </Card>
  );
}
