"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { apiFetch, type DocumentGraphResponse, type GraphEdge, type GraphNode } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ProcessingContentSkeleton } from "@/components/ui/loading-skeletons";
import { cn } from "@/lib/utils";

type MindMapPanelProps = {
  documentId: string;
  ready: boolean;
  accessToken: string | null;
};

type ConceptNodeData = {
  label: string;
  topic: string;
};

const NODE_WIDTH = 176;
const NODE_HEIGHT = 44;
const H_GAP = 48;
const V_GAP = 72;
const TOPIC_BAND = 96;

const TOPIC_COLORS = [
  "border-blue-400/60 bg-blue-500/10 text-blue-900 dark:text-blue-100",
  "border-violet-400/60 bg-violet-500/10 text-violet-900 dark:text-violet-100",
  "border-emerald-400/60 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100",
  "border-amber-400/60 bg-amber-500/10 text-amber-900 dark:text-amber-100",
  "border-rose-400/60 bg-rose-500/10 text-rose-900 dark:text-rose-100",
  "border-cyan-400/60 bg-cyan-500/10 text-cyan-900 dark:text-cyan-100",
] as const;

function ConceptNode({ data }: NodeProps<Node<ConceptNodeData>>) {
  return (
    <div className="rounded-lg border bg-card px-3 py-2 shadow-sm">
      <Handle type="target" position={Position.Top} className="!bg-primary" />
      <p className="max-w-[148px] truncate text-xs font-medium">{data.label}</p>
      <p className="max-w-[148px] truncate text-[10px] text-muted-foreground">{data.topic}</p>
      <Handle type="source" position={Position.Bottom} className="!bg-primary" />
    </div>
  );
}

const nodeTypes = { concept: ConceptNode };

function topicColorIndex(topic: string): number {
  let hash = 0;
  for (let index = 0; index < topic.length; index += 1) {
    hash = topic.charCodeAt(index) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % TOPIC_COLORS.length;
}

function layoutMindMap(nodes: GraphNode[], edges: GraphEdge[]): { nodes: Node<ConceptNodeData>[]; edges: Edge[] } {
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

  const sortedTopics = Array.from(groups.entries()).sort(([left], [right]) => left.localeCompare(right));
  const flowNodes: Node<ConceptNodeData>[] = [];
  let bandIndex = 0;

  for (const [topic, topicNodes] of sortedTopics) {
    const sortedNodes = [...topicNodes].sort((left, right) => left.label.localeCompare(right.label));
    const cols = Math.min(Math.max(sortedNodes.length, 1), 4);

    sortedNodes.forEach((node, index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      flowNodes.push({
        id: node.id,
        type: "concept",
        position: {
          x: col * (NODE_WIDTH + H_GAP),
          y: bandIndex * TOPIC_BAND + row * (NODE_HEIGHT + V_GAP),
        },
        data: { label: node.label, topic },
      });
    });

    const rows = Math.ceil(sortedNodes.length / cols);
    bandIndex += Math.max(rows, 1) + 0.5;
  }

  const flowEdges: Edge[] = edges.map((edge, index) => ({
    id: `edge-${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    type: "smoothstep",
    animated: edge.type === "prerequisite_for",
    label: edge.type === "prerequisite_for" ? "builds on" : edge.type === "belongs_to" ? "part of" : undefined,
    style: {
      stroke: edge.type === "related_to" ? "hsl(var(--muted-foreground))" : "hsl(var(--primary))",
      strokeDasharray: edge.type === "related_to" ? "4 4" : undefined,
    },
  }));

  return { nodes: flowNodes, edges: flowEdges };
}

export function MindMapPanel({ documentId, ready, accessToken }: MindMapPanelProps) {
  const [graph, setGraph] = useState<DocumentGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { nodes, edges } = useMemo(
    () => (graph && graph.nodes.length > 0 ? layoutMindMap(graph.nodes, graph.edges) : { nodes: [], edges: [] }),
    [graph],
  );

  const topicLegend = useMemo(() => {
    const topics = new Set<string>();
    for (const node of graph?.nodes ?? []) {
      topics.add(node.topic?.trim() || "General");
    }
    return Array.from(topics).sort((left, right) => left.localeCompare(right));
  }, [graph?.nodes]);

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
      setError(body.error || `Could not load mind map (${response.status})`);
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
      setError(body.error || `Could not update mind map (${response.status})`);
      return;
    }
    await loadGraph();
  }

  const topicCount = graph?.nodes.length ?? 0;
  const connectionCount = graph?.edges.length ?? 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Mind map</CardTitle>
        <CardDescription>
          Visual map of main ideas and how they connect.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" disabled={!ready || syncing} onClick={() => void handleSync()}>
            {syncing ? "Updating..." : "Update map"}
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
            No concepts yet. They appear automatically after the document is processed.
          </p>
        )}

        {graph && graph.nodes.length > 0 && (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              {topicCount} concept{topicCount === 1 ? "" : "s"}
              {connectionCount > 0
                ? ` · ${connectionCount} connection${connectionCount === 1 ? "" : "s"}`
                : ""}
            </p>

            {topicLegend.length > 1 && (
              <div className="flex flex-wrap gap-2">
                {topicLegend.map((topic) => (
                  <span
                    key={topic}
                    className={cn(
                      "rounded-full border px-2 py-0.5 text-[11px] font-medium",
                      TOPIC_COLORS[topicColorIndex(topic)],
                    )}
                  >
                    {topic}
                  </span>
                ))}
              </div>
            )}

            <div className="h-[420px] overflow-hidden rounded-lg border bg-muted/20">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.2 }}
                minZoom={0.3}
                maxZoom={1.5}
                nodesDraggable
                nodesConnectable={false}
                elementsSelectable
                proOptions={{ hideAttribution: true }}
              >
                <Background gap={16} size={1} />
                <Controls showInteractive={false} />
                <MiniMap pannable zoomable nodeStrokeWidth={2} className="!bg-background/80" />
              </ReactFlow>
            </div>
          </div>
        )}

        {!ready && (
          <p className="text-sm text-muted-foreground">Process the document to see its mind map.</p>
        )}
      </CardContent>
    </Card>
  );
}
