"use client";

import { useMemo } from "react";
import Link from "next/link";
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
import type { GraphEdge, GraphNode } from "@/lib/api";
import { cn } from "@/lib/utils";

const NODE_WIDTH = 176;
const NODE_HEIGHT = 52;
const H_GAP = 48;
const V_GAP = 72;
const TOPIC_BAND = 96;

export const TOPIC_COLORS = [
  "border-blue-400/60 bg-blue-500/10 text-blue-900 dark:text-blue-100",
  "border-violet-400/60 bg-violet-500/10 text-violet-900 dark:text-violet-100",
  "border-emerald-400/60 bg-emerald-500/10 text-emerald-900 dark:text-emerald-100",
  "border-amber-400/60 bg-amber-500/10 text-amber-900 dark:text-amber-100",
  "border-rose-400/60 bg-rose-500/10 text-rose-900 dark:text-rose-100",
  "border-cyan-400/60 bg-cyan-500/10 text-cyan-900 dark:text-cyan-100",
] as const;

type ConceptNodeData = {
  label: string;
  topic: string;
  documentFilename?: string;
  masteryStatus?: "untested" | "needs_practice" | "strong";
  href?: string;
};

function masteryBorderClass(status: ConceptNodeData["masteryStatus"]): string {
  switch (status) {
    case "needs_practice":
      return "border-rose-500 ring-1 ring-rose-500/30";
    case "strong":
      return "border-emerald-500 ring-1 ring-emerald-500/30";
    default:
      return "";
  }
}

function ConceptNode({ data }: NodeProps<Node<ConceptNodeData>>) {
  const inner = (
    <div
      className={cn(
        "rounded-lg border bg-card px-3 py-2 shadow-sm",
        masteryBorderClass(data.masteryStatus),
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-primary" />
      <p className="max-w-[148px] truncate text-xs font-medium">{data.label}</p>
      <p className="max-w-[148px] truncate text-[10px] text-muted-foreground">{data.topic}</p>
      {data.documentFilename ? (
        <p className="max-w-[148px] truncate text-[10px] text-muted-foreground/80">{data.documentFilename}</p>
      ) : null}
      <Handle type="source" position={Position.Bottom} className="!bg-primary" />
    </div>
  );

  if (data.href) {
    return (
      <Link href={data.href} className="block no-underline">
        {inner}
      </Link>
    );
  }

  return inner;
}

const nodeTypes = { concept: ConceptNode };

export function topicColorIndex(topic: string): number {
  let hash = 0;
  for (let index = 0; index < topic.length; index += 1) {
    hash = topic.charCodeAt(index) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % TOPIC_COLORS.length;
}

export function layoutConceptGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  options?: {
    getNodeHref?: (node: GraphNode) => string | undefined;
    groupByDocument?: boolean;
  },
): { nodes: Node<ConceptNodeData>[]; edges: Edge[] } {
  const groups = new Map<string, GraphNode[]>();

  for (const node of nodes) {
    let groupKey: string;
    if (options?.groupByDocument && node.document_filename) {
      groupKey = node.document_filename;
    } else {
      groupKey = node.topic?.trim() || "General";
    }
    const existing = groups.get(groupKey);
    if (existing) {
      existing.push(node);
    } else {
      groups.set(groupKey, [node]);
    }
  }

  const sortedGroups = Array.from(groups.entries()).sort(([left], [right]) => left.localeCompare(right));
  const flowNodes: Node<ConceptNodeData>[] = [];
  let bandIndex = 0;

  for (const [, groupNodes] of sortedGroups) {
    const sortedNodes = [...groupNodes].sort((left, right) => left.label.localeCompare(right.label));
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
        data: {
          label: node.label,
          topic: node.topic?.trim() || "General",
          documentFilename: options?.groupByDocument ? undefined : node.document_filename ?? undefined,
          masteryStatus: node.mastery?.status,
          href: options?.getNodeHref?.(node),
        },
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

type ConceptGraphFlowProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  heightClassName?: string;
  groupByDocument?: boolean;
  getNodeHref?: (node: GraphNode) => string | undefined;
};

export function ConceptGraphFlow({
  nodes,
  edges,
  heightClassName = "h-[420px]",
  groupByDocument = false,
  getNodeHref,
}: ConceptGraphFlowProps) {
  const layout = useMemo(
    () => layoutConceptGraph(nodes, edges, { getNodeHref, groupByDocument }),
    [nodes, edges, getNodeHref, groupByDocument],
  );

  const topicLegend = useMemo(() => {
    const topics = new Set<string>();
    for (const node of nodes) {
      topics.add(node.topic?.trim() || "General");
    }
    return Array.from(topics).sort((left, right) => left.localeCompare(right));
  }, [nodes]);

  if (nodes.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      {topicLegend.length > 1 ? (
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
      ) : null}

      <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full border border-emerald-500" /> Strong
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full border border-rose-500" /> Needs practice
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full border border-border" /> Untested
        </span>
      </div>

      <div className={cn("overflow-hidden rounded-lg border bg-muted/20", heightClassName)}>
        <ReactFlow
          nodes={layout.nodes}
          edges={layout.edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.25}
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
  );
}
