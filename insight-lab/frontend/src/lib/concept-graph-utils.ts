import type { GraphEdge, GraphNode } from "@/lib/api";

export function filterConceptGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  weakOnly: boolean,
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  if (!weakOnly) {
    return { nodes, edges };
  }

  const weakIds = new Set(
    nodes.filter((node) => node.mastery?.status === "needs_practice").map((node) => node.id),
  );
  const filteredNodes = nodes.filter((node) => weakIds.has(node.id));
  const filteredEdges = edges.filter(
    (edge) => weakIds.has(edge.source) && weakIds.has(edge.target),
  );
  return { nodes: filteredNodes, edges: filteredEdges };
}

export function countWeakConcepts(nodes: GraphNode[]): number {
  return nodes.filter((node) => node.mastery?.status === "needs_practice").length;
}
