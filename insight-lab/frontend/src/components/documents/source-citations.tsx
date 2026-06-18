"use client";

import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SourceCitation } from "@/lib/api";
import { Button } from "@/components/ui/button";

type SourceCitationsProps = {
  sources: SourceCitation[];
  selectable?: boolean;
  selectedKeys?: Set<string>;
  onToggle?: (source: SourceCitation) => void;
  groupByDocument?: boolean;
  className?: string;
};

function sourceKey(source: SourceCitation): string {
  return `${source.document_id}:${source.chunk_index ?? source.id}`;
}

function SourceCard({
  source,
  selectable,
  checked,
  onToggle,
}: {
  source: SourceCitation;
  selectable: boolean;
  checked: boolean;
  onToggle?: (source: SourceCitation) => void;
}) {
  return (
    <div
      className={cn(
        "rounded-md border bg-muted/20 p-3 text-sm",
        selectable && !checked && "opacity-60",
      )}
    >
      <div className="flex items-start gap-2">
        {selectable && onToggle && (
          <input
            type="checkbox"
            className="mt-1"
            checked={checked}
            onChange={() => onToggle(source)}
                    aria-label={`Include source from ${source.filename}`}
          />
        )}
        <div className="min-w-0 flex-1">
          {!selectable && <p className="font-medium">{source.filename}</p>}
          <p className={cn("whitespace-pre-wrap text-muted-foreground", !selectable && "mt-1")}>
            {source.preview}
          </p>
          {source.similarity != null && source.similarity > 0 && (
            <p className="mt-1 text-xs text-muted-foreground">
              Match strength: {Math.round(source.similarity * 100)}%
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export function SourceCitations({
  sources,
  selectable = false,
  selectedKeys,
  onToggle,
  groupByDocument,
  className,
}: SourceCitationsProps) {
  const [collapsedDocs, setCollapsedDocs] = useState<Set<string>>(new Set());

  const groups = useMemo(() => {
    const byDocument = new Map<string, { filename: string; sources: SourceCitation[] }>();
    for (const source of sources) {
      const existing = byDocument.get(source.document_id);
      if (existing) {
        existing.sources.push(source);
      } else {
        byDocument.set(source.document_id, { filename: source.filename, sources: [source] });
      }
    }
    return Array.from(byDocument.entries());
  }, [sources]);

  const shouldGroup =
    groupByDocument ?? (selectable && groups.length > 1);

  if (sources.length === 0) {
    return null;
  }

  function toggleGroup(documentId: string) {
    setCollapsedDocs((current) => {
      const next = new Set(current);
      if (next.has(documentId)) {
        next.delete(documentId);
      } else {
        next.add(documentId);
      }
      return next;
    });
  }

  function toggleGroupSelection(documentId: string, selectAll: boolean) {
    if (!onToggle) {
      return;
    }
    const group = groups.find(([id]) => id === documentId);
    if (!group) {
      return;
    }
    for (const source of group[1].sources) {
      const key = sourceKey(source);
      const isSelected = selectedKeys?.has(key) ?? source.selected ?? true;
      if (selectAll && !isSelected) {
        onToggle(source);
      }
      if (!selectAll && isSelected) {
        onToggle(source);
      }
    }
  }

  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sources</p>
      {shouldGroup
        ? groups.map(([documentId, { filename, sources: docSources }]) => {
            const collapsed = collapsedDocs.has(documentId);
            const selectedInGroup = docSources.filter((source) => {
              const key = sourceKey(source);
              return selectable ? (selectedKeys?.has(key) ?? source.selected ?? true) : true;
            }).length;

            return (
              <div key={documentId} className="rounded-md border">
                <div className="flex items-center justify-between gap-2 border-b bg-muted/30 px-3 py-2">
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-2 text-left text-sm font-medium"
                    onClick={() => toggleGroup(documentId)}
                    aria-expanded={!collapsed}
                  >
                    {collapsed ? (
                      <ChevronRight className="size-4 shrink-0" />
                    ) : (
                      <ChevronDown className="size-4 shrink-0" />
                    )}
                    <span className="truncate">{filename}</span>
                    <span className="shrink-0 text-xs font-normal text-muted-foreground">
                      {docSources.length} result{docSources.length === 1 ? "" : "s"}
                      {selectable && ` · ${selectedInGroup} selected`}
                    </span>
                  </button>
                  {selectable && onToggle && (
                    <div className="flex shrink-0 gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => toggleGroupSelection(documentId, true)}
                      >
                        All
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => toggleGroupSelection(documentId, false)}
                      >
                        None
                      </Button>
                    </div>
                  )}
                </div>
                {!collapsed && (
                  <div className="space-y-2 p-2">
                    {docSources.map((source) => {
                      const key = sourceKey(source);
                      const checked = selectable
                        ? (selectedKeys?.has(key) ?? source.selected ?? true)
                        : true;
                      return (
                        <SourceCard
                          key={key}
                          source={source}
                          selectable={selectable}
                          checked={checked}
                          onToggle={onToggle}
                        />
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })
        : sources.map((source) => {
            const key = sourceKey(source);
            const checked = selectable
              ? (selectedKeys?.has(key) ?? source.selected ?? true)
              : true;
            return (
              <SourceCard
                key={key}
                source={source}
                selectable={selectable}
                checked={checked}
                onToggle={onToggle}
              />
            );
          })}
    </div>
  );
}

export { sourceKey };
