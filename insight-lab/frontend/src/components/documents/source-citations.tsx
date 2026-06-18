"use client";

import { cn } from "@/lib/utils";
import type { SourceCitation } from "@/lib/api";

type SourceCitationsProps = {
  sources: SourceCitation[];
  selectable?: boolean;
  selectedKeys?: Set<string>;
  onToggle?: (source: SourceCitation) => void;
  className?: string;
};

function sourceKey(source: SourceCitation): string {
  return `${source.document_id}:${source.chunk_index ?? source.id}`;
}

export function SourceCitations({
  sources,
  selectable = false,
  selectedKeys,
  onToggle,
  className,
}: SourceCitationsProps) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sources</p>
      {sources.map((source) => {
        const key = sourceKey(source);
        const checked = selectable ? (selectedKeys?.has(key) ?? source.selected ?? true) : true;
        return (
          <div
            key={key}
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
                <p className="font-medium">{source.filename}</p>
                <p className="mt-1 whitespace-pre-wrap text-muted-foreground">{source.preview}</p>
                {source.similarity != null && source.similarity > 0 && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    Match strength: {Math.round(source.similarity * 100)}%
                  </p>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export { sourceKey };
