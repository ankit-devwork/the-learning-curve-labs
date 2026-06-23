"use client";

import type { SourceCitation } from "@/lib/api";

type CitationChipsProps = {
  sources: SourceCitation[];
  onCitationClick?: (source: SourceCitation, index: number) => void;
};

export function CitationChips({ sources, onCitationClick }: CitationChipsProps) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      {sources.map((source, index) => (
        <button
          key={`${source.document_id}-${source.chunk_index ?? source.id}-${index}`}
          type="button"
          className="inline-flex max-w-full items-center gap-1 rounded-full bg-primary/10 px-2.5 py-1 text-left text-[11px] font-medium text-primary transition-colors hover:bg-primary/15"
          title={source.preview}
          onClick={() => onCitationClick?.(source, index)}
        >
          <span className="font-semibold">[{index + 1}]</span>
          <span className="truncate">{source.filename}</span>
        </button>
      ))}
    </div>
  );
}
