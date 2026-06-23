"use client";

import Link from "next/link";
import type { DocumentSummary } from "@/lib/api";
import { documentHref } from "@/lib/notebook-utils";
import { DocumentTypeIcon } from "@/components/ui/document-type-icon";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/utils";

type SourcesStripProps = {
  setId: string;
  documents: DocumentSummary[];
  className?: string;
};

export function SourcesStrip({ setId, documents, className }: SourcesStripProps) {
  if (documents.length === 0) {
    return null;
  }

  return (
    <div className={cn("notebook-surface rounded-xl p-4", className)} data-tour="sources-strip">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div>
          <p className="text-sm font-semibold">Sources</p>
          <p className="text-xs text-muted-foreground">
            {documents.filter((doc) => doc.status === "ready").length} of {documents.length} ready
          </p>
        </div>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {documents.map((doc) => (
          <Link
            key={doc.id}
            href={documentHref(setId, doc)}
            className="flex min-w-[200px] max-w-[240px] shrink-0 items-center gap-3 rounded-lg border bg-background/80 px-3 py-2.5 transition-shadow hover:shadow-sm"
          >
            <DocumentTypeIcon fileType={doc.file_type} />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{doc.filename}</p>
              <p className="text-[11px] capitalize text-muted-foreground">{doc.file_type}</p>
            </div>
            <StatusBadge status={doc.status} />
          </Link>
        ))}
      </div>
    </div>
  );
}
