"use client";

import Link from "next/link";
import type { DocumentSummary } from "@/lib/api";
import { documentHref } from "@/lib/notebook-utils";
import { DocumentTypeIcon } from "@/components/ui/document-type-icon";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn } from "@/lib/utils";

type SourcesRailProps = {
  setId: string;
  documents: DocumentSummary[];
  activeDocumentId: string;
  className?: string;
};

export function SourcesRail({ setId, documents, activeDocumentId, className }: SourcesRailProps) {
  const textDocs = documents.filter((doc) => doc.file_type === "document");
  const excelDocs = documents.filter((doc) => doc.file_type === "excel");

  return (
    <aside className={cn("space-y-4", className)}>
      <div>
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sources</p>
        <p className="mt-1 text-xs text-muted-foreground">
          {documents.length} file{documents.length === 1 ? "" : "s"} in this notebook
        </p>
      </div>

      {textDocs.length > 0 ? (
        <div className="space-y-1">
          <p className="px-1 text-[11px] font-medium text-muted-foreground">Documents</p>
          {textDocs.map((doc) => {
            const active = doc.id === activeDocumentId;
            return (
              <Link
                key={doc.id}
                href={documentHref(setId, doc)}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2 py-2 text-sm transition-colors",
                  active
                    ? "bg-primary/10 text-primary ring-1 ring-primary/20"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
              >
                <DocumentTypeIcon fileType={doc.file_type} className="h-7 w-7" />
                <span className="min-w-0 flex-1 truncate">{doc.filename}</span>
                <StatusBadge status={doc.status} className="scale-90" />
              </Link>
            );
          })}
        </div>
      ) : null}

      {excelDocs.length > 0 ? (
        <div className="space-y-1">
          <p className="px-1 text-[11px] font-medium text-muted-foreground">Spreadsheets</p>
          {excelDocs.map((doc) => {
            const active = doc.id === activeDocumentId;
            return (
              <Link
                key={doc.id}
                href={documentHref(setId, doc)}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2 py-2 text-sm transition-colors",
                  active
                    ? "bg-primary/10 text-primary ring-1 ring-primary/20"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
              >
                <DocumentTypeIcon fileType={doc.file_type} className="h-7 w-7" />
                <span className="min-w-0 flex-1 truncate">{doc.filename}</span>
                <StatusBadge status={doc.status} className="scale-90" />
              </Link>
            );
          })}
        </div>
      ) : null}

      {documents.length === 0 ? (
        <p className="text-xs text-muted-foreground">No sources yet — upload from the study set page.</p>
      ) : null}
    </aside>
  );
}
