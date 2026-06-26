"use client";

import Link from "next/link";
import { useState } from "react";
import { ChevronLeft, ChevronRight, FileStack } from "lucide-react";
import type { DocumentSummary } from "@/lib/api";
import { documentHref } from "@/lib/notebook-utils";
import { DocumentTypeIcon } from "@/components/ui/document-type-icon";
import { StatusBadge } from "@/components/ui/status-badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type SourcesRailProps = {
  setId: string;
  documents: DocumentSummary[];
  activeDocumentId: string;
  className?: string;
  defaultCollapsed?: boolean;
};

export function SourcesRail({
  setId,
  documents,
  activeDocumentId,
  className,
  defaultCollapsed = false,
}: SourcesRailProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const textDocs = documents.filter((doc) => doc.file_type === "document");
  const excelDocs = documents.filter((doc) => doc.file_type === "excel");
  const activeDoc = documents.find((doc) => doc.id === activeDocumentId);

  if (documents.length === 0) {
    return (
      <aside className={cn("text-xs text-muted-foreground", className)}>
        No sources yet — upload from the study sheet page.
      </aside>
    );
  }

  if (collapsed) {
    return (
      <aside
        className={cn(
          "flex w-12 shrink-0 flex-col items-center gap-2 rounded-xl border bg-card/50 py-3",
          className,
        )}
      >
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          aria-label={`Expand sources (${documents.length} files)`}
          title={`Sources (${documents.length})`}
          onClick={() => setCollapsed(false)}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <FileStack className="h-4 w-4 text-muted-foreground" aria-hidden />
        <span className="text-[10px] font-medium tabular-nums text-muted-foreground">
          {documents.length}
        </span>
        {activeDoc ? (
          <DocumentTypeIcon fileType={activeDoc.file_type} className="h-7 w-7 opacity-80" />
        ) : null}
      </aside>
    );
  }

  return (
    <aside className={cn("min-w-0 space-y-3", className)}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Sources
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {documents.length} file{documents.length === 1 ? "" : "s"}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          aria-label="Collapse sources"
          onClick={() => setCollapsed(true)}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      {textDocs.length > 0 ? (
        <div className="space-y-1">
          {textDocs.length > 0 && excelDocs.length > 0 ? (
            <p className="px-1 text-[11px] font-medium text-muted-foreground">Documents</p>
          ) : null}
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
          {textDocs.length > 0 ? (
            <p className="px-1 text-[11px] font-medium text-muted-foreground">Spreadsheets</p>
          ) : null}
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
    </aside>
  );
}
