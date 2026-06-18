"use client";

import { cn } from "@/lib/utils";
import type { DocumentReviewOption } from "@/lib/api";

type DocumentReviewPickerProps = {
  documents: DocumentReviewOption[];
  selectedIds: Set<string>;
  onToggle: (documentId: string) => void;
  className?: string;
};

export function DocumentReviewPicker({
  documents,
  selectedIds,
  onToggle,
  className,
}: DocumentReviewPickerProps) {
  if (documents.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-3", className)}>
      {documents.map((document) => {
        const checked = selectedIds.has(document.document_id);
        return (
          <label
            key={document.document_id}
            className={cn(
              "block cursor-pointer rounded-md border p-4 transition-colors",
              checked ? "border-primary/40 bg-muted/20" : "opacity-70",
            )}
          >
            <div className="flex items-start gap-3">
              <input
                type="checkbox"
                className="mt-1"
                checked={checked}
                onChange={() => onToggle(document.document_id)}
                aria-label={`Use ${document.filename} to answer`}
              />
              <div className="min-w-0 flex-1">
                <p className="font-medium">{document.filename}</p>
                <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">
                  {document.summary}
                </p>
              </div>
            </div>
          </label>
        );
      })}
    </div>
  );
}
