"use client";

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type SourceViewerDrawerProps = {
  open: boolean;
  title: string;
  content: string;
  chunkIndex?: number | null;
  onClose: () => void;
  className?: string;
};

export function SourceViewerDrawer({
  open,
  title,
  content,
  chunkIndex,
  onClose,
  className,
}: SourceViewerDrawerProps) {
  if (!open) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-[120] bg-black/40"
        aria-label="Close source viewer"
        onClick={onClose}
      />
      <aside
        className={cn(
          "fixed inset-y-0 right-0 z-[121] flex w-[min(420px,100vw)] flex-col border-l bg-background shadow-xl",
          className,
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-viewer-title"
      >
        <div className="flex items-start justify-between gap-3 border-b px-4 py-4">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-primary">Source excerpt</p>
            <h3 id="source-viewer-title" className="truncate text-base font-semibold">
              {title}
            </h3>
            {chunkIndex != null ? (
              <p className="text-xs text-muted-foreground">Section {chunkIndex + 1}</p>
            ) : null}
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">{content}</p>
        </div>
      </aside>
    </>
  );
}
