"use client";

import Link from "next/link";
import {
  BookOpen,
  Brain,
  FileText,
  Layers,
  Volume2,
} from "lucide-react";
import type { CoursePackDocumentResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type CoursePackResultsProps = {
  setId: string;
  documents: CoursePackDocumentResult[];
};

const ARTIFACTS = [
  { key: "summary", label: "Brief", icon: FileText, hash: "brief" },
  { key: "quiz_id", label: "Quiz", icon: Brain, hash: "quiz" },
  { key: "flashcard_set_id", label: "Flashcards", icon: Layers, hash: "flashcards" },
  { key: "study_guide_id", label: "Study guide", icon: BookOpen, hash: "guide" },
  { key: "audio_script", label: "Audio", icon: Volume2, hash: "audio" },
] as const;

export function CoursePackResults({ setId, documents }: CoursePackResultsProps) {
  return (
    <ul className="space-y-4">
      {documents.map((item) => (
        <li key={item.document_id} className="notebook-surface rounded-xl p-4">
          <p className="font-medium">{item.filename}</p>
          <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {ARTIFACTS.map(({ key, label, icon: Icon, hash }) => {
              const ready =
                key === "summary"
                  ? Boolean(item.artifacts.summary)
                  : Boolean(item.artifacts[key as keyof typeof item.artifacts]);
              const href = `/dashboard/sets/${setId}/documents/${item.document_id}#${hash}`;

              return (
                <div
                  key={key}
                  className={cn(
                    "flex items-center justify-between gap-2 rounded-lg border px-3 py-2.5",
                    ready ? "bg-background/80" : "bg-muted/20 opacity-60",
                  )}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <Icon className="h-4 w-4 shrink-0 text-primary" aria-hidden />
                    <span className="truncate text-sm">{label}</span>
                  </div>
                  {ready ? (
                    <Button type="button" variant="ghost" size="sm" className="h-7 shrink-0 px-2" asChild>
                      <Link href={href}>Open</Link>
                    </Button>
                  ) : (
                    <span className="text-[10px] text-muted-foreground">—</span>
                  )}
                </div>
              );
            })}
          </div>
          {item.errors.length > 0 ? (
            <p className="mt-2 text-xs text-destructive">{item.errors.join("; ")}</p>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
