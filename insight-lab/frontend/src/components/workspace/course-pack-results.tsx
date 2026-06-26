"use client";

import Link from "next/link";
import {
  BookOpen,
  Brain,
  FileText,
  GraduationCap,
  Image,
  Layers,
  Presentation,
  Volume2,
} from "lucide-react";
import type { CoursePackDocumentResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type CoursePackResultsProps = {
  setId: string;
  documents: CoursePackDocumentResult[];
};

const DOCUMENT_ARTIFACTS = [
  { key: "summary", label: "Brief", icon: FileText, hash: "brief" },
  { key: "quiz_id", label: "Quiz", icon: Brain, hash: "quiz" },
  { key: "flashcard_set_id", label: "Flashcards", icon: Layers, hash: "flashcards" },
  { key: "study_guide_id", label: "Study guide", icon: BookOpen, hash: "guide" },
  { key: "audio_script", label: "Audio", icon: Volume2, hash: "audio" },
  { key: "infographic_id", label: "Infographic", icon: Image, hash: "infographic" },
  { key: "slide_deck_id", label: "Slides", icon: Presentation, hash: "slides" },
  { key: "homework_solution_id", label: "Homework sample", icon: GraduationCap, hash: "homework" },
] as const;

function artifactReady(
  item: CoursePackDocumentResult,
  key: (typeof DOCUMENT_ARTIFACTS)[number]["key"],
): boolean {
  if (key === "summary") {
    return Boolean(item.artifacts.summary);
  }
  return Boolean(item.artifacts[key as keyof typeof item.artifacts]);
}

export function CoursePackResults({ setId, documents }: CoursePackResultsProps) {
  return (
    <ul className="space-y-4">
      {documents.map((item) => {
        const isExcel = item.file_type === "excel";
        const basePath = isExcel
          ? `/dashboard/sets/${setId}/excel/${item.document_id}`
          : `/dashboard/sets/${setId}/documents/${item.document_id}`;

        return (
          <li key={item.document_id} className="notebook-surface rounded-xl p-4">
            <p className="font-medium">
              {item.filename}
              {isExcel ? <span className="ml-2 text-xs font-normal text-muted-foreground">spreadsheet</span> : null}
            </p>
            {isExcel ? (
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                <div
                  className={cn(
                    "flex items-center justify-between gap-2 rounded-lg border px-3 py-2.5",
                    item.artifacts.summary ? "bg-background/80" : "bg-muted/20 opacity-60",
                  )}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <FileText className="h-4 w-4 shrink-0 text-primary" aria-hidden />
                    <span className="truncate text-sm">Analysis summary</span>
                  </div>
                  {item.artifacts.summary ? (
                    <Button type="button" variant="ghost" size="sm" className="h-7 shrink-0 px-2" asChild>
                      <Link href={`${basePath}#brief`}>Open</Link>
                    </Button>
                  ) : (
                    <span className="text-[10px] text-muted-foreground">—</span>
                  )}
                </div>
                <div
                  className={cn(
                    "flex items-center justify-between gap-2 rounded-lg border px-3 py-2.5",
                    item.artifacts.quiz_id ? "bg-background/80" : "bg-muted/20 opacity-60",
                  )}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <Brain className="h-4 w-4 shrink-0 text-primary" aria-hidden />
                    <span className="truncate text-sm">Quiz</span>
                  </div>
                  {item.artifacts.quiz_id ? (
                    <Button type="button" variant="ghost" size="sm" className="h-7 shrink-0 px-2" asChild>
                      <Link href={`${basePath}#quiz`}>Open</Link>
                    </Button>
                  ) : (
                    <span className="text-[10px] text-muted-foreground">—</span>
                  )}
                </div>
              </div>
            ) : (
              <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {DOCUMENT_ARTIFACTS.map(({ key, label, icon: Icon, hash }) => {
                  const ready = artifactReady(item, key);
                  const href = `${basePath}#${hash}`;

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
            )}
            {item.artifacts.homework_question ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Homework sample: {item.artifacts.homework_question}
              </p>
            ) : null}
            {item.errors.length > 0 ? (
              <p className="mt-2 text-xs text-destructive">{item.errors.join("; ")}</p>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
