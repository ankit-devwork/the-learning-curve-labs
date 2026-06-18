"use client";

import { cn } from "@/lib/utils";
import type { ConceptMasteryItem } from "@/lib/api";

const WEAK_THRESHOLD = 60;
const STRONG_THRESHOLD = 80;

function masteryStatus(item: ConceptMasteryItem): {
  label: string;
  tone: "muted" | "success" | "warning" | "danger";
} {
  if (item.attempts === 0 || item.percent == null) {
    return { label: "Not tried yet", tone: "muted" };
  }
  if (item.percent >= STRONG_THRESHOLD) {
    return { label: "Got it", tone: "success" };
  }
  if (item.percent >= WEAK_THRESHOLD) {
    return { label: "Getting there", tone: "warning" };
  }
  return { label: "Needs practice", tone: "danger" };
}

const toneClasses = {
  muted: "bg-muted text-muted-foreground",
  success: "bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-200",
  warning: "bg-amber-100 text-amber-900 dark:bg-amber-950/40 dark:text-amber-100",
  danger: "bg-red-100 text-red-900 dark:bg-red-950/40 dark:text-red-100",
};

type QuizMasteryProgressProps = {
  concepts: ConceptMasteryItem[];
  migrationRequired?: boolean;
  notice?: string;
  className?: string;
};

export function QuizMasteryProgress({
  concepts,
  migrationRequired,
  notice,
  className,
}: QuizMasteryProgressProps) {
  if (migrationRequired && notice) {
    return (
      <p className={cn("rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900", className)}>
        Progress tracking is unavailable until the latest database migration is applied.
      </p>
    );
  }

  if (concepts.length === 0) {
    return (
      <p className={cn("text-sm text-muted-foreground", className)}>
        Topic progress will appear here after you complete a quiz.
      </p>
    );
  }

  const practiced = concepts.filter((item) => item.attempts > 0);
  const needsPractice = concepts.filter(
    (item) => item.attempts > 0 && item.percent != null && item.percent < WEAK_THRESHOLD,
  );

  return (
    <div className={cn("space-y-3", className)}>
      <div className="space-y-1">
        <p className="text-sm font-medium">Your progress by topic</p>
        <p className="text-xs text-muted-foreground">
          {practiced.length} of {concepts.length} topics attempted
          {needsPractice.length > 0
            ? ` · ${needsPractice.length} need${needsPractice.length === 1 ? "s" : ""} more practice`
            : practiced.length > 0
              ? " · nice work"
              : ""}
        </p>
      </div>
      <ul className="space-y-2">
        {concepts.map((item) => {
          const status = masteryStatus(item);
          return (
            <li
              key={item.concept_id}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md border px-3 py-2 text-sm"
            >
              <div className="min-w-0">
                <p className="font-medium">{item.name}</p>
                {item.topic && <p className="text-xs text-muted-foreground">{item.topic}</p>}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {item.percent != null && (
                  <span className="text-xs text-muted-foreground">{Math.round(item.percent)}%</span>
                )}
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 text-xs font-medium",
                    toneClasses[status.tone],
                  )}
                >
                  {status.label}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function hasWeakConcepts(concepts: ConceptMasteryItem[]): boolean {
  return concepts.some(
    (item) => item.attempts > 0 && item.percent != null && item.percent < WEAK_THRESHOLD,
  );
}
