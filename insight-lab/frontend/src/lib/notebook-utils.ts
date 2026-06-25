import type { DocumentSummary } from "@/lib/api";

const NOTEBOOK_ACCENTS = [
  "from-blue-500/15 to-blue-500/5 border-blue-200/60 dark:border-blue-800/40",
  "from-violet-500/15 to-violet-500/5 border-violet-200/60 dark:border-violet-800/40",
  "from-emerald-500/15 to-emerald-500/5 border-emerald-200/60 dark:border-emerald-800/40",
  "from-amber-500/15 to-amber-500/5 border-amber-200/60 dark:border-amber-800/40",
  "from-rose-500/15 to-rose-500/5 border-rose-200/60 dark:border-rose-800/40",
  "from-cyan-500/15 to-cyan-500/5 border-cyan-200/60 dark:border-cyan-800/40",
] as const;

export function notebookAccentClass(name: string): string {
  let hash = 0;
  for (let index = 0; index < name.length; index += 1) {
    hash = name.charCodeAt(index) + ((hash << 5) - hash);
  }
  return NOTEBOOK_ACCENTS[Math.abs(hash) % NOTEBOOK_ACCENTS.length];
}

export function documentHref(setId: string, doc: Pick<DocumentSummary, "id" | "file_type">): string {
  if (doc.file_type === "excel") {
    return `/dashboard/sets/${setId}/excel/${doc.id}`;
  }
  return `/dashboard/sets/${setId}/documents/${doc.id}`;
}

export type StudioTab = "brief" | "session" | "quiz" | "flashcards" | "guide" | "audio" | "infographic" | "concepts";

export type ExcelCanvasTab = "brief" | "preview" | "charts" | "builder" | "concepts" | "quiz";

export const STUDIO_TAB_LABELS: Record<StudioTab, string> = {
  brief: "Brief",
  session: "Study plan",
  quiz: "Quiz",
  flashcards: "Flashcards",
  guide: "Study guide",
  audio: "Audio",
  infographic: "Infographic",
  concepts: "Concept graph",
};

export const EXCEL_TAB_LABELS: Record<ExcelCanvasTab, string> = {
  brief: "Insights",
  preview: "Preview",
  charts: "Charts",
  builder: "Builder",
  concepts: "Concept graph",
  quiz: "Quiz",
};
