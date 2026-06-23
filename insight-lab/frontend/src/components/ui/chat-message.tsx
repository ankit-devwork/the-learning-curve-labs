"use client";

import { cn } from "@/lib/utils";
import type { SourceCitation } from "@/lib/api";
import { CitationChips } from "@/components/ui/citation-chips";

type ChatMessageProps = {
  role: "user" | "assistant";
  question?: string;
  answer?: string;
  sources?: SourceCitation[];
  footer?: string | null;
  onCitationClick?: (source: SourceCitation, index: number) => void;
};

export function ChatMessageBubble({
  role,
  question,
  answer,
  sources,
  footer,
  onCitationClick,
}: ChatMessageProps) {
  const isUser = role === "user";
  const content = isUser ? question : answer;

  if (!content) {
    return null;
  }

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[92%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm sm:max-w-[85%]",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-card ring-1 ring-border/60",
        )}
      >
        {!isUser ? (
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            InsightLab
          </p>
        ) : null}
        <p className="whitespace-pre-wrap">{content}</p>
        {!isUser && sources && sources.length > 0 ? (
          <CitationChips sources={sources} onCitationClick={onCitationClick} />
        ) : null}
        {!isUser && footer ? (
          <p className="mt-2 text-[11px] text-muted-foreground">{footer}</p>
        ) : null}
        {!isUser ? (
          <p className="mt-2 text-[10px] text-muted-foreground/80">Grounded in your sources</p>
        ) : null}
      </div>
    </div>
  );
}
