"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const VISIBLE_COUNT = 2;

type SuggestedQuestionsProps = {
  questions: string[];
  disabled?: boolean;
  onSelect: (question: string) => void;
  className?: string;
};

export function SuggestedQuestions({
  questions,
  disabled,
  onSelect,
  className,
}: SuggestedQuestionsProps) {
  const [expanded, setExpanded] = useState(false);

  if (questions.length === 0) {
    return null;
  }

  const visible = expanded ? questions : questions.slice(0, VISIBLE_COUNT);
  const hiddenCount = questions.length - VISIBLE_COUNT;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Try asking
        </p>
        {!expanded && hiddenCount > 0 ? (
          <button
            type="button"
            className="text-xs font-medium text-primary hover:underline"
            onClick={() => setExpanded(true)}
          >
            Show {hiddenCount} more
          </button>
        ) : expanded && questions.length > VISIBLE_COUNT ? (
          <button
            type="button"
            className="text-xs font-medium text-muted-foreground hover:text-foreground"
            onClick={() => setExpanded(false)}
          >
            Show less
          </button>
        ) : null}
      </div>
      <div className="flex flex-wrap gap-2">
        {visible.map((question) => (
          <Button
            key={question}
            type="button"
            size="sm"
            variant="outline"
            disabled={disabled}
            className="h-auto whitespace-normal px-3 py-2 text-left text-xs"
            onClick={() => onSelect(question)}
          >
            {question}
          </Button>
        ))}
      </div>
    </div>
  );
}
