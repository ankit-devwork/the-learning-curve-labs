"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

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
  if (questions.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Suggested questions
      </p>
      <div className="flex flex-wrap gap-2">
        {questions.map((question) => (
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
