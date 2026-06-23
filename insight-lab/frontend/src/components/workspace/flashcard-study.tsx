"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { FlashcardItem } from "@/lib/api";
import { cn } from "@/lib/utils";

type FlashcardStudyProps = {
  title: string;
  cards: FlashcardItem[];
  onReview: (flashcardId: string, knew: boolean) => Promise<void>;
  onViewSource?: (card: FlashcardItem) => void;
  className?: string;
};

export function FlashcardStudy({
  title,
  cards,
  onReview,
  onViewSource,
  className,
}: FlashcardStudyProps) {
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const card = cards[index];
  const progressLabel = useMemo(
    () => `${Math.min(index + 1, cards.length)} / ${cards.length}`,
    [index, cards.length],
  );

  if (cards.length === 0) {
    return null;
  }

  async function handleReview(knew: boolean) {
    if (!card || submitting) {
      return;
    }
    setSubmitting(true);
    try {
      await onReview(card.id, knew);
      setFlipped(false);
      setIndex((current) => Math.min(current + 1, cards.length - 1));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card className={cn("shadow-sm", className)}>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>Tap the card to flip · mark whether you knew it</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Card {progressLabel}</span>
          {card.source_chunk_index != null && onViewSource ? (
            <button
              type="button"
              className="text-primary hover:underline"
              onClick={() => onViewSource(card)}
            >
              View source
            </button>
          ) : null}
        </div>

        <button
          type="button"
          className="min-h-40 w-full rounded-xl border bg-gradient-to-br from-muted/40 to-background p-6 text-left shadow-sm transition-transform hover:scale-[1.01]"
          onClick={() => setFlipped((current) => !current)}
        >
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {flipped ? "Back" : "Front"}
          </p>
          <p className="mt-3 text-base leading-relaxed">{flipped ? card.back : card.front}</p>
        </button>

        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            className="flex-1"
            disabled={submitting || index >= cards.length}
            onClick={() => void handleReview(false)}
          >
            Still learning
          </Button>
          <Button
            type="button"
            className="flex-1"
            disabled={submitting || index >= cards.length}
            onClick={() => void handleReview(true)}
          >
            Got it
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
