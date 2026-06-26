"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { ExplainResponse, FlashcardItem } from "@/lib/api";
import { downloadAnkiCsv } from "@/lib/export-utils";
import { cn } from "@/lib/utils";

type FlashcardStudyProps = {
  title: string;
  cards: FlashcardItem[];
  setId?: string;
  dueIds?: string[];
  dueCount?: number;
  accessToken?: string | null;
  onReview: (flashcardId: string, knew: boolean) => Promise<void>;
  onExplain?: (flashcardId: string) => Promise<ExplainResponse | null>;
  onViewSource?: (card: FlashcardItem) => void;
  className?: string;
};

export function FlashcardStudy({
  title,
  cards,
  setId,
  dueIds = [],
  dueCount = 0,
  onReview,
  onExplain,
  onViewSource,
  className,
}: FlashcardStudyProps) {
  const [reviewDueOnly, setReviewDueOnly] = useState(false);
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [explainText, setExplainText] = useState<string | null>(null);

  const dueSet = useMemo(() => new Set(dueIds), [dueIds]);
  const deck = useMemo(() => {
    if (!reviewDueOnly || dueSet.size === 0) {
      return cards;
    }
    const filtered = cards.filter((card) => dueSet.has(card.id));
    return filtered.length > 0 ? filtered : cards;
  }, [cards, dueSet, reviewDueOnly]);

  const card = deck[index];
  const progressLabel = useMemo(
    () => `${Math.min(index + 1, deck.length)} / ${deck.length}`,
    [index, deck.length],
  );

  if (cards.length === 0) {
    return null;
  }

  async function handleReview(knew: boolean) {
    if (!card || submitting) {
      return;
    }
    setSubmitting(true);
    setExplainText(null);
    try {
      await onReview(card.id, knew);
      setFlipped(false);
      setIndex((current) => Math.min(current + 1, Math.max(deck.length - 1, 0)));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleExplain() {
    if (!card || !onExplain || explaining) {
      return;
    }
    setExplaining(true);
    try {
      const result = await onExplain(card.id);
      setExplainText(result?.explanation ?? null);
    } finally {
      setExplaining(false);
    }
  }

  return (
    <Card className={cn("shadow-sm", className)}>
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription>
            Tap to flip · mark whether you knew it
            {dueCount > 0 ? ` · ${dueCount} due today` : ""}
          </CardDescription>
        </div>
        <div className="flex flex-wrap gap-2">
          {dueCount > 0 ? (
            <Button
              type="button"
              variant={reviewDueOnly ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setReviewDueOnly((current) => !current);
                setIndex(0);
                setFlipped(false);
                setExplainText(null);
              }}
            >
              {reviewDueOnly ? "Due only" : "Review due today"}
            </Button>
          ) : null}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => downloadAnkiCsv(cards, `${setId ?? "flashcards"}-anki.csv`)}
          >
            Anki CSV
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Card {progressLabel}</span>
          {card?.source_chunk_index != null && onViewSource ? (
            <button
              type="button"
              className="text-primary hover:underline"
              onClick={() => onViewSource(card)}
            >
              View source
            </button>
          ) : null}
        </div>

        {card ? (
          <>
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

            {onExplain ? (
              <Button type="button" variant="outline" size="sm" disabled={explaining} onClick={() => void handleExplain()}>
                {explaining ? "Explaining…" : "Explain with citation"}
              </Button>
            ) : null}

            {explainText ? (
              <div className="rounded-md border bg-background/80 p-3 text-sm text-muted-foreground whitespace-pre-wrap">
                {explainText}
              </div>
            ) : null}

            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                className="flex-1"
                disabled={submitting}
                onClick={() => void handleReview(false)}
              >
                Still learning
              </Button>
              <Button type="button" className="flex-1" disabled={submitting} onClick={() => void handleReview(true)}>
                Got it
              </Button>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No cards due — great job!</p>
        )}
      </CardContent>
    </Card>
  );
}
