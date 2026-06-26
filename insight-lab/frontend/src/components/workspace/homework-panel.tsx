"use client";

import { useState } from "react";
import { apiFetch, type HomeworkSolutionResponse, type SourceCitation } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatMessageBubble } from "@/components/ui/chat-message";

type HomeworkPanelProps = {
  documentId: string;
  accessToken: string | null;
  ready: boolean;
  onCitationClick?: (source: SourceCitation) => void;
};

export function HomeworkPanel({
  documentId,
  accessToken,
  ready,
  onCitationClick,
}: HomeworkPanelProps) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<HomeworkSolutionResponse | null>(null);

  async function handleSolve(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || !accessToken) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await apiFetch(`/documents/${documentId}/homework/solve`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || "Could not solve this question.");
        return;
      }
      setResult((await response.json()) as HomeworkSolutionResponse);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="shadow-sm" data-tour="homework-panel">
      <CardHeader>
        <CardTitle className="text-lg">Homework help</CardTitle>
        <CardDescription>
          Step-by-step guidance grounded in this document when possible. Verify critical steps with your instructor.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSolve} className="space-y-3">
          <textarea
            className="flex min-h-[120px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Paste a homework question or problem from your assignment…"
            rows={4}
            disabled={!ready || loading}
          />
          <Button type="submit" disabled={!ready || loading || !question.trim()}>
            {loading ? "Working…" : "Get step-by-step help"}
          </Button>
        </form>

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {result ? (
          <div className="space-y-4 rounded-xl border bg-muted/20 p-4">
            {result.summary ? (
              <p className="text-sm font-medium">{result.summary}</p>
            ) : null}
            <ol className="space-y-3">
              {result.steps.map((step, index) => (
                <li key={`${step.title}-${index}`} className="text-sm">
                  <p className="font-medium">
                    {index + 1}. {step.title}
                  </p>
                  <p className="mt-1 text-muted-foreground">{step.detail}</p>
                </li>
              ))}
            </ol>
            {result.disclaimer ? (
              <p className="text-xs text-muted-foreground">{result.disclaimer}</p>
            ) : null}
            {result.sources && result.sources.length > 0 ? (
              <ChatMessageBubble
                role="assistant"
                answer="Sources from your document:"
                sources={result.sources}
                onCitationClick={onCitationClick}
              />
            ) : null}
          </div>
        ) : null}

        {!ready ? (
          <p className="text-sm text-muted-foreground">Process the document first to use homework help.</p>
        ) : null}
      </CardContent>
    </Card>
  );
}
