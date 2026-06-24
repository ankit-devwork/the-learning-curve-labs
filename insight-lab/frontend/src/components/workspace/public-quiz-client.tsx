"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { PublicQuizResponse, QuizSubmitResponse } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api-backend";

export function PublicQuizClient({ shareToken }: { shareToken: string }) {
  const [quiz, setQuiz] = useState<PublicQuizResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [results, setResults] = useState<QuizSubmitResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const response = await fetch(`${API_BASE}/public/quizzes/${shareToken}`);
      setLoading(false);
      if (!response.ok) {
        setError("This quiz link is invalid or no longer published.");
        return;
      }
      setQuiz((await response.json()) as PublicQuizResponse);
    }
    void load();
  }, [shareToken]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!quiz) {
      return;
    }
    setSubmitting(true);
    const response = await fetch(`${API_BASE}/public/quizzes/${shareToken}/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName.trim() || "Guest", answers }),
    });
    setSubmitting(false);
    if (!response.ok) {
      setError("Could not submit quiz.");
      return;
    }
    setResults((await response.json()) as QuizSubmitResponse);
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading quiz…</p>;
  }

  if (error || !quiz) {
    return <p className="text-sm text-destructive">{error || "Quiz unavailable"}</p>;
  }

  if (results) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>
            {results.score}/{results.total} correct ({results.percent}%)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {(results.results ?? []).map((item) => (
            <div key={item.question_id} className="rounded-md border px-3 py-2">
              <p className="font-medium">{item.question_text}</p>
              <p className={item.correct ? "text-emerald-600" : "text-destructive"}>
                {item.correct ? "Correct" : "Incorrect"}
              </p>
              {item.explanation ? <p className="mt-1 text-muted-foreground">{item.explanation}</p> : null}
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{quiz.title}</CardTitle>
        {quiz.source_filename ? (
          <CardDescription>From {quiz.source_filename}</CardDescription>
        ) : null}
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-5">
          <Input
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
            placeholder="Your name (optional)"
          />
          {quiz.questions.map((question) => (
            <fieldset key={question.id} className="space-y-2 rounded-md border px-3 py-3">
              <legend className="px-1 text-sm font-medium">{question.question_text}</legend>
              {question.options.map((option, index) => (
                <label key={option} className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name={question.id}
                    checked={answers[question.id] === index}
                    onChange={() => setAnswers((prev) => ({ ...prev, [question.id]: index }))}
                  />
                  {option}
                </label>
              ))}
            </fieldset>
          ))}
          <Button type="submit" disabled={submitting}>
            {submitting ? "Submitting…" : "Submit quiz"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
