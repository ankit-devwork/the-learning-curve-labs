"use client";

import { useCallback, useEffect, useState } from "react";
import {
  apiFetch,
  type ConceptMasteryItem,
  type GenerateQuizRequest,
  type QuizResponse,
  type QuizSubmitResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureGuide } from "@/components/ui/feature-guide";
import { Label } from "@/components/ui/label";
import { QuizMasteryProgress, hasWeakConcepts } from "@/components/documents/quiz-mastery-progress";
import { cn } from "@/lib/utils";

const selectClassName = cn(
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
  "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  "disabled:cursor-not-allowed disabled:opacity-50",
);

type SetQuizPanelProps = {
  setId: string;
  accessToken: string | null;
  hasReadyDocuments: boolean;
};

export function SetQuizPanel({ setId, accessToken, hasReadyDocuments }: SetQuizPanelProps) {
  const [quiz, setQuiz] = useState<QuizResponse | null>(null);
  const [questionType, setQuestionType] = useState<GenerateQuizRequest["question_type"]>("scq");
  const [difficulty, setDifficulty] = useState<GenerateQuizRequest["difficulty"]>("medium");
  const [numQuestions, setNumQuestions] = useState(5);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [results, setResults] = useState<QuizSubmitResponse | null>(null);
  const [mastery, setMastery] = useState<ConceptMasteryItem[]>([]);
  const [generating, setGenerating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMastery = useCallback(async () => {
    if (!accessToken || !hasReadyDocuments) {
      return;
    }
    const response = await apiFetch(`/workspaces/${setId}/concepts/mastery`, accessToken);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    setMastery(data.concepts ?? []);
  }, [accessToken, hasReadyDocuments, setId]);

  useEffect(() => {
    void loadMastery();
  }, [loadMastery]);

  async function generateAdaptiveQuiz() {
    if (!accessToken) {
      return;
    }
    setGenerating(true);
    setError(null);
    setResults(null);
    setAnswers({});
    try {
      const response = await apiFetch(
        `/workspaces/${setId}/quiz/adaptive/generate`,
        accessToken,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question_type: questionType,
            difficulty,
            num_questions: numQuestions,
          }),
        },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || body.detail || "Failed to generate set quiz");
        return;
      }
      setQuiz((await response.json()) as QuizResponse);
    } finally {
      setGenerating(false);
    }
  }

  async function submitQuiz() {
    if (!accessToken || !quiz) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await apiFetch(`/quizzes/${quiz.quiz_id}/submit`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || "Submit failed");
        return;
      }
      setResults((await response.json()) as QuizSubmitResponse);
      await loadMastery();
    } finally {
      setSubmitting(false);
    }
  }

  const weakAvailable = hasWeakConcepts(mastery);

  return (
    <Card className="shadow-sm" data-tour="set-quiz">
      <CardHeader>
        <CardTitle>Set-wide adaptive quiz</CardTitle>
        <CardDescription>
          Practice weak topics across all documents in this study set after completing at least one
          document quiz.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FeatureGuide
          title="How it works"
          steps={[
            "Take a quiz on any document in this set first.",
            "We track topic mastery per document.",
            "This quiz pulls questions from weak concepts across the whole set.",
          ]}
        />

        {mastery.length > 0 ? (
          <QuizMasteryProgress concepts={mastery} title="Topic progress across this set" />
        ) : null}

        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="set-quiz-type">Question type</Label>
            <select
              id="set-quiz-type"
              className={selectClassName}
              value={questionType}
              onChange={(event) =>
                setQuestionType(event.target.value as GenerateQuizRequest["question_type"])
              }
            >
              <option value="scq">Single choice</option>
              <option value="mcq">Multiple choice</option>
              <option value="true_false">True / false</option>
            </select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="set-quiz-difficulty">Difficulty</Label>
            <select
              id="set-quiz-difficulty"
              className={selectClassName}
              value={difficulty}
              onChange={(event) =>
                setDifficulty(event.target.value as GenerateQuizRequest["difficulty"])
              }
            >
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="set-quiz-count">Questions</Label>
            <select
              id="set-quiz-count"
              className={selectClassName}
              value={numQuestions}
              onChange={(event) => setNumQuestions(Number(event.target.value))}
            >
              {[3, 5, 8, 10].map((count) => (
                <option key={count} value={count}>
                  {count}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Button
          type="button"
          disabled={!hasReadyDocuments || generating || !weakAvailable}
          onClick={() => void generateAdaptiveQuiz()}
        >
          {generating ? "Generating…" : "Generate set-wide adaptive quiz"}
        </Button>

        {!weakAvailable && hasReadyDocuments ? (
          <p className="text-sm text-muted-foreground">
            Complete a quiz on at least one document in this set to unlock adaptive practice.
          </p>
        ) : null}

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {quiz ? (
          <div className="space-y-4 rounded-xl border p-4">
            <h3 className="font-medium">{quiz.title}</h3>
            {quiz.target_concepts && quiz.target_concepts.length > 0 ? (
              <p className="text-xs text-muted-foreground">
                Targeting: {quiz.target_concepts.map((item) => item.name).join(", ")}
              </p>
            ) : null}
            {quiz.questions.map((question) => (
              <div key={question.id} className="space-y-2">
                <p className="text-sm font-medium">{question.question_text}</p>
                <div className="space-y-1">
                  {question.options.map((option, index) => (
                    <label key={`${question.id}-${index}`} className="flex items-center gap-2 text-sm">
                      <input
                        type="radio"
                        name={question.id}
                        checked={answers[question.id] === index}
                        disabled={Boolean(results)}
                        onChange={() =>
                          setAnswers((current) => ({ ...current, [question.id]: index }))
                        }
                      />
                      {option}
                    </label>
                  ))}
                </div>
              </div>
            ))}
            {!results ? (
              <Button
                type="button"
                disabled={submitting || quiz.questions.some((q) => answers[q.id] === undefined)}
                onClick={() => void submitQuiz()}
              >
                {submitting ? "Submitting…" : "Submit answers"}
              </Button>
            ) : (
              <div className="rounded-md bg-muted/40 p-3 text-sm">
                <p className="font-medium">
                  Score: {results.score}/{results.total} ({results.percent}%)
                </p>
              </div>
            )}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
