"use client";

import { useCallback, useState } from "react";
import {
  apiFetch,
  type GenerateQuizRequest,
  type QuizResponse,
  type QuizSubmitResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

const selectClassName = cn(
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
  "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  "disabled:cursor-not-allowed disabled:opacity-50",
);

type DocumentQuizPanelProps = {
  documentId: string;
  ready: boolean;
  accessToken: string | null;
  initialQuiz?: QuizResponse | null;
};

export function DocumentQuizPanel({
  documentId,
  ready,
  accessToken,
  initialQuiz = null,
}: DocumentQuizPanelProps) {
  const [quiz, setQuiz] = useState<QuizResponse | null>(initialQuiz);
  const [questionType, setQuestionType] = useState<GenerateQuizRequest["question_type"]>("scq");
  const [difficulty, setDifficulty] = useState<GenerateQuizRequest["difficulty"]>("medium");
  const [numQuestions, setNumQuestions] = useState(5);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [results, setResults] = useState<QuizSubmitResponse | null>(null);
  const [generating, setGenerating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateQuiz = useCallback(async () => {
    if (!accessToken) {
      return;
    }
    setGenerating(true);
    setError(null);
    setResults(null);
    setAnswers({});

    const response = await apiFetch(`/documents/${documentId}/quiz/generate`, accessToken, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question_type: questionType,
        difficulty,
        num_questions: numQuestions,
      }),
    });
    setGenerating(false);

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || body.detail || `Failed to generate quiz (${response.status})`);
      return;
    }

    const data = (await response.json()) as QuizResponse;
    setQuiz(data);
  }, [accessToken, documentId, questionType, difficulty, numQuestions]);

  async function handleSubmit() {
    if (!accessToken || !quiz) {
      return;
    }
    if (quiz.questions.some((question) => answers[question.id] === undefined)) {
      setError("Answer every question before submitting.");
      return;
    }

    setSubmitting(true);
    setError(null);
    const response = await apiFetch(`/quizzes/${quiz.quiz_id}/submit`, accessToken, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    });
    setSubmitting(false);

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || body.detail || `Submit failed (${response.status})`);
      return;
    }

    setResults((await response.json()) as QuizSubmitResponse);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Quiz</CardTitle>
        <CardDescription>
          Generate single-choice questions from this document&apos;s embedded chunks.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="quiz-type">Question type</Label>
            <select
              id="quiz-type"
              className={selectClassName}
              value={questionType}
              disabled={!ready || generating}
              onChange={(event) =>
                setQuestionType(event.target.value as GenerateQuizRequest["question_type"])
              }
            >
              <option value="scq">Single choice</option>
              <option value="mcq">Multiple choice</option>
              <option value="true_false">True / False</option>
            </select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="quiz-difficulty">Difficulty</Label>
            <select
              id="quiz-difficulty"
              className={selectClassName}
              value={difficulty}
              disabled={!ready || generating}
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
            <Label htmlFor="quiz-count">Questions</Label>
            <select
              id="quiz-count"
              className={selectClassName}
              value={numQuestions}
              disabled={!ready || generating}
              onChange={(event) => setNumQuestions(Number.parseInt(event.target.value, 10))}
            >
              {[3, 5, 7, 10].map((count) => (
                <option key={count} value={count}>
                  {count}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Button type="button" disabled={!ready || generating} onClick={() => void generateQuiz()}>
          {generating ? "Generating..." : quiz ? "Regenerate quiz" : "Generate quiz"}
        </Button>

        <Button
          type="button"
          variant="secondary"
          disabled={!ready || generating}
          onClick={async () => {
            if (!accessToken) {
              return;
            }
            setGenerating(true);
            setError(null);
            setResults(null);
            setAnswers({});
            const response = await apiFetch(
              `/documents/${documentId}/quiz/adaptive/generate`,
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
            setGenerating(false);
            if (!response.ok) {
              const body = await response.json().catch(() => ({}));
              setError(body.error || body.detail || `Adaptive quiz failed (${response.status})`);
              return;
            }
            setQuiz((await response.json()) as QuizResponse);
          }}
        >
          {generating ? "Generating..." : "Adaptive quiz (weak concepts)"}
        </Button>

        {error && <p className="text-sm text-destructive">{error}</p>}

        {quiz && !results && (
          <div className="space-y-5">
            <div>
              <p className="font-medium">{quiz.title}</p>
              <p className="text-sm text-muted-foreground">
                {quiz.question_type} · {quiz.difficulty}
                {quiz.cached ? " · cached" : ""}
              </p>
              {quiz.target_concepts && quiz.target_concepts.length > 0 && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Targeting: {quiz.target_concepts.map((c) => c.name).join(", ")}
                </p>
              )}
            </div>
            {quiz.questions.map((question, index) => (
              <div key={question.id} className="space-y-2 rounded-md border p-4">
                <p className="text-sm font-medium">
                  {index + 1}. {question.question_text}
                </p>
                <div className="space-y-2">
                  {question.options.map((option, optionIndex) => (
                    <label
                      key={`${question.id}-${optionIndex}`}
                      className="flex cursor-pointer items-center gap-2 text-sm"
                    >
                      <input
                        type="radio"
                        name={question.id}
                        checked={answers[question.id] === optionIndex}
                        onChange={() =>
                          setAnswers((current) => ({ ...current, [question.id]: optionIndex }))
                        }
                      />
                      <span>{option}</span>
                    </label>
                  ))}
                </div>
              </div>
            ))}
            <Button type="button" disabled={submitting} onClick={() => void handleSubmit()}>
              {submitting ? "Scoring..." : "Submit answers"}
            </Button>
          </div>
        )}

        {results && (
          <div className="space-y-4">
            <div className="rounded-md border bg-muted/30 p-4">
              <p className="text-lg font-semibold">
                Score: {results.score}/{results.total} ({results.percent}%)
              </p>
            </div>
            {results.results.map((result) => (
              <div
                key={result.question_id}
                className={cn(
                  "rounded-md border p-4 text-sm",
                  result.correct ? "border-green-300 bg-green-50/50" : "border-red-300 bg-red-50/50",
                )}
              >
                <p className="font-medium">{result.question_text}</p>
                <p className="mt-2 text-muted-foreground">
                  Your answer: {quiz?.questions.find((q) => q.id === result.question_id)?.options[result.selected_option_index]}
                </p>
                {!result.correct && (
                  <p className="mt-1 text-muted-foreground">
                    Correct:{" "}
                    {quiz?.questions.find((q) => q.id === result.question_id)?.options[result.correct_option_index]}
                  </p>
                )}
                {result.explanation && (
                  <p className="mt-2 text-muted-foreground">{result.explanation}</p>
                )}
              </div>
            ))}
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setResults(null);
                setAnswers({});
              }}
            >
              Try again
            </Button>
          </div>
        )}

        {!ready && (
          <p className="text-sm text-muted-foreground">Process the document first to generate a quiz.</p>
        )}
      </CardContent>
    </Card>
  );
}
