"use client";

import { useCallback, useEffect, useState } from "react";
import {
  apiFetch,
  type ConceptMasteryItem,
  type GenerateQuizRequest,
  type QuizResponse,
  type QuizSubmitResponse,
  type QuizQuestionEditable,
  type StudySessionRecord,
} from "@/lib/api";
import { resolveActiveQuizStepId } from "@/lib/study-session-utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureGuide } from "@/components/ui/feature-guide";
import { Label } from "@/components/ui/label";
import { QuizMasteryProgress, hasWeakConcepts } from "@/components/documents/quiz-mastery-progress";
import { cn } from "@/lib/utils";
import { downloadAuthenticatedText } from "@/lib/export-utils";
import { useToast } from "@/components/ui/toast";

const selectClassName = cn(
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
  "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  "disabled:cursor-not-allowed disabled:opacity-50",
);

type SetQuizPanelProps = {
  setId: string;
  accessToken: string | null;
  hasReadyDocuments: boolean;
  canEdit?: boolean;
  embedded?: boolean;
};

export function SetQuizPanel({
  setId,
  accessToken,
  hasReadyDocuments,
  canEdit = true,
  embedded = false,
}: SetQuizPanelProps) {
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
  const [editMode, setEditMode] = useState(false);
  const [editQuestions, setEditQuestions] = useState<QuizQuestionEditable[]>([]);
  const [savingQuestionId, setSavingQuestionId] = useState<string | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [exportingQti, setExportingQti] = useState(false);
  const { toast } = useToast();

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
    if (!accessToken || !canEdit) {
      return;
    }
    setGenerating(true);
    setError(null);
    setResults(null);
    setAnswers({});
    setEditMode(false);
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

  async function fetchActiveSession(): Promise<StudySessionRecord | null> {
    if (!accessToken || !hasReadyDocuments) {
      return null;
    }
    const response = await apiFetch(`/workspaces/${setId}/study-session/active`, accessToken);
    if (!response.ok) {
      return null;
    }
    const data = await response.json();
    return (data.session as StudySessionRecord | null) ?? null;
  }

  async function submitQuiz() {
    if (!accessToken || !quiz) {
      return;
    }
    if (quiz.questions.some((question) => answers[question.id] === undefined)) {
      setError("Answer every question before submitting.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const activeSession = await fetchActiveSession();
      const studySessionStepId = resolveActiveQuizStepId(activeSession);
      const response = await apiFetch(`/quizzes/${quiz.quiz_id}/submit`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answers,
          study_session_step_id: studySessionStepId ?? undefined,
        }),
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

  async function loadEditMode() {
    if (!accessToken || !quiz || !canEdit) {
      return;
    }
    const response = await apiFetch(`/quizzes/${quiz.quiz_id}/edit`, accessToken);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Could not load quiz for editing");
      return;
    }
    const data = (await response.json()) as QuizResponse & { questions: QuizQuestionEditable[] };
    setEditQuestions(data.questions);
    setEditMode(true);
  }

  async function saveQuestion(question: QuizQuestionEditable) {
    if (!accessToken || !quiz || !canEdit) {
      return;
    }
    setSavingQuestionId(question.id);
    const response = await apiFetch(
      `/quizzes/${quiz.quiz_id}/questions/${question.id}`,
      accessToken,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_text: question.question_text,
          options: question.options,
          correct_option_index: question.correct_option_index,
          explanation: question.explanation,
        }),
      },
    );
    setSavingQuestionId(null);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Save failed");
    }
  }

  async function handlePublish() {
    if (!accessToken || !quiz || !canEdit) {
      return;
    }
    setPublishing(true);
    const response = await apiFetch(`/quizzes/${quiz.quiz_id}/publish`, accessToken, {
      method: "POST",
    });
    setPublishing(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Publish failed");
      return;
    }
    const data = (await response.json()) as QuizResponse;
    setQuiz(data);
    setEditMode(false);
  }

  async function copyPublicLink() {
    if (!quiz?.public_share_token) {
      return;
    }
    const url = `${window.location.origin}/quiz/${quiz.public_share_token}`;
    try {
      await navigator.clipboard.writeText(url);
      toast({ title: "Public link copied", description: "Anyone with the link can take this quiz.", variant: "success" });
    } catch {
      toast({ title: "Could not copy link", variant: "error" });
    }
  }

  async function exportQti() {
    if (!accessToken || !quiz) {
      return;
    }
    setExportingQti(true);
    try {
      await downloadAuthenticatedText(
        `/quizzes/${quiz.quiz_id}/export/qti`,
        accessToken,
        `${quiz.title.replace(/\s+/g, "-").toLowerCase()}-qti.xml`,
      );
      toast({ title: "QTI export downloaded", variant: "success" });
    } catch {
      toast({ title: "QTI export failed", variant: "error" });
    } finally {
      setExportingQti(false);
    }
  }

  const weakAvailable = hasWeakConcepts(mastery);

  const content = (
    <div className="space-y-4" data-tour={embedded ? "set-quiz" : undefined}>
      {!embedded ? (
        <FeatureGuide
          title="How it works"
          steps={[
            "Take a quiz on any file in this sheet first.",
            "We track topic mastery per file.",
            "Editors can generate, review, and publish a whole-sheet quiz from weak topics.",
          ]}
        />
      ) : null}

      {mastery.length > 0 ? (
        <QuizMasteryProgress
          concepts={mastery}
          title="Topic progress across this set"
          collapsible
          defaultExpanded={!embedded}
          maxVisibleRows={5}
        />
      ) : null}

        {canEdit ? (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="set-quiz-type">Question type</Label>
                <select
                  id="set-quiz-type"
                  className={selectClassName}
                  value={questionType}
                  disabled={generating}
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
                  disabled={generating}
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
                  disabled={generating}
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
              {generating ? "Generating…" : "Generate practice quiz"}
            </Button>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            You have viewer access — you can take quizzes but not generate or edit them.
          </p>
        )}

        {!weakAvailable && hasReadyDocuments && canEdit ? (
          <p className="text-sm text-muted-foreground">
            Complete a quiz on at least one document in this set to unlock adaptive practice.
          </p>
        ) : null}

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {quiz && !results ? (
          <div className="space-y-4 rounded-xl border p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 className="font-medium">{quiz.title}</h3>
                <p className="text-sm text-muted-foreground">
                  {quiz.difficulty.charAt(0).toUpperCase() + quiz.difficulty.slice(1)} difficulty
                  {quiz.published === false ? " · Draft — review before publishing" : ""}
                  {quiz.target_concepts && quiz.target_concepts.length > 0 ? (
                    <> · Targeting: {quiz.target_concepts.map((item) => item.name).join(", ")}</>
                  ) : null}
                </p>
              </div>
              {canEdit ? (
                <div className="flex flex-wrap gap-2">
                  {!editMode ? (
                    <Button type="button" variant="outline" size="sm" onClick={() => void loadEditMode()}>
                      Edit quiz
                    </Button>
                  ) : null}
                  {quiz.published === false ? (
                    <Button
                      type="button"
                      size="sm"
                      disabled={publishing}
                      onClick={() => void handlePublish()}
                    >
                      {publishing ? "Publishing…" : "Publish quiz"}
                    </Button>
                  ) : null}
                  {quiz.published !== false && quiz.public_share_token ? (
                    <Button type="button" variant="outline" size="sm" onClick={() => void copyPublicLink()}>
                      Copy public link
                    </Button>
                  ) : null}
                  {quiz.published !== false ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={exportingQti}
                      onClick={() => void exportQti()}
                    >
                      {exportingQti ? "Exporting…" : "Export QTI"}
                    </Button>
                  ) : null}
                </div>
              ) : null}
            </div>

            {editMode && canEdit ? (
              <div className="space-y-4">
                {editQuestions.map((question, index) => (
                  <div key={question.id} className="space-y-2 rounded-md border p-4">
                    <Label htmlFor={`set-q-${question.id}`}>Question {index + 1}</Label>
                    <textarea
                      id={`set-q-${question.id}`}
                      className="min-h-20 w-full rounded-md border px-3 py-2 text-sm"
                      value={question.question_text}
                      onChange={(event) =>
                        setEditQuestions((current) =>
                          current.map((row) =>
                            row.id === question.id
                              ? { ...row, question_text: event.target.value }
                              : row,
                          ),
                        )
                      }
                    />
                    {question.options.map((option, optionIndex) => (
                      <Input
                        key={`${question.id}-opt-${optionIndex}`}
                        value={option}
                        onChange={(event) =>
                          setEditQuestions((current) =>
                            current.map((row) =>
                              row.id === question.id
                                ? {
                                    ...row,
                                    options: row.options.map((value, idx) =>
                                      idx === optionIndex ? event.target.value : value,
                                    ),
                                  }
                                : row,
                            ),
                          )
                        }
                      />
                    ))}
                    <select
                      className={selectClassName}
                      value={question.correct_option_index ?? 0}
                      onChange={(event) =>
                        setEditQuestions((current) =>
                          current.map((row) =>
                            row.id === question.id
                              ? { ...row, correct_option_index: Number(event.target.value) }
                              : row,
                          ),
                        )
                      }
                    >
                      {question.options.map((option, optionIndex) => (
                        <option key={optionIndex} value={optionIndex}>
                          Correct: {option || `Option ${optionIndex + 1}`}
                        </option>
                      ))}
                    </select>
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      disabled={savingQuestionId === question.id}
                      onClick={() => void saveQuestion(question)}
                    >
                      {savingQuestionId === question.id ? "Saving…" : "Save question"}
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <>
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
                <Button
                  type="button"
                  disabled={submitting || quiz.questions.some((q) => answers[q.id] === undefined)}
                  onClick={() => void submitQuiz()}
                >
                  {submitting ? "Submitting…" : "Submit answers"}
                </Button>
              </>
            )}
          </div>
        ) : null}

        {results && quiz ? (
          <div className="space-y-4 rounded-xl border p-4">
            <div className="rounded-md bg-muted/40 p-3 text-sm">
              <p className="font-medium">
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
                {!result.correct && result.explanation ? (
                  <p className="mt-2 text-muted-foreground">{result.explanation}</p>
                ) : null}
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
        ) : null}
    </div>
  );

  if (embedded) {
    return content;
  }

  return (
    <Card className="shadow-sm" id="set-quiz" data-tour="set-quiz">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Practice quiz (whole sheet)</CardTitle>
        <CardDescription>
          Test yourself on weak topics from every file in this sheet. Take at least one file quiz first.
        </CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}
