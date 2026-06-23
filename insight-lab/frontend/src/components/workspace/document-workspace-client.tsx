"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type AskResponse,
  type DocumentChunkResponse,
  type DocumentDetail,
  type FlashcardSetResponse,
  type ProcessingStatus,
  type QuizResponse,
  type StudyGuideResponse,
  type SummaryResponse,
  type AudioOverviewResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { DocumentQuizPanel } from "@/components/documents/document-quiz-panel";
import { ConceptGraphPanel } from "@/components/documents/concept-graph-panel";
import { SourceCitations } from "@/components/documents/source-citations";
import { FlashcardStudy } from "@/components/workspace/flashcard-study";
import { ProcessingStepper } from "@/components/workspace/processing-stepper";
import { SourceViewerDrawer } from "@/components/workspace/source-viewer-drawer";
import { StudioPanel } from "@/components/workspace/studio-panel";
import { StudyGuideView } from "@/components/workspace/study-guide-view";
import { AudioOverviewPanel } from "@/components/workspace/audio-overview-panel";
import { SuggestedQuestions } from "@/components/workspace/suggested-questions";
import type { SourceCitation } from "@/lib/api";

type ChatMessage = {
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cached?: boolean;
};

export function DocumentWorkspaceClient({
  setId,
  documentId,
}: {
  setId: string;
  documentId: string;
}) {
  const { toast } = useToast();
  const askRef = useRef<HTMLDivElement>(null);
  const quizRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLDivElement>(null);
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [asking, setAsking] = useState(false);
  const [studioBusy, setStudioBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [existingQuiz, setExistingQuiz] = useState<QuizResponse | null>(null);
  const [flashcards, setFlashcards] = useState<FlashcardSetResponse | null>(null);
  const [studyGuide, setStudyGuide] = useState<StudyGuideResponse | null>(null);
  const [audioOverview, setAudioOverview] = useState<AudioOverviewResponse | null>(null);
  const [sourceViewer, setSourceViewer] = useState<{
    title: string;
    content: string;
    chunkIndex?: number | null;
  } | null>(null);
  const autoProcessed = useRef(false);

  const loadDocument = useCallback(async () => {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setError("Sign in required");
      setLoading(false);
      return null;
    }

    const response = await apiFetch(`/documents/${documentId}`, session.access_token);
    if (!response.ok) {
      setError("Failed to load document");
      setLoading(false);
      return null;
    }

    const data = (await response.json()) as DocumentDetail;
    setDocument(data);
    setAccessToken(session.access_token);
    setLoading(false);
    return { data, token: session.access_token };
  }, [documentId]);

  const refreshStatus = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/status`, token);
    if (response.ok) {
      setProcessingStatus((await response.json()) as ProcessingStatus);
    }
  }, [documentId]);

  const loadSummary = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/summary`, token);
    if (response.ok) {
      const data = (await response.json()) as SummaryResponse;
      setSummary(data.summary);
    }
  }, [documentId]);

  const loadSuggested = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/suggested-questions`, token);
    if (response.ok) {
      const data = await response.json();
      setSuggestedQuestions(data.questions ?? []);
    }
  }, [documentId]);

  const loadQuiz = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/quiz`, token);
    if (response.ok) {
      const data = (await response.json()) as QuizResponse;
      if (data.quiz_id) {
        setExistingQuiz(data);
      }
    }
  }, [documentId]);

  const processDocument = useCallback(async (token: string) => {
    setProcessing(true);
    const response = await apiFetch(`/documents/${documentId}/process`, token, { method: "POST" });
    setProcessing(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Processing failed");
      return false;
    }
    return true;
  }, [documentId]);

  useEffect(() => {
    async function init() {
      const result = await loadDocument();
      if (!result) {
        return;
      }
      const { data, token } = result;
      await refreshStatus(token);
      if (data.status === "ready") {
        await Promise.all([loadSummary(token), loadQuiz(token), loadSuggested(token)]);
        return;
      }
      if (data.status === "pending" && !autoProcessed.current) {
        autoProcessed.current = true;
        const ok = await processDocument(token);
        if (ok) {
          await loadDocument();
          await refreshStatus(token);
          await loadSummary(token);
        }
      }
    }
    void init();
  }, [documentId, loadDocument, loadSummary, loadQuiz, loadSuggested, processDocument, refreshStatus]);

  useEffect(() => {
    if (!accessToken || !document || document.status === "ready") {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshStatus(accessToken);
      void loadDocument();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [accessToken, document, loadDocument, refreshStatus]);

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || !accessToken) {
      return;
    }
    setAsking(true);
    setError(null);
    try {
      const response = await apiFetch(`/documents/${documentId}/ask`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || "Ask failed");
        return;
      }
      const data = (await response.json()) as AskResponse;
      setMessages((prev) => [
        ...prev,
        { question: trimmed, answer: data.answer, sources: data.sources, cached: data.cached },
      ]);
      setQuestion("");
    } finally {
      setAsking(false);
    }
  }

  async function openChunk(chunkIndex: number) {
    if (!accessToken || !document) {
      return;
    }
    const response = await apiFetch(
      `/documents/${documentId}/chunks/${chunkIndex}`,
      accessToken,
    );
    if (!response.ok) {
      return;
    }
    const chunk = (await response.json()) as DocumentChunkResponse;
    setSourceViewer({
      title: document.filename,
      content: chunk.content,
      chunkIndex: chunk.chunk_index,
    });
  }

  async function generateFlashcards() {
    if (!accessToken) {
      return;
    }
    setStudioBusy(true);
    try {
      const response = await apiFetch(`/documents/${documentId}/flashcards/generate`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_cards: 10 }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({ title: "Flashcard generation failed", description: body.error, variant: "error" });
        return;
      }
      setFlashcards((await response.json()) as FlashcardSetResponse);
      toast({ title: "Flashcards ready", variant: "success" });
    } finally {
      setStudioBusy(false);
    }
  }

  async function generateStudyGuide() {
    if (!accessToken) {
      return;
    }
    setStudioBusy(true);
    try {
      const response = await apiFetch(`/documents/${documentId}/study-guide/generate`, accessToken, {
        method: "POST",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({ title: "Study guide failed", description: body.error, variant: "error" });
        return;
      }
      setStudyGuide((await response.json()) as StudyGuideResponse);
      toast({ title: "Study guide ready", variant: "success" });
    } finally {
      setStudioBusy(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading workspace…</p>;
  }

  if (!document) {
    return <p className="text-sm text-destructive">{error || "Document not found"}</p>;
  }

  const ready = document.status === "ready";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b pb-4">
        <div className="min-w-0">
          <Link
            href={`/dashboard/sets/${setId}`}
            className="text-sm text-muted-foreground hover:text-primary"
          >
            ← Back to study set
          </Link>
          <h1 className="mt-2 truncate text-2xl font-semibold">{document.filename}</h1>
        </div>
      </div>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      <div className="grid gap-4 xl:grid-cols-[240px_minmax(0,1fr)_280px]">
        <aside className="space-y-4">
          <Card className="shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Source</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p className="capitalize">{document.file_type}</p>
              <p className="capitalize">Status: {document.status}</p>
            </CardContent>
          </Card>
          {processingStatus ? (
            <ProcessingStepper
              stage={processingStatus.stage}
              progressPct={processingStatus.progress_pct}
              message={processingStatus.message}
            />
          ) : null}
        </aside>

        <section className="space-y-4 min-w-0">
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-lg">Summary</CardTitle>
              <CardDescription>Read this first, then ask questions or use Studio tools.</CardDescription>
            </CardHeader>
            <CardContent>
              {processing ? (
                <p className="text-sm text-muted-foreground">Processing document…</p>
              ) : summary ? (
                <div className="max-h-56 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
                  {summary}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Summary will appear after processing.</p>
              )}
            </CardContent>
          </Card>

          <Card className="shadow-sm" ref={askRef}>
            <CardHeader>
              <CardTitle className="text-lg">Ask this document</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <SuggestedQuestions
                questions={suggestedQuestions}
                disabled={asking || !ready}
                onSelect={(value) => setQuestion(value)}
              />
              <form onSubmit={handleAsk} className="flex gap-2">
                <Input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a question…"
                  disabled={asking || !ready}
                />
                <Button type="submit" disabled={asking || !ready}>
                  {asking ? "Thinking…" : "Ask"}
                </Button>
              </form>
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div key={`${message.question}-${index}`} className="rounded-md border p-4 text-sm">
                    <p className="font-medium">Q: {message.question}</p>
                    <p className="mt-2 whitespace-pre-wrap text-muted-foreground">{message.answer}</p>
                    {message.sources && message.sources.length > 0 ? (
                      <div className="mt-3 space-y-2">
                        <SourceCitations sources={message.sources} />
                        {message.sources[0]?.chunk_index != null ? (
                          <Button
                            type="button"
                            variant="link"
                            className="h-auto p-0 text-xs"
                            onClick={() => void openChunk(message.sources![0].chunk_index!)}
                          >
                            View source excerpt
                          </Button>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div ref={quizRef}>
            <DocumentQuizPanel
              documentId={documentId}
              ready={ready}
              accessToken={accessToken}
              initialQuiz={existingQuiz}
            />
          </div>

          {flashcards ? (
            <FlashcardStudy
              title={flashcards.title}
              setId={flashcards.set_id}
              cards={flashcards.cards}
              onReview={async (flashcardId, knew) => {
                if (!accessToken) {
                  return;
                }
                await apiFetch(`/flashcards/${flashcards.set_id}/review`, accessToken, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ flashcard_id: flashcardId, knew }),
                });
              }}
              onViewSource={(card) => {
                if (card.source_chunk_index != null) {
                  void openChunk(card.source_chunk_index);
                }
              }}
            />
          ) : null}

          {studyGuide ? (
            <StudyGuideView title={studyGuide.title} content={studyGuide.content} />
          ) : null}

          <div ref={audioRef}>
            <AudioOverviewPanel
              documentId={documentId}
              accessToken={accessToken}
              ready={ready}
              overview={audioOverview}
              onGenerated={setAudioOverview}
            />
          </div>

          <div ref={graphRef}>
            <ConceptGraphPanel documentId={documentId} ready={ready} accessToken={accessToken} />
          </div>
        </section>

        <aside className="space-y-4">
          <StudioPanel
            ready={ready}
            busy={studioBusy}
            onFocusAsk={() => askRef.current?.scrollIntoView({ behavior: "smooth" })}
            onGenerateQuiz={() => quizRef.current?.scrollIntoView({ behavior: "smooth" })}
            onGenerateFlashcards={() => void generateFlashcards()}
            onGenerateStudyGuide={() => void generateStudyGuide()}
            onGenerateAudioOverview={() =>
              audioRef.current?.scrollIntoView({ behavior: "smooth" })
            }
            onOpenGraph={() => graphRef.current?.scrollIntoView({ behavior: "smooth" })}
          />
        </aside>
      </div>

      <SourceViewerDrawer
        open={Boolean(sourceViewer)}
        title={sourceViewer?.title ?? ""}
        content={sourceViewer?.content ?? ""}
        chunkIndex={sourceViewer?.chunkIndex}
        onClose={() => setSourceViewer(null)}
      />
    </div>
  );
}
