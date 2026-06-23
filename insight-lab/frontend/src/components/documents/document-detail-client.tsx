"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type AskResponse,
  type DocumentDetail,
  type QuizResponse,
  type SummaryResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureGuide } from "@/components/ui/feature-guide";
import { Input } from "@/components/ui/input";
import { DocumentQuizPanel } from "@/components/documents/document-quiz-panel";
import { SourceCitations } from "@/components/documents/source-citations";
import { cn } from "@/lib/utils";
import type { SourceCitation } from "@/lib/api";

type DetailPanel = "quiz" | "ask";

type ChatMessage = {
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cached?: boolean;
};

export function DocumentDetailClient({ documentId }: { documentId: string }) {
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [existingQuiz, setExistingQuiz] = useState<QuizResponse | null>(null);
  const [activePanel, setActivePanel] = useState<DetailPanel>("ask");
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
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Failed to load document (${response.status})`);
      setLoading(false);
      return null;
    }

    const data = (await response.json()) as DocumentDetail;
    setDocument(data);
    setAccessToken(session.access_token);
    setLoading(false);
    return { data, token: session.access_token };
  }, [documentId]);

  const loadQuiz = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/quiz`, token);
    if (!response.ok) {
      return;
    }
    const data = (await response.json()) as QuizResponse & { quiz?: null };
    if (data.quiz_id) {
      setExistingQuiz(data);
    }
  }, [documentId]);

  const loadSummary = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/summary`, token);
    if (!response.ok) {
      return;
    }
    const data = (await response.json()) as SummaryResponse;
    setSummary(data.summary);
  }, [documentId]);

  const processDocument = useCallback(async (token: string) => {
    setProcessing(true);
    setError(null);
    const response = await apiFetch(`/documents/${documentId}/process`, token, {
      method: "POST",
    });
    setProcessing(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Processing failed (${response.status})`);
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
      if (data.file_type !== "document") {
        setError("Chat is only available for document uploads");
        return;
      }
      if (data.status === "ready") {
        await Promise.all([loadSummary(token), loadQuiz(token)]);
        return;
      }
      if (data.status === "pending" && !autoProcessed.current) {
        autoProcessed.current = true;
        const ok = await processDocument(token);
        if (ok) {
          const refreshed = await loadDocument();
          if (refreshed?.token) {
            await loadSummary(refreshed.token);
          }
        }
      }
    }
    void init();
  }, [documentId, loadDocument, loadSummary, loadQuiz, processDocument]);

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      return;
    }

    setAsking(true);
    setError(null);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        setError("Sign in required");
        return;
      }

      const response = await apiFetch(`/documents/${documentId}/ask`, session.access_token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || `Ask failed (${response.status})`);
        return;
      }

      const data = (await response.json()) as AskResponse;
      setMessages((prev) => [
        ...prev,
        {
          question: trimmed,
          answer: data.answer,
          sources: data.sources,
          cached: data.cached,
        },
      ]);
      setQuestion("");
    } finally {
      setAsking(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading document...</p>;
  }

  if (!document) {
    return <p className="text-sm text-destructive">{error || "Document not found"}</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b pb-6">
        <div className="min-w-0">
          <Link
            href="/dashboard"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            ← Back to workspace
          </Link>
          <h2 className="mt-2 truncate text-2xl font-semibold tracking-tight">{document.filename}</h2>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="rounded-md bg-muted px-2 py-0.5 text-xs font-medium capitalize text-muted-foreground">
              {document.file_type}
            </span>
            <span className="rounded-md bg-muted px-2 py-0.5 text-xs font-medium capitalize text-muted-foreground">
              {document.status}
            </span>
          </div>
        </div>
        <div className="flex gap-2">
          {(document.status === "ready" || document.status === "failed") && (
            <Button
              type="button"
              variant="outline"
              disabled={processing}
              onClick={async () => {
                const supabase = createClient();
                const {
                  data: { session },
                } = await supabase.auth.getSession();
                if (!session?.access_token) {
                  return;
                }
                const ok = await processDocument(session.access_token);
                if (ok) {
                  await loadDocument();
                  await loadSummary(session.access_token);
                }
              }}
            >
              {processing ? "Re-processing..." : "Re-process (refresh embeddings)"}
            </Button>
          )}
          {document.status !== "ready" && document.status !== "failed" && (
            <Button
              type="button"
              disabled={processing}
              onClick={async () => {
                const supabase = createClient();
                const {
                  data: { session },
                } = await supabase.auth.getSession();
                if (!session?.access_token) {
                  return;
                }
                const ok = await processDocument(session.access_token);
                if (ok) {
                  await loadDocument();
                  await loadSummary(session.access_token);
                }
              }}
            >
              {processing ? "Processing..." : "Process document"}
            </Button>
          )}
        </div>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {document.error_message && (
        <p className="text-sm text-destructive">{document.error_message}</p>
      )}

      {document.status === "ready" && (
        <FeatureGuide
          title="What you can do here"
          steps={[
            "Read the Summary for a quick overview of the document.",
            "Use Ask (right panel) to type questions — answers cite which file they came from.",
            "Use Quiz (left panel) to generate questions, submit answers, and track weak topics.",
          ]}
        />
      )}

      {document.status !== "ready" && !processing && (
        <FeatureGuide
          title="Processing"
          steps={[
            "We extract text, build search embeddings, and generate a summary — this runs automatically.",
            "When status shows Ready, refresh if needed and the summary, Ask, and Quiz panels will unlock.",
          ]}
        />
      )}

      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg">Summary</CardTitle>
          <CardDescription>
            AI-generated overview — read this first before asking questions or taking a quiz.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {processing && <p className="text-sm text-muted-foreground">Processing document...</p>}
          {!processing && summary && (
            <div className="max-h-64 overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
              {summary}
            </div>
          )}
          {!processing && !summary && document.status !== "ready" && (
            <p className="text-sm text-muted-foreground">Summary will appear after processing.</p>
          )}
        </CardContent>
      </Card>

      <div className="space-y-4">
        <div className="flex gap-2 lg:hidden">
          <Button
            type="button"
            variant={activePanel === "quiz" ? "default" : "outline"}
            size="sm"
            onClick={() => setActivePanel("quiz")}
          >
            Quiz
          </Button>
          <Button
            type="button"
            variant={activePanel === "ask" ? "default" : "outline"}
            size="sm"
            onClick={() => setActivePanel("ask")}
          >
            Ask
          </Button>
        </div>

        <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
          <div
            className={cn(
              "lg:max-h-[calc(100vh-11rem)] lg:overflow-y-auto lg:pr-1",
              activePanel !== "quiz" && "hidden lg:block",
            )}
          >
            <DocumentQuizPanel
              documentId={documentId}
              ready={document.status === "ready"}
              accessToken={accessToken}
              initialQuiz={existingQuiz}
            />
          </div>

          <div
            className={cn(
              "lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto lg:pl-1",
              activePanel !== "ask" && "hidden lg:block",
            )}
          >
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle className="text-lg">Ask this document</CardTitle>
                <CardDescription>
                  Type a question in plain English. The answer is based on this file only — sources
                  appear below each reply.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <form onSubmit={handleAsk} className="flex gap-2">
                  <Input
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="e.g. What are the main conclusions?"
                    disabled={asking || document.status !== "ready"}
                  />
                  <Button type="submit" disabled={asking || document.status !== "ready"}>
                    {asking ? "Thinking..." : "Ask"}
                  </Button>
                </form>

                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div key={`${message.question}-${index}`} className="rounded-md border p-4 text-sm">
                      <p className="font-medium">Q: {message.question}</p>
                      <p className="mt-2 whitespace-pre-wrap text-muted-foreground">{message.answer}</p>
                      {message.sources && message.sources.length > 0 && (
                        <SourceCitations sources={message.sources} className="mt-3" />
                      )}
                      {message.cached && (
                        <p className="mt-2 text-xs text-muted-foreground">Cached response</p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
