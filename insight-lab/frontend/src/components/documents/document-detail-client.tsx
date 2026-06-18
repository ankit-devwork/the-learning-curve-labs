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
import { Input } from "@/components/ui/input";
import { DocumentQuizPanel } from "@/components/documents/document-quiz-panel";
import { SourceCitations } from "@/components/documents/source-citations";
import type { SourceCitation } from "@/lib/api";

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
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link href="/dashboard" className="text-sm text-muted-foreground hover:underline">
            ← Back to dashboard
          </Link>
          <h2 className="mt-2 text-2xl font-semibold">{document.filename}</h2>
          <p className="text-sm text-muted-foreground capitalize">
            {document.file_type} · {document.status}
          </p>
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

      <Card>
        <CardHeader>
          <CardTitle>Summary</CardTitle>
          <CardDescription>Generated by the backend LLM (cached in Redis when enabled)</CardDescription>
        </CardHeader>
        <CardContent>
          {processing && <p className="text-sm text-muted-foreground">Processing document...</p>}
          {!processing && summary && (
            <div className="whitespace-pre-wrap text-sm leading-relaxed">{summary}</div>
          )}
          {!processing && !summary && document.status !== "ready" && (
            <p className="text-sm text-muted-foreground">Summary will appear after processing.</p>
          )}
        </CardContent>
      </Card>

      <ConceptGraphPanel
        documentId={documentId}
        ready={document.status === "ready"}
        accessToken={accessToken}
      />

      <DocumentQuizPanel
        documentId={documentId}
        ready={document.status === "ready"}
        accessToken={accessToken}
        initialQuiz={existingQuiz}
      />

      <Card>
        <CardHeader>
          <CardTitle>Ask this document</CardTitle>
          <CardDescription>
            Answers are grounded in passages from this document. Sources are shown below each reply.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form onSubmit={handleAsk} className="flex gap-2">
            <Input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about this document..."
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
  );
}
