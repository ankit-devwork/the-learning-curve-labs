"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type AskResponse,
  type DocumentSummary,
  type MultiAskResponse,
  type MultiRetrieveResponse,
  type SourceCitation,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { SourceCitations, sourceKey } from "@/components/documents/source-citations";

type ChatMessage = {
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cached?: boolean;
};

export function MultiDocChatPanel({ documents }: { documents: DocumentSummary[] }) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [pendingSources, setPendingSources] = useState<SourceCitation[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [selectedSourceKeys, setSelectedSourceKeys] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [retrieving, setRetrieving] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const readyDocuments = documents.filter(
    (doc) => doc.file_type === "document" && doc.status === "ready",
  );
  const pendingDocumentCount = useMemo(
    () => new Set(pendingSources.map((source) => source.document_id)).size,
    [pendingSources],
  );
  const hitlMode = selectedIds.length > 1;

  useEffect(() => {
    async function loadSession() {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      setAccessToken(session?.access_token ?? null);
    }
    void loadSession();
  }, []);

  const toggleDocument = useCallback((documentId: string) => {
    setPendingSources([]);
    setPendingQuestion("");
    setSelectedSourceKeys(new Set());
    setSelectedIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }, []);

  const toggleSource = useCallback((source: SourceCitation) => {
    const key = sourceKey(source);
    setSelectedSourceKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  async function handleAskSingle(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    const documentId = selectedIds[0];
    if (!trimmed || !documentId || !accessToken) {
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

  async function handleFindSources(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || selectedIds.length === 0 || !accessToken) {
      return;
    }

    setRetrieving(true);
    setError(null);
    setPendingSources([]);
    try {
      const response = await apiFetch("/documents/multi/retrieve", accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: selectedIds, question: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || `Could not search documents (${response.status})`);
        return;
      }

      const data = (await response.json()) as MultiRetrieveResponse;
      setPendingQuestion(trimmed);
      setPendingSources(data.sources);
      setSelectedSourceKeys(new Set(data.sources.map((source) => sourceKey(source))));
    } finally {
      setRetrieving(false);
    }
  }

  async function handleGenerateAnswer() {
    if (!accessToken || !pendingQuestion || pendingSources.length === 0) {
      return;
    }

    const approved = pendingSources.filter((source) => selectedSourceKeys.has(sourceKey(source)));
    if (approved.length === 0) {
      setError("Select at least one source to continue.");
      return;
    }

    setAsking(true);
    setError(null);
    try {
      const response = await apiFetch("/documents/multi/ask", accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_ids: selectedIds,
          question: pendingQuestion,
          approved_sources: approved.map((source) => ({
            document_id: source.document_id,
            chunk_index: source.chunk_index,
            similarity: source.similarity,
          })),
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || `Ask failed (${response.status})`);
        return;
      }

      const data = (await response.json()) as MultiAskResponse;
      setMessages((prev) => [
        ...prev,
        {
          question: pendingQuestion,
          answer: data.answer,
          sources: data.sources,
          cached: data.cached,
        },
      ]);
      setQuestion("");
      setPendingQuestion("");
      setPendingSources([]);
      setSelectedSourceKeys(new Set());
    } finally {
      setAsking(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Document chat</CardTitle>
        <CardDescription>
          Select one or more documents and ask a question. With multiple documents selected, you
          can review sources before the answer is generated.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {readyDocuments.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Upload and process at least one document to use document chat.
          </p>
        ) : (
          <>
            <div className="space-y-2">
              <p className="text-sm font-medium">Select documents</p>
              <div className="flex flex-wrap gap-2">
                {readyDocuments.map((doc) => {
                  const selected = selectedIds.includes(doc.id);
                  return (
                    <Button
                      key={doc.id}
                      type="button"
                      size="sm"
                      variant={selected ? "default" : "outline"}
                      onClick={() => toggleDocument(doc.id)}
                    >
                      {doc.filename}
                    </Button>
                  );
                })}
              </div>
            </div>

            <form
              onSubmit={(event) =>
                void (hitlMode ? handleFindSources(event) : handleAskSingle(event))
              }
              className="flex gap-2"
            >
              <Input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask a question..."
                disabled={retrieving || asking || selectedIds.length === 0}
              />
              <Button type="submit" disabled={retrieving || asking || selectedIds.length === 0}>
                {retrieving || asking ? "Thinking..." : "Ask"}
              </Button>
            </form>

            {hitlMode && pendingSources.length > 0 && (
              <div className="space-y-3 rounded-md border border-dashed p-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Review sources</p>
                  <p className="text-xs text-muted-foreground">
                    {pendingSources.length} result{pendingSources.length === 1 ? "" : "s"} from{" "}
                    {pendingDocumentCount} document{pendingDocumentCount === 1 ? "" : "s"}. Uncheck
                    any you do not need.
                  </p>
                </div>
                <SourceCitations
                  sources={pendingSources}
                  selectable
                  selectedKeys={selectedSourceKeys}
                  onToggle={toggleSource}
                  groupByDocument
                />
                <Button type="button" disabled={asking} onClick={() => void handleGenerateAnswer()}>
                  {asking ? "Thinking..." : "Continue"}
                </Button>
              </div>
            )}
          </>
        )}

        {error && <p className="text-sm text-destructive">{error}</p>}

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
  );
}
