"use client";

import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type DocumentSummary,
  type MultiAskResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

type ChatMessage = {
  question: string;
  answer: string;
  cached?: boolean;
  retrievalMethod?: string;
  documentIds: string[];
};

export function MultiDocChatPanel({ documents }: { documents: DocumentSummary[] }) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const readyDocuments = documents.filter(
    (doc) => doc.file_type === "document" && doc.status === "ready",
  );

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
    setSelectedIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }, []);

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || selectedIds.length === 0 || !accessToken) {
      return;
    }

    setAsking(true);
    setError(null);
    try {
      const response = await apiFetch("/documents/multi/ask", accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_ids: selectedIds, question: trimmed }),
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
          question: trimmed,
          answer: data.answer,
          cached: data.cached,
          retrievalMethod: data.retrieval_method,
          documentIds: data.document_ids,
        },
      ]);
      setQuestion("");
    } finally {
      setAsking(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Multi-document chat</CardTitle>
        <CardDescription>
          Ask questions across multiple processed documents with vector search.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {readyDocuments.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Upload and process at least one document to use multi-doc chat.
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

            <form onSubmit={handleAsk} className="flex gap-2">
              <Input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask across selected documents..."
                disabled={asking || selectedIds.length === 0}
              />
              <Button type="submit" disabled={asking || selectedIds.length === 0}>
                {asking ? "Thinking..." : "Ask"}
              </Button>
            </form>
          </>
        )}

        {error && <p className="text-sm text-destructive">{error}</p>}

        <div className="space-y-4">
          {messages.map((message, index) => (
            <div key={`${message.question}-${index}`} className="rounded-md border p-4 text-sm">
              <p className="font-medium">Q: {message.question}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Documents: {message.documentIds.length}
              </p>
              <p className="mt-2 whitespace-pre-wrap text-muted-foreground">{message.answer}</p>
              {message.cached && (
                <p className="mt-2 text-xs text-muted-foreground">Cached response</p>
              )}
              {message.retrievalMethod && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Retrieval: {message.retrievalMethod}
                </p>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
