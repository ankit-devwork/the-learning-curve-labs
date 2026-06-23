"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type AskResponse,
  type DocumentReviewOption,
  type DocumentSummary,
  type MultiAskResponse,
  type MultiRetrieveResponse,
  type SourceCitation,
} from "@/lib/api";
import { ChatMessageBubble } from "@/components/ui/chat-message";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureGuide } from "@/components/ui/feature-guide";
import { Input } from "@/components/ui/input";
import { DocumentReviewPicker } from "@/components/documents/document-review-picker";
import { cacheResponseLabel } from "@/lib/workspace-roles";

type ChatMessage = {
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cached?: boolean;
  cacheMatch?: string;
  similarity?: number;
};

export function MultiDocChatPanel({
  documents,
  workspaceId,
}: {
  documents: DocumentSummary[];
  workspaceId?: string;
}) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [pendingDocuments, setPendingDocuments] = useState<DocumentReviewOption[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [approvedDocumentIds, setApprovedDocumentIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [retrieving, setRetrieving] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const readyDocuments = documents.filter(
    (doc) => doc.file_type === "document" && doc.status === "ready",
  );
  const excelDocuments = documents.filter((doc) => doc.file_type === "excel");
  const processingTextDocuments = documents.filter(
    (doc) => doc.file_type === "document" && doc.status !== "ready",
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
    setPendingDocuments([]);
    setPendingQuestion("");
    setApprovedDocumentIds(new Set());
    setSelectedIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }, []);

  const toggleReviewDocument = useCallback((documentId: string) => {
    setApprovedDocumentIds((current) => {
      const next = new Set(current);
      if (next.has(documentId)) {
        next.delete(documentId);
      } else {
        next.add(documentId);
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
          cacheMatch: data.cache_match,
          similarity: data.similarity,
        },
      ]);
      setQuestion("");
    } finally {
      setAsking(false);
    }
  }

  async function handlePrepareReview(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || selectedIds.length < 2 || !accessToken) {
      return;
    }

    setRetrieving(true);
    setError(null);
    setPendingDocuments([]);
    try {
      const response = await apiFetch("/documents/multi/retrieve", accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_ids: selectedIds,
          question: trimmed,
          workspace_id: workspaceId,
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || `Could not prepare document summaries (${response.status})`);
        return;
      }

      const data = (await response.json()) as MultiRetrieveResponse;
      setPendingQuestion(trimmed);
      setPendingDocuments(data.documents);
      setApprovedDocumentIds(
        new Set(data.documents.map((document) => document.document_id)),
      );
    } finally {
      setRetrieving(false);
    }
  }

  async function handleContinue() {
    if (!accessToken || !pendingQuestion || pendingDocuments.length === 0) {
      return;
    }

    const approved = [...approvedDocumentIds];
    if (approved.length === 0) {
      setError("Select at least one document to continue.");
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
          approved_document_ids: approved,
          workspace_id: workspaceId,
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
          cacheMatch: data.cache_match,
          similarity: data.similarity,
        },
      ]);
      setQuestion("");
      setPendingQuestion("");
      setPendingDocuments([]);
      setApprovedDocumentIds(new Set());
    } finally {
      setAsking(false);
    }
  }

  return (
    <Card className="notebook-surface border-0 shadow-none" data-tour="compare-docs">
      <CardHeader>
        <CardTitle className="text-lg">Compare documents</CardTitle>
        <CardDescription>
          Ask one question across multiple files — useful for comparing readings or lecture notes.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FeatureGuide
          title="Steps"
          steps={[
            "Select one document for a direct answer, or two or more to compare.",
            "With multiple files, you will see a short summary per document — pick which to include.",
            "Click Continue to get an answer combined from your selected documents.",
          ]}
        />
        {readyDocuments.length === 0 ? (
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>
              Compare works with <strong className="font-medium text-foreground">ready PDF and Word files</strong>{" "}
              in this study set. Excel spreadsheets use a separate chat and chart view.
            </p>
            {processingTextDocuments.length > 0 ? (
              <p>
                {processingTextDocuments.length} document file
                {processingTextDocuments.length === 1 ? "" : "s"} still processing — open them from the study set and
                wait for status <strong className="font-medium text-foreground">Ready</strong>.
              </p>
            ) : null}
            {excelDocuments.length > 0 ? (
              <div className="rounded-md border bg-muted/30 p-4">
                <p className="font-medium text-foreground">Excel files in this set</p>
                <ul className="mt-2 space-y-2">
                  {excelDocuments.map((doc) => (
                    <li key={doc.id}>
                      <Link
                        href={
                          workspaceId
                            ? `/dashboard/sets/${workspaceId}/excel/${doc.id}`
                            : `/dashboard/excel/${doc.id}`
                        }
                        className="text-primary hover:underline"
                      >
                        {doc.filename}
                      </Link>
                      <span className="ml-2 capitalize">· {doc.status}</span>
                      {doc.status !== "ready" ? (
                        <span className="ml-1">— open to analyze</span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p>Upload at least one PDF or Word file to this study set to compare documents here.</p>
            )}
          </div>
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
                void (hitlMode ? handlePrepareReview(event) : handleAskSingle(event))
              }
              className="flex gap-2"
            >
              <Input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="e.g. How do these sources disagree on climate policy?"
                disabled={retrieving || asking || selectedIds.length === 0}
              />
              <Button type="submit" disabled={retrieving || asking || selectedIds.length === 0}>
                {retrieving || asking ? "Thinking..." : "Ask"}
              </Button>
            </form>

            {hitlMode && pendingDocuments.length > 0 && (
              <div className="space-y-3 rounded-md border border-dashed p-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Which document should we answer from?</p>
                  <p className="text-xs text-muted-foreground">
                    Read the summaries below and select one or more documents for your answer.
                  </p>
                </div>
                <DocumentReviewPicker
                  documents={pendingDocuments}
                  selectedIds={approvedDocumentIds}
                  onToggle={toggleReviewDocument}
                />
                <Button type="button" disabled={asking} onClick={() => void handleContinue()}>
                  {asking ? "Thinking..." : "Continue"}
                </Button>
              </div>
            )}
          </>
        )}

        {error && <p className="text-sm text-destructive">{error}</p>}

        <div className="space-y-4">
          {messages.map((message, index) => (
            <div key={`${message.question}-${index}`} className="space-y-3">
              <ChatMessageBubble role="user" question={message.question} />
              <ChatMessageBubble
                role="assistant"
                answer={message.answer}
                sources={message.sources}
                footer={cacheResponseLabel(message.cached, message.cacheMatch, message.similarity)}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
