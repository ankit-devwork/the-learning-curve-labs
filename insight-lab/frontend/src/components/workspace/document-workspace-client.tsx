"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type AskResponse,
  type DocumentChunkResponse,
  type DocumentDetail,
  type DocumentSummary,
  type FlashcardSetResponse,
  type ProcessingStatus,
  type QuizResponse,
  type StudyGuideResponse,
  type SummaryResponse,
  type AudioOverviewResponse,
  type SourceCitation,
  type WorkspaceSummary,
} from "@/lib/api";
import { ContextBreadcrumb } from "@/components/layout/context-breadcrumb";
import { DocumentQuizPanel } from "@/components/documents/document-quiz-panel";
import { ConceptGraphPanel } from "@/components/documents/concept-graph-panel";
import { ChatMessageBubble } from "@/components/ui/chat-message";
import { NotebookTabs } from "@/components/ui/notebook-tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { AudioOverviewPanel } from "@/components/workspace/audio-overview-panel";
import { AudioPlayerBar } from "@/components/workspace/audio-player-bar";
import { FlashcardStudy } from "@/components/workspace/flashcard-study";
import { ProcessingStepper } from "@/components/workspace/processing-stepper";
import { SourcesRail } from "@/components/workspace/sources-rail";
import { SourceViewerDrawer } from "@/components/workspace/source-viewer-drawer";
import { StudioPanel } from "@/components/workspace/studio-panel";
import { StudyGuideView } from "@/components/workspace/study-guide-view";
import { SuggestedQuestions } from "@/components/workspace/suggested-questions";
import { STUDIO_TAB_LABELS, type StudioTab } from "@/lib/notebook-utils";
import { cacheResponseLabel, canEditWorkspace, workspaceRoleLabel } from "@/lib/workspace-roles";

type ChatMessage = {
  question: string;
  answer: string;
  sources?: SourceCitation[];
  cached?: boolean;
  cacheMatch?: string;
  similarity?: number;
};

const VALID_TABS = new Set<string>(Object.keys(STUDIO_TAB_LABELS));

export function DocumentWorkspaceClient({
  setId,
  documentId,
}: {
  setId: string;
  documentId: string;
}) {
  const { toast } = useToast();
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [setDocuments, setSetDocuments] = useState<DocumentSummary[]>([]);
  const [workspaceName, setWorkspaceName] = useState<string>("");
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
  const [workspaceRole, setWorkspaceRole] = useState<WorkspaceSummary["access_role"]>(undefined);
  const [existingQuiz, setExistingQuiz] = useState<QuizResponse | null>(null);
  const [flashcards, setFlashcards] = useState<FlashcardSetResponse | null>(null);
  const [studyGuide, setStudyGuide] = useState<StudyGuideResponse | null>(null);
  const [audioOverview, setAudioOverview] = useState<AudioOverviewResponse | null>(null);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState<StudioTab>("brief");
  const [sourceViewer, setSourceViewer] = useState<{
    title: string;
    content: string;
    chunkIndex?: number | null;
  } | null>(null);
  const autoProcessed = useRef(false);
  const audioControls = useRef<{ playPause: () => void; stop: () => void } | null>(null);

  const selectTab = useCallback((tab: StudioTab) => {
    setActiveTab(tab);
    window.history.replaceState(null, "", `#${tab}`);
  }, []);

  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    if (VALID_TABS.has(hash)) {
      setActiveTab(hash as StudioTab);
    }
  }, []);

  const loadSetDocuments = useCallback(async (token: string) => {
    const response = await apiFetch(`/workspaces/${setId}/documents`, token);
    if (response.ok) {
      const data = await response.json();
      setSetDocuments(data.documents ?? []);
    }
  }, [setId]);

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

    const workspaceResponse = await apiFetch(`/workspaces/${setId}`, session.access_token);
    let accessRole: WorkspaceSummary["access_role"];
    if (workspaceResponse.ok) {
      const workspaceData = (await workspaceResponse.json()) as WorkspaceSummary;
      setWorkspaceRole(workspaceData.access_role);
      setWorkspaceName(workspaceData.name);
      accessRole = workspaceData.access_role;
    }

    const data = (await response.json()) as DocumentDetail;
    setDocument(data);
    setAccessToken(session.access_token);
    setLoading(false);
    return { data, token: session.access_token, accessRole };
  }, [documentId, setId]);

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

  const loadFlashcards = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/flashcards`, token);
    if (response.ok) {
      const data = await response.json();
      if (data.flashcards) {
        setFlashcards(data.flashcards as FlashcardSetResponse);
      }
    }
  }, [documentId]);

  const loadStudyGuide = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/study-guide`, token);
    if (response.ok) {
      const data = await response.json();
      if (data.study_guide) {
        setStudyGuide(data.study_guide as StudyGuideResponse);
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
      const { data, token, accessRole } = result;
      await Promise.all([refreshStatus(token), loadSetDocuments(token)]);
      if (data.status === "ready") {
        await Promise.all([
          loadSummary(token),
          loadQuiz(token),
          loadSuggested(token),
          loadFlashcards(token),
          loadStudyGuide(token),
        ]);
        return;
      }
      if (data.status === "pending" && !autoProcessed.current && canEditWorkspace(accessRole)) {
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
  }, [
    documentId,
    loadDocument,
    loadFlashcards,
    loadSetDocuments,
    loadStudyGuide,
    loadSummary,
    loadQuiz,
    loadSuggested,
    processDocument,
    refreshStatus,
  ]);

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

  async function openCitation(source: SourceCitation) {
    if (source.chunk_index != null) {
      await openChunk(source.chunk_index);
    }
  }

  async function generateFlashcards() {
    if (!accessToken) {
      return;
    }
    setStudioBusy(true);
    selectTab("flashcards");
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
    selectTab("guide");
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
  const canEdit = canEditWorkspace(workspaceRole);
  const tabItems = [
    { id: "brief", label: STUDIO_TAB_LABELS.brief },
    { id: "quiz", label: STUDIO_TAB_LABELS.quiz, badge: existingQuiz ? "✓" : undefined },
    { id: "flashcards", label: STUDIO_TAB_LABELS.flashcards, badge: flashcards ? flashcards.card_count : undefined },
    { id: "guide", label: STUDIO_TAB_LABELS.guide, badge: studyGuide ? "✓" : undefined },
    { id: "audio", label: STUDIO_TAB_LABELS.audio, badge: audioOverview ? "✓" : undefined },
    { id: "graph", label: STUDIO_TAB_LABELS.graph },
  ];

  return (
    <div className={audioPlaying ? "space-y-4 pb-24" : "space-y-4"}>
      <ContextBreadcrumb
        items={[
          { label: "Study sheets", href: "/dashboard/sets" },
          { label: workspaceName || "Study sheet", href: `/dashboard/sets/${setId}` },
          { label: document.filename },
        ]}
      />

      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate font-display text-2xl font-semibold sm:text-3xl">{document.filename}</h1>
          <p className="mt-1 text-sm text-muted-foreground capitalize">
            {document.file_type} · {document.status}
            {workspaceRole ? ` · ${workspaceRoleLabel(workspaceRole)}` : ""}
          </p>
        </div>
        {processingStatus && document.status !== "ready" ? (
          <ProcessingStepper
            stage={processingStatus.stage}
            progressPct={processingStatus.progress_pct}
            message={processingStatus.message}
          />
        ) : null}
      </div>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      <div className="grid gap-4 xl:grid-cols-[220px_minmax(0,1fr)_260px]">
        <SourcesRail
          setId={setId}
          documents={setDocuments}
          activeDocumentId={documentId}
          className="hidden xl:block xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:self-start xl:overflow-y-auto"
        />

        <section className="min-w-0 space-y-4">
          <Card className="notebook-surface border-0 shadow-none" id="chat">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Chat with this source</CardTitle>
              <CardDescription>Ask questions grounded in your document. Citations link to excerpts.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {messages.length === 0 ? (
                <div className="rounded-xl border border-dashed bg-muted/20 px-4 py-6 text-center text-sm text-muted-foreground">
                  Try a suggested question below, or ask your own.
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div key={`${message.question}-${index}`} className="space-y-3">
                      <ChatMessageBubble role="user" question={message.question} />
                      <ChatMessageBubble
                        role="assistant"
                        answer={message.answer}
                        sources={message.sources}
                        footer={cacheResponseLabel(message.cached, message.cacheMatch, message.similarity)}
                        onCitationClick={(source) => void openCitation(source)}
                      />
                    </div>
                  ))}
                </div>
              )}

              <SuggestedQuestions
                questions={suggestedQuestions}
                disabled={asking || !ready}
                onSelect={(value) => setQuestion(value)}
              />
              <form onSubmit={handleAsk} className="flex gap-2">
                <Input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a question about this source…"
                  disabled={asking || !ready}
                />
                <Button type="submit" disabled={asking || !ready}>
                  {asking ? "Thinking…" : "Ask"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <NotebookTabs
            tabs={tabItems}
            active={activeTab}
            onChange={(id) => selectTab(id as StudioTab)}
          />

          {activeTab === "brief" ? (
            <Card className="notebook-surface border-0 shadow-none" id="brief">
              <CardHeader>
                <CardTitle className="text-lg">Brief</CardTitle>
                <CardDescription>AI summary of this source — read before quizzing.</CardDescription>
              </CardHeader>
              <CardContent>
                {processing ? (
                  <p className="text-sm text-muted-foreground">Processing document…</p>
                ) : summary ? (
                  <div className="max-h-[420px] overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">
                    {summary}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">Summary will appear after processing.</p>
                )}
              </CardContent>
            </Card>
          ) : null}

          {activeTab === "quiz" ? (
            <div id="quiz">
              <DocumentQuizPanel
                documentId={documentId}
                ready={ready}
                accessToken={accessToken}
                initialQuiz={existingQuiz}
                canEdit={canEdit}
              />
            </div>
          ) : null}

          {activeTab === "flashcards" ? (
            <div id="flashcards">
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
              ) : (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    {canEdit
                      ? "Use Studio → Flashcards to generate a deck from this source."
                      : "No flashcards yet for this source."}
                  </CardContent>
                </Card>
              )}
            </div>
          ) : null}

          {activeTab === "guide" ? (
            <div id="guide">
              {studyGuide ? (
                <StudyGuideView title={studyGuide.title} content={studyGuide.content} />
              ) : (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    {canEdit
                      ? "Use Studio → Study guide to generate a structured overview."
                      : "No study guide yet for this source."}
                  </CardContent>
                </Card>
              )}
            </div>
          ) : null}

          {activeTab === "audio" ? (
            <div id="audio">
              <AudioOverviewPanel
                documentId={documentId}
                accessToken={accessToken}
                ready={ready}
                overview={audioOverview}
                onGenerated={setAudioOverview}
                onControlsReady={(controls) => {
                  audioControls.current = controls;
                }}
                onPlayingChange={(playing, overview) => {
                  setAudioPlaying(playing);
                  if (overview) {
                    setAudioOverview(overview);
                  }
                }}
              />
            </div>
          ) : null}

          {activeTab === "graph" ? (
            <div id="graph">
              <ConceptGraphPanel documentId={documentId} ready={ready} accessToken={accessToken} />
            </div>
          ) : null}
        </section>

        <aside className="space-y-4 xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:self-start xl:overflow-y-auto">
          <StudioPanel
            ready={ready}
            busy={studioBusy}
            canEdit={canEdit}
            onFocusAsk={() => {
              window.document.getElementById("chat")?.scrollIntoView({ behavior: "smooth" });
            }}
            onGenerateQuiz={() => selectTab("quiz")}
            onGenerateFlashcards={() => void generateFlashcards()}
            onGenerateStudyGuide={() => void generateStudyGuide()}
            onGenerateAudioOverview={() => selectTab("audio")}
            onOpenGraph={() => selectTab("graph")}
          />
        </aside>
      </div>

      {audioPlaying && audioOverview ? (
        <AudioPlayerBar
          title={document.filename}
          subtitle="Audio overview"
          playing={audioPlaying}
          onPlayPause={() => audioControls.current?.playPause()}
          onStop={() => audioControls.current?.stop()}
        />
      ) : null}

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
