"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type DocumentSummary,
  type ExcelAnalysisResponse,
  type ExcelAskResponse,
  type ExcelChart,
  type ExcelPreviewResponse,
  type SourceCitation,
  type WorkspaceSummary,
} from "@/lib/api";
import { SourceCitations } from "@/components/documents/source-citations";
import { ContextBreadcrumb } from "@/components/layout/context-breadcrumb";
import { ChartCardWithExport } from "@/components/excel/chart-export-actions";
import { ExcelChartView } from "@/components/excel/excel-chart-view";
import { ExcelChartBuilder } from "@/components/excel/excel-chart-builder";
import { ExcelPreviewTable } from "@/components/excel/excel-preview-table";
import { ExcelToolsPanel } from "@/components/excel/excel-tools-panel";
import { ChatMessageBubble } from "@/components/ui/chat-message";
import { NotebookTabs } from "@/components/ui/notebook-tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { SourcesRail } from "@/components/workspace/sources-rail";
import {
  ChartBuilderSkeleton,
  ExcelDetailSkeleton,
  ProcessingContentSkeleton,
} from "@/components/ui/loading-skeletons";
import { EXCEL_TAB_LABELS, type ExcelCanvasTab } from "@/lib/notebook-utils";
import { cacheResponseLabel, canEditWorkspace, workspaceRoleLabel } from "@/lib/workspace-roles";

type ExcelChatMessage = {
  question: string;
  answer: string;
  cached?: boolean;
  cacheMatch?: string;
  similarity?: number;
  sources?: string[];
  documentCitations?: SourceCitation[];
};

type ExcelDetailClientProps = {
  documentId: string;
  setId?: string;
};

const VALID_EXCEL_TABS = new Set<string>(Object.keys(EXCEL_TAB_LABELS));

export function ExcelDetailClient({ documentId, setId }: ExcelDetailClientProps) {
  const [analysis, setAnalysis] = useState<ExcelAnalysisResponse | null>(null);
  const [preview, setPreview] = useState<ExcelPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [customCharts, setCustomCharts] = useState<ExcelChart[]>([]);
  const [filename, setFilename] = useState<string>("");
  const [status, setStatus] = useState<string>("pending");
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [workspaceName, setWorkspaceName] = useState<string>("");
  const [workspaceRole, setWorkspaceRole] = useState<WorkspaceSummary["access_role"]>(undefined);
  const [setDocuments, setSetDocuments] = useState<DocumentSummary[]>([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ExcelChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const [canEdit, setCanEdit] = useState(true);
  const [activeTab, setActiveTab] = useState<ExcelCanvasTab>("brief");
  const autoAnalyzed = useRef(false);

  const selectTab = useCallback((tab: ExcelCanvasTab) => {
    setActiveTab(tab);
    window.history.replaceState(null, "", `#${tab}`);
  }, []);

  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    if (VALID_EXCEL_TABS.has(hash)) {
      setActiveTab(hash as ExcelCanvasTab);
    }
  }, []);

  const loadSetDocuments = useCallback(async (token: string, workspaceId: string) => {
    const response = await apiFetch(`/workspaces/${workspaceId}/documents`, token);
    if (response.ok) {
      const data = await response.json();
      setSetDocuments(data.documents ?? []);
    }
  }, []);

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

    const doc = await response.json();
    setFilename(doc.filename);
    setStatus(doc.status);
    setAccessToken(session.access_token);

    let accessRole: WorkspaceSummary["access_role"];
    if (setId) {
      const workspaceResponse = await apiFetch(`/workspaces/${setId}`, session.access_token);
      if (workspaceResponse.ok) {
        const workspaceData = (await workspaceResponse.json()) as WorkspaceSummary;
        accessRole = workspaceData.access_role;
        setWorkspaceRole(workspaceData.access_role);
        setWorkspaceName(workspaceData.name);
        setCanEdit(canEditWorkspace(accessRole));
        await loadSetDocuments(session.access_token, setId);
      }
    }

    setLoading(false);
    return { token: session.access_token, status: doc.status as string, accessRole };
  }, [documentId, setId, loadSetDocuments]);

  const loadPreview = useCallback(async (token: string) => {
    setPreviewLoading(true);
    const response = await apiFetch(`/documents/${documentId}/excel/preview?limit=50`, token);
    setPreviewLoading(false);
    if (!response.ok) {
      return;
    }
    setPreview((await response.json()) as ExcelPreviewResponse);
  }, [documentId]);

  const loadCharts = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/charts`, token);
    if (!response.ok) {
      return false;
    }
    const data = (await response.json()) as ExcelAnalysisResponse;
    setAnalysis(data);
    setStatus(data.status);
    setCustomCharts([]);
    await loadPreview(token);
    return true;
  }, [documentId, loadPreview]);

  const handleCustomChartCreated = useCallback((chart: ExcelChart) => {
    setCustomCharts((current) => {
      const withoutDuplicate = current.filter((item) => item.id !== chart.id);
      return [chart, ...withoutDuplicate];
    });
    selectTab("charts");
  }, [selectTab]);

  const analyzeSpreadsheet = useCallback(async (token: string) => {
    setAnalyzing(true);
    setError(null);
    const response = await apiFetch(`/documents/${documentId}/analyze`, token, {
      method: "POST",
    });
    setAnalyzing(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Analysis failed (${response.status})`);
      return false;
    }
    const data = (await response.json()) as ExcelAnalysisResponse;
    setAnalysis(data);
    setStatus(data.status);
    setCustomCharts([]);
    await loadPreview(token);
    return true;
  }, [documentId, loadPreview]);

  useEffect(() => {
    async function init() {
      const result = await loadDocument();
      if (!result) {
        return;
      }
      const { token, status: docStatus, accessRole } = result;
      if (docStatus === "ready") {
        await loadCharts(token);
        return;
      }
      if (
        (docStatus === "pending" || docStatus === "failed") &&
        !autoAnalyzed.current &&
        canEditWorkspace(accessRole)
      ) {
        autoAnalyzed.current = true;
        await analyzeSpreadsheet(token);
      }
    }
    void init();
  }, [documentId, loadDocument, loadCharts, analyzeSpreadsheet]);

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || !accessToken) {
      return;
    }

    setAsking(true);
    setError(null);
    try {
      const response = await apiFetch(`/documents/${documentId}/excel/ask`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setError(body.error || `Ask failed (${response.status})`);
        return;
      }

      const data = (await response.json()) as ExcelAskResponse;
      setMessages((prev) => [
        ...prev,
        {
          question: trimmed,
          answer: data.answer,
          cached: data.cached,
          cacheMatch: data.cache_match,
          similarity: data.similarity,
          sources: data.sources,
          documentCitations: data.document_citations,
        },
      ]);
      setQuestion("");
    } finally {
      setAsking(false);
    }
  }

  if (loading) {
    return <ExcelDetailSkeleton />;
  }

  const ready = status === "ready";
  const chartCount = (analysis?.charts.length ?? 0) + customCharts.length;
  const tabItems = [
    { id: "brief", label: EXCEL_TAB_LABELS.brief, badge: analysis?.summary ? "✓" : undefined },
    { id: "preview", label: EXCEL_TAB_LABELS.preview },
    { id: "charts", label: EXCEL_TAB_LABELS.charts, badge: chartCount || undefined },
    { id: "builder", label: EXCEL_TAB_LABELS.builder },
  ];

  const breadcrumbItems = setId
    ? [
        { label: "Study sheets", href: "/dashboard/sets" },
        { label: workspaceName || "Study sheet", href: `/dashboard/sets/${setId}` },
        { label: filename },
      ]
    : [{ label: "Study sheets", href: "/dashboard/sets" }, { label: filename }];

  return (
    <div className="space-y-4" data-tour="excel-canvas">
      <ContextBreadcrumb items={breadcrumbItems} />

      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h1 className="truncate font-display text-2xl font-semibold sm:text-3xl">{filename}</h1>
          <p className="mt-1 text-sm capitalize text-muted-foreground">
            Excel · {status}
            {workspaceRole ? ` · ${workspaceRoleLabel(workspaceRole)}` : ""}
          </p>
        </div>
      </div>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      <div className="grid gap-4 xl:grid-cols-[220px_minmax(0,1fr)_260px]">
        {setId ? (
          <SourcesRail
            setId={setId}
            documents={setDocuments}
            activeDocumentId={documentId}
            className="hidden xl:block xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:self-start xl:overflow-y-auto"
          />
        ) : null}

        <section className="min-w-0 space-y-4">
          <Card className="notebook-surface border-0 shadow-none" id="excel-chat">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Chat with this spreadsheet</CardTitle>
              <CardDescription>
                Ask about trends, outliers, and comparisons — grounded in your columns and charts.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {messages.length === 0 ? (
                <div className="rounded-xl border border-dashed bg-muted/20 px-4 py-6 text-center text-sm text-muted-foreground">
                  Try: &quot;Which month had the highest revenue?&quot; or &quot;Summarize the main trend.&quot;
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div key={`${message.question}-${index}`} className="space-y-3">
                      <ChatMessageBubble role="user" question={message.question} />
                      <ChatMessageBubble
                        role="assistant"
                        answer={message.answer}
                        footer={[
                          cacheResponseLabel(message.cached, message.cacheMatch, message.similarity),
                          message.sources?.length ? `Columns: ${message.sources.join(", ")}` : null,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      />
                      {message.documentCitations && message.documentCitations.length > 0 ? (
                        <SourceCitations
                          sources={message.documentCitations}
                          groupByDocument
                          className="ml-1"
                        />
                      ) : null}
                    </div>
                  ))}
                </div>
              )}

              <form onSubmit={handleAsk} className="flex gap-2">
                <Input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a question about your data…"
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
            onChange={(id) => selectTab(id as ExcelCanvasTab)}
          />

          {activeTab === "brief" ? (
            <div id="brief">
              {analyzing ? (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardHeader>
                    <CardTitle>Insights</CardTitle>
                    <CardDescription>Analyzing your spreadsheet…</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ProcessingContentSkeleton lines={5} />
                  </CardContent>
                </Card>
              ) : analysis?.summary ? (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardHeader>
                    <CardTitle>Insights</CardTitle>
                    <CardDescription>
                      AI summary of patterns in your data {analysis.cached ? "(cached)" : ""}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-[420px] overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">
                      {analysis.summary}
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    {canEdit
                      ? "Insights appear after analysis — use Re-analyze in Spreadsheet tools."
                      : "Insights will appear once an editor analyzes this file."}
                  </CardContent>
                </Card>
              )}
            </div>
          ) : null}

          {activeTab === "preview" ? (
            <Card className="notebook-surface border-0 shadow-none" id="preview" data-tour="excel-preview">
              <CardHeader>
                <CardTitle>Data preview</CardTitle>
                <CardDescription>First rows from your spreadsheet</CardDescription>
              </CardHeader>
              <CardContent>
                <ExcelPreviewTable preview={preview} loading={previewLoading || analyzing} />
              </CardContent>
            </Card>
          ) : null}

          {activeTab === "charts" ? (
            <div id="charts" className="space-y-4">
              {customCharts.map((chart) => (
                <Card key={chart.id} className="notebook-surface border-0 shadow-none">
                  <CardHeader>
                    <CardTitle>{chart.title}</CardTitle>
                    <CardDescription>
                      Custom · {chart.chart_type} · {chart.x_column}
                      {chart.y_column ? ` vs ${chart.y_column}` : ""}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ChartCardWithExport chart={chart}>
                      <ExcelChartView chart={chart} />
                    </ChartCardWithExport>
                  </CardContent>
                </Card>
              ))}

              {analysis && analysis.charts.length > 0 ? (
                <h3 className="text-sm font-semibold text-muted-foreground">Suggested charts</h3>
              ) : null}

              {analysis?.charts.map((chart) => (
                <Card key={chart.id} className="notebook-surface border-0 shadow-none">
                  <CardHeader>
                    <CardTitle>{chart.title}</CardTitle>
                    <CardDescription>
                      Suggested · {chart.chart_type} · {chart.x_column}
                      {chart.y_column ? ` vs ${chart.y_column}` : ""}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ChartCardWithExport chart={chart}>
                      <ExcelChartView chart={chart} />
                    </ChartCardWithExport>
                  </CardContent>
                </Card>
              ))}

              {analyzing && !analysis ? <ChartBuilderSkeleton /> : null}

              {!analyzing && !analysis && status !== "ready" ? (
                <p className="text-sm text-muted-foreground">Charts will appear after analysis.</p>
              ) : null}

              {!analyzing && analysis && chartCount === 0 ? (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    No charts yet — try Chart builder or re-analyze for suggestions.
                  </CardContent>
                </Card>
              ) : null}
            </div>
          ) : null}

          {activeTab === "builder" ? (
            <div id="builder">
              {analysis?.profile?.columns && accessToken && canEdit ? (
                <ExcelChartBuilder
                  documentId={documentId}
                  columns={analysis.profile.columns}
                  accessToken={accessToken}
                  onChartCreated={handleCustomChartCreated}
                />
              ) : (
                <Card className="notebook-surface border-0 shadow-none">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    Chart builder unlocks after the spreadsheet is analyzed and ready.
                  </CardContent>
                </Card>
              )}
            </div>
          ) : null}
        </section>

        <aside className="xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:self-start xl:overflow-y-auto">
          <ExcelToolsPanel
            analysis={analysis}
            canEdit={canEdit}
            reanalyzing={analyzing}
            onFocusAsk={() => {
              window.document.getElementById("excel-chat")?.scrollIntoView({ behavior: "smooth" });
            }}
            onReanalyze={async () => {
              if (!accessToken) {
                return;
              }
              await analyzeSpreadsheet(accessToken);
            }}
          />
        </aside>
      </div>
    </div>
  );
}
