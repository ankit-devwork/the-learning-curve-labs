"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  type ExcelAnalysisResponse,
  type ExcelAskResponse,
  type ExcelChart,
  type ExcelPreviewResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FeatureGuide } from "@/components/ui/feature-guide";
import { Input } from "@/components/ui/input";
import { ChartCardWithExport } from "@/components/excel/chart-export-actions";
import { ExcelChartView } from "@/components/excel/excel-chart-view";
import { ExcelChartBuilder } from "@/components/excel/excel-chart-builder";
import { ExcelPreviewTable } from "@/components/excel/excel-preview-table";
import {
  ChartBuilderSkeleton,
  ExcelDetailSkeleton,
  ProcessingContentSkeleton,
} from "@/components/ui/loading-skeletons";

type ExcelChatMessage = {
  question: string;
  answer: string;
  cached?: boolean;
  sources?: string[];
};

type ExcelDetailClientProps = {
  documentId: string;
  setId?: string;
};

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
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ExcelChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const autoAnalyzed = useRef(false);

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
    setLoading(false);
    return { token: session.access_token, status: doc.status as string };
  }, [documentId]);

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
  }, []);

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
      const { token, status: docStatus } = result;
      if (docStatus === "ready") {
        await loadCharts(token);
        return;
      }
      if ((docStatus === "pending" || docStatus === "failed") && !autoAnalyzed.current) {
        autoAnalyzed.current = true;
        await analyzeSpreadsheet(token);
      }
    }
    void init();
  }, [documentId, loadDocument, loadCharts, analyzeSpreadsheet]);

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

      const response = await apiFetch(`/documents/${documentId}/excel/ask`, session.access_token, {
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
          sources: data.sources,
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

  const backHref = setId ? `/dashboard/sets/${setId}` : "/dashboard/sets";

  return (
    <div className="space-y-6" data-tour="excel-canvas">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b pb-6">
        <div className="min-w-0">
          <Link
            href={backHref}
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            ← Back to study set
          </Link>
          <h2 className="mt-2 truncate text-2xl font-semibold tracking-tight">{filename}</h2>
          <span className="mt-2 inline-block rounded-md bg-muted px-2 py-0.5 text-xs font-medium capitalize text-muted-foreground">
            excel · {status}
          </span>
        </div>
        <Button
          type="button"
          variant="outline"
          disabled={analyzing}
          onClick={async () => {
            const supabase = createClient();
            const {
              data: { session },
            } = await supabase.auth.getSession();
            if (!session?.access_token) {
              return;
            }
            await analyzeSpreadsheet(session.access_token);
          }}
        >
          {analyzing ? "Analyzing..." : "Re-analyze"}
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {status === "ready" && (
        <FeatureGuide
          title="Excel canvas"
          steps={[
            "Preview your data and charts on the left.",
            "Chat on the right to ask questions in plain English.",
            "Export any chart as CSV or PNG for reports and slides.",
          ]}
        />
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.9fr)]">
        <div className="space-y-6 min-w-0">
          {analyzing && (
            <Card>
              <CardHeader>
                <CardTitle>Insights</CardTitle>
                <CardDescription>Analyzing your spreadsheet…</CardDescription>
              </CardHeader>
              <CardContent>
                <ProcessingContentSkeleton lines={5} />
              </CardContent>
            </Card>
          )}

          {analysis?.summary && (
            <Card>
              <CardHeader>
                <CardTitle>Insights</CardTitle>
                <CardDescription>
                  AI summary of patterns in your data {analysis.cached ? "(cached)" : ""}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{analysis.summary}</div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Data preview</CardTitle>
              <CardDescription>First rows from your spreadsheet</CardDescription>
            </CardHeader>
            <CardContent>
              <ExcelPreviewTable preview={preview} loading={previewLoading || analyzing} />
            </CardContent>
          </Card>

          {analysis?.profile?.columns && accessToken && (
            <ExcelChartBuilder
              documentId={documentId}
              columns={analysis.profile.columns}
              accessToken={accessToken}
              onChartCreated={handleCustomChartCreated}
            />
          )}

          {customCharts.map((chart) => (
            <Card key={chart.id}>
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

          {analysis && analysis.charts.length > 0 && (
            <h3 className="text-lg font-semibold">Suggested charts</h3>
          )}

          {analysis?.charts.map((chart) => (
            <Card key={chart.id}>
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

          {analyzing && !analysis && <ChartBuilderSkeleton />}

          {!analyzing && !analysis && status !== "ready" && (
            <p className="text-sm text-muted-foreground">Charts will appear after analysis.</p>
          )}
        </div>

        <aside className="xl:sticky xl:top-6 xl:self-start">
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Ask this spreadsheet</CardTitle>
              <CardDescription>
                Julius-style chat grounded in your columns, charts, and insights.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleAsk} className="flex gap-2">
                <Input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. Which month had the highest revenue?"
                  disabled={asking || status !== "ready"}
                />
                <Button type="submit" disabled={asking || status !== "ready"}>
                  {asking ? "Thinking..." : "Ask"}
                </Button>
              </form>

              <div className="max-h-[70vh] space-y-4 overflow-y-auto pr-1">
                {messages.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    Ask about trends, outliers, or comparisons in your data.
                  </p>
                ) : null}
                {messages.map((message, index) => (
                  <div key={`${message.question}-${index}`} className="rounded-md border p-4 text-sm">
                    <p className="font-medium">Q: {message.question}</p>
                    <p className="mt-2 whitespace-pre-wrap text-muted-foreground">{message.answer}</p>
                    {message.cached && (
                      <p className="mt-2 text-xs text-muted-foreground">Cached response</p>
                    )}
                    {message.sources && message.sources.length > 0 && (
                      <p className="mt-1 text-xs text-muted-foreground">
                        Sources: {message.sources.join(", ")}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}
