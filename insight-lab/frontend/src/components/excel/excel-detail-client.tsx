"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type ExcelAnalysisResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

function SimpleBarChart({ labels, values }: { labels: string[]; values: number[] }) {
  const max = Math.max(...values, 1);
  return (
    <div className="space-y-2">
      {labels.map((label, index) => (
        <div key={`${label}-${index}`} className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span className="truncate pr-2">{label}</span>
            <span>{values[index]}</span>
          </div>
          <div className="h-2 rounded bg-muted">
            <div
              className="h-2 rounded bg-primary"
              style={{ width: `${(values[index] / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ChartRenderer({ chart }: { chart: ExcelAnalysisResponse["charts"][number] }) {
  if (chart.chart_type === "pie") {
    return <SimpleBarChart labels={chart.labels} values={chart.values} />;
  }
  if (chart.chart_type === "scatter") {
    return (
      <p className="text-sm text-muted-foreground">
        Scatter preview: {chart.labels.length} points ({chart.x_column} vs {chart.y_column})
      </p>
    );
  }
  return <SimpleBarChart labels={chart.labels} values={chart.values} />;
}

export function ExcelDetailClient({ documentId }: { documentId: string }) {
  const [analysis, setAnalysis] = useState<ExcelAnalysisResponse | null>(null);
  const [filename, setFilename] = useState<string>("");
  const [status, setStatus] = useState<string>("pending");
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
    setLoading(false);
    return { token: session.access_token, status: doc.status as string };
  }, [documentId]);

  const loadCharts = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/charts`, token);
    if (!response.ok) {
      return false;
    }
    const data = (await response.json()) as ExcelAnalysisResponse;
    setAnalysis(data);
    setStatus(data.status);
    return true;
  }, [documentId]);

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
    return true;
  }, [documentId]);

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

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading spreadsheet...</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link href="/dashboard" className="text-sm text-muted-foreground hover:underline">
            ← Back to dashboard
          </Link>
          <h2 className="mt-2 text-2xl font-semibold">{filename}</h2>
          <p className="text-sm text-muted-foreground capitalize">excel · {status}</p>
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
      {analyzing && <p className="text-sm text-muted-foreground">Analyzing spreadsheet...</p>}

      {analysis?.summary && (
        <Card>
          <CardHeader>
            <CardTitle>Insights</CardTitle>
            <CardDescription>
              LLM narrative {analysis.cached ? "(cached)" : ""}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="whitespace-pre-wrap text-sm leading-relaxed">{analysis.summary}</div>
          </CardContent>
        </Card>
      )}

      {analysis?.charts.map((chart) => (
        <Card key={chart.id}>
          <CardHeader>
            <CardTitle>{chart.title}</CardTitle>
            <CardDescription>
              {chart.chart_type} · {chart.x_column}
              {chart.y_column ? ` vs ${chart.y_column}` : ""}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartRenderer chart={chart} />
          </CardContent>
        </Card>
      ))}

      {!analyzing && !analysis && status !== "ready" && (
        <p className="text-sm text-muted-foreground">Charts will appear after analysis.</p>
      )}
    </div>
  );
}
