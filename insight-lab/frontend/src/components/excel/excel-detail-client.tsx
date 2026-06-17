"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type ExcelAnalysisResponse, type ExcelChart } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ExcelChartView } from "@/components/excel/excel-chart-view";
import { ExcelChartBuilder } from "@/components/excel/excel-chart-builder";

export function ExcelDetailClient({ documentId }: { documentId: string }) {
  const [analysis, setAnalysis] = useState<ExcelAnalysisResponse | null>(null);
  const [customCharts, setCustomCharts] = useState<ExcelChart[]>([]);
  const [filename, setFilename] = useState<string>("");
  const [status, setStatus] = useState<string>("pending");
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
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

  const loadCharts = useCallback(async (token: string) => {
    const response = await apiFetch(`/documents/${documentId}/charts`, token);
    if (!response.ok) {
      return false;
    }
    const data = (await response.json()) as ExcelAnalysisResponse;
    setAnalysis(data);
    setStatus(data.status);
    setCustomCharts([]);
    return true;
  }, [documentId]);

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
            <ExcelChartView chart={chart} />
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
            <ExcelChartView chart={chart} />
          </CardContent>
        </Card>
      ))}

      {!analyzing && !analysis && status !== "ready" && (
        <p className="text-sm text-muted-foreground">Charts will appear after analysis.</p>
      )}
    </div>
  );
}
