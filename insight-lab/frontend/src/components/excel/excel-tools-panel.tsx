"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { MessageSquare, Sparkles } from "lucide-react";
import type { ExcelAnalysisResponse } from "@/lib/api";

type PanelTab = "ask" | "insights";

const PANEL_TABS: { id: PanelTab; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "ask", label: "Ask", icon: MessageSquare },
  { id: "insights", label: "Insights", icon: Sparkles },
];

interface ExcelToolsPanelProps {
  analysis: ExcelAnalysisResponse | null;
  canEdit: boolean;
  reanalyzing: boolean;
  onFocusAsk: () => void;
  onReanalyze: () => void;
  className?: string;
}

export function ExcelToolsPanel({
  analysis,
  canEdit,
  reanalyzing,
  onFocusAsk,
  onReanalyze,
  className,
}: ExcelToolsPanelProps) {
  const [tab, setTab] = useState<PanelTab>("ask");

  return (
    <aside
      className={cn(
        "flex w-full shrink-0 flex-col rounded-xl border border-border bg-card lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:w-72 xl:w-80",
        className
      )}
      data-tour="excel-tools"
    >
      <div className="border-b border-border px-3 py-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Spreadsheet tools</p>
      </div>
      <div className="flex gap-1 border-b border-border p-2">
        {PANEL_TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={cn(
              "flex flex-1 items-center justify-center gap-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
              tab === id
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted hover:text-foreground"
            )}
          >
            <Icon className="h-3.5 w-3.5 shrink-0" />
            {label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {tab === "ask" && (
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>Ask questions in the chat — answers cite columns and sheets from this file.</p>
            <ul className="list-inside list-disc space-y-1 text-xs">
              <li>Which month had the highest revenue?</li>
              <li>Summarize trends by region</li>
              <li>Which columns have missing data?</li>
            </ul>
            <button
              type="button"
              onClick={onFocusAsk}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs font-medium hover:bg-muted"
            >
              Jump to chat
            </button>
          </div>
        )}
        {tab === "insights" && analysis ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-border bg-muted/30 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">{analysis.profile.row_count}</p>
                <p className="text-[10px] text-muted-foreground">Rows</p>
              </div>
              <div className="rounded-lg border border-border bg-muted/30 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">{analysis.profile.column_count}</p>
                <p className="text-[10px] text-muted-foreground">Columns</p>
              </div>
              <div className="rounded-lg border border-border bg-muted/30 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">{analysis.profile.columns.length}</p>
                <p className="text-[10px] text-muted-foreground">Fields</p>
              </div>
              <div className="rounded-lg border border-border bg-muted/30 p-2 text-center">
                <p className="text-lg font-semibold tabular-nums">{analysis.charts.length}</p>
                <p className="text-[10px] text-muted-foreground">Charts</p>
              </div>
            </div>
            {canEdit ? (
              <button
                type="button"
                onClick={onReanalyze}
                disabled={reanalyzing}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-xs font-medium hover:bg-muted disabled:opacity-50"
              >
                {reanalyzing ? "Re-analyzing…" : "Re-analyze spreadsheet"}
              </button>
            ) : (
              <p className="text-xs text-muted-foreground">Only editors can re-analyze this file.</p>
            )}
          </div>
        ) : null}
        {tab === "insights" && !analysis ? (
          <p className="text-sm text-muted-foreground">Run analysis to see row, column, and chart stats.</p>
        ) : null}
      </div>
    </aside>
  );
}
