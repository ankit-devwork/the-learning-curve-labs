"use client";

import { cn } from "@/lib/utils";
import type { ExcelAnalysisResponse } from "@/lib/api";

interface ExcelToolsPanelProps {
  analysis: ExcelAnalysisResponse | null;
  canEdit: boolean;
  reanalyzing: boolean;
  onReanalyze: () => void;
  className?: string;
}

export function ExcelToolsPanel({
  analysis,
  canEdit,
  reanalyzing,
  onReanalyze,
  className,
}: ExcelToolsPanelProps) {
  return (
    <aside
      className={cn(
        "flex w-full shrink-0 flex-col rounded-xl border border-border bg-card lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:w-56 xl:w-60",
        className,
      )}
      data-tour="excel-tools"
    >
      <div className="border-b border-border px-3 py-2.5">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Spreadsheet tools
        </p>
      </div>
      <div className="space-y-3 p-3">
        {analysis ? (
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Rows" value={analysis.profile.row_count} />
            <Stat label="Columns" value={analysis.profile.column_count} />
            <Stat label="Fields" value={analysis.profile.columns.length} />
            <Stat label="Charts" value={analysis.charts.length} />
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Run analysis to see row, column, and chart stats.
          </p>
        )}
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
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-2 text-center">
      <p className="text-base font-semibold tabular-nums">{value}</p>
      <p className="text-[10px] text-muted-foreground">{label}</p>
    </div>
  );
}
