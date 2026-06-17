"use client";

import { useMemo, useState } from "react";
import { apiFetch, type CustomChartRequest, type ExcelChart } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ExcelChartView } from "@/components/excel/excel-chart-view";
import { cn } from "@/lib/utils";

type ProfileColumn = {
  name: string;
  dtype: string;
  null_pct: number;
  unique_count: number;
};

const CHART_TYPES: Array<{ value: CustomChartRequest["chart_type"]; label: string }> = [
  { value: "bar", label: "Bar" },
  { value: "line", label: "Line" },
  { value: "pie", label: "Pie" },
  { value: "scatter", label: "Scatter" },
];

const AGGREGATIONS: Array<{ value: NonNullable<CustomChartRequest["aggregation"]>; label: string }> =
  [
    { value: "sum", label: "Sum" },
    { value: "mean", label: "Average" },
    { value: "count", label: "Count" },
  ];

const selectClassName = cn(
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
  "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
  "disabled:cursor-not-allowed disabled:opacity-50",
);

function isNumericDtype(dtype: string): boolean {
  return dtype.includes("int") || dtype.includes("float") || dtype === "number";
}

type ExcelChartBuilderProps = {
  documentId: string;
  columns: ProfileColumn[];
  accessToken: string;
  onChartCreated: (chart: ExcelChart) => void;
};

export function ExcelChartBuilder({
  documentId,
  columns,
  accessToken,
  onChartCreated,
}: ExcelChartBuilderProps) {
  const [chartType, setChartType] = useState<CustomChartRequest["chart_type"]>("bar");
  const [xColumn, setXColumn] = useState(columns[0]?.name ?? "");
  const [yColumn, setYColumn] = useState(
    columns.find((column) => column.name !== columns[0]?.name)?.name ?? "",
  );
  const [aggregation, setAggregation] = useState<NonNullable<CustomChartRequest["aggregation"]>>("sum");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<ExcelChart | null>(null);

  const yRequired = chartType === "bar" || chartType === "line" || chartType === "scatter";
  const showAggregation = chartType !== "scatter" && (yRequired || chartType === "pie");

  const xColumnMeta = columns.find((column) => column.name === xColumn);
  const yColumnMeta = columns.find((column) => column.name === yColumn);

  const yOptions = useMemo(
    () => columns.filter((column) => column.name !== xColumn),
    [columns, xColumn],
  );

  async function handleGenerate() {
    if (!xColumn) {
      setError("Select an X column");
      return;
    }
    if (yRequired && !yColumn) {
      setError("Select a Y column for this chart type");
      return;
    }

    setLoading(true);
    setError(null);

    const body: CustomChartRequest = {
      chart_type: chartType,
      x_column: xColumn,
      aggregation,
    };
    if (yRequired || (chartType === "pie" && yColumn)) {
      body.y_column = yColumn;
    }
    if (title.trim()) {
      body.title = title.trim();
    }

    const response = await apiFetch(`/documents/${documentId}/charts/custom`, accessToken, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    setLoading(false);

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      setError(payload.error || payload.detail || `Failed to build chart (${response.status})`);
      return;
    }

    const data = await response.json();
    setPreview(data.chart);
    onChartCreated(data.chart);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Build custom chart</CardTitle>
        <CardDescription>
          Pick columns and chart type to explore the spreadsheet your way.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="chart-type">Chart type</Label>
            <select
              id="chart-type"
              className={selectClassName}
              value={chartType}
              onChange={(event) =>
                setChartType(event.target.value as CustomChartRequest["chart_type"])
              }
            >
              {CHART_TYPES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="x-column">X column</Label>
            <select
              id="x-column"
              className={selectClassName}
              value={xColumn}
              onChange={(event) => {
                setXColumn(event.target.value);
                if (event.target.value === yColumn) {
                  const fallback = columns.find((column) => column.name !== event.target.value);
                  setYColumn(fallback?.name ?? "");
                }
              }}
            >
              {columns.map((column) => (
                <option key={column.name} value={column.name}>
                  {column.name} ({column.dtype})
                </option>
              ))}
            </select>
            {xColumnMeta && (
              <p className="text-xs text-muted-foreground">
                {xColumnMeta.unique_count} unique · {xColumnMeta.null_pct}% null
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="y-column">
              Y column{yRequired ? "" : " (optional)"}
            </Label>
            <select
              id="y-column"
              className={selectClassName}
              value={yColumn}
              disabled={!yRequired && chartType !== "pie"}
              onChange={(event) => setYColumn(event.target.value)}
            >
              {!yRequired && chartType === "pie" && <option value="">Count by category</option>}
              {yOptions.map((column) => (
                <option key={column.name} value={column.name}>
                  {column.name} ({column.dtype})
                </option>
              ))}
            </select>
            {yColumnMeta && (
              <p className="text-xs text-muted-foreground">
                {isNumericDtype(yColumnMeta.dtype) ? "Numeric" : "Text"} · {yColumnMeta.null_pct}% null
              </p>
            )}
            {chartType === "scatter" && xColumnMeta && !isNumericDtype(xColumnMeta.dtype) && (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Scatter works best when X is numeric (e.g. ShippingCost).
              </p>
            )}
          </div>

          {showAggregation && (
            <div className="space-y-2">
              <Label htmlFor="aggregation">Aggregation</Label>
              <select
                id="aggregation"
                className={selectClassName}
                value={aggregation}
                onChange={(event) =>
                  setAggregation(event.target.value as NonNullable<CustomChartRequest["aggregation"]>)
                }
              >
                {AGGREGATIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="space-y-2 sm:col-span-2">
            <Label htmlFor="chart-title">Title (optional)</Label>
            <Input
              id="chart-title"
              placeholder="Auto-generated if left blank"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
          </div>
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}

        <Button type="button" onClick={() => void handleGenerate()} disabled={loading || columns.length === 0}>
          {loading ? "Building..." : "Generate chart"}
        </Button>

        {preview && (
          <div className="space-y-3 rounded-lg border border-border/60 bg-muted/20 p-4">
            <div>
              <p className="font-medium">{preview.title}</p>
              <p className="text-sm text-muted-foreground">
                {preview.chart_type} · {preview.x_column}
                {preview.y_column ? ` vs ${preview.y_column}` : ""}
              </p>
            </div>
            <ExcelChartView chart={preview} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
