"use client";

import { Download, Image as ImageIcon } from "lucide-react";
import { useRef } from "react";
import type { ExcelChart } from "@/lib/api";
import { downloadChartCsv, downloadChartPng } from "@/lib/export-utils";
import { Button } from "@/components/ui/button";

type ChartExportActionsProps = {
  chart: ExcelChart;
};

export function ChartExportActions({ chart }: ChartExportActionsProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="gap-2"
          onClick={() => downloadChartCsv(chart)}
        >
          <Download className="h-4 w-4" aria-hidden />
          CSV
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="gap-2"
          onClick={() => void downloadChartPng(chart, containerRef.current)}
        >
          <ImageIcon className="h-4 w-4" aria-hidden />
          PNG
        </Button>
      </div>
      <div ref={containerRef}>
        {/* Chart content rendered by parent */}
      </div>
    </div>
  );
}

export function ChartCardWithExport({
  chart,
  children,
}: {
  chart: ExcelChart;
  children: React.ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="gap-2"
          onClick={() => downloadChartCsv(chart)}
        >
          <Download className="h-4 w-4" aria-hidden />
          Export CSV
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="gap-2"
          onClick={() => void downloadChartPng(chart, containerRef.current)}
        >
          <ImageIcon className="h-4 w-4" aria-hidden />
          Export PNG
        </Button>
      </div>
      <div ref={containerRef}>{children}</div>
    </div>
  );
}
