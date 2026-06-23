"use client";

import type { ExcelPreviewResponse } from "@/lib/api";

type ExcelPreviewTableProps = {
  preview: ExcelPreviewResponse | null;
  loading?: boolean;
};

export function ExcelPreviewTable({ preview, loading }: ExcelPreviewTableProps) {
  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading data preview…</p>;
  }

  if (!preview) {
    return (
      <p className="text-sm text-muted-foreground">
        Analyze the spreadsheet to preview rows here.
      </p>
    );
  }

  return (
    <div className="space-y-2" data-tour="excel-preview">
      <p className="text-xs text-muted-foreground">
        Showing {preview.preview_rows} of {preview.total_rows} rows · {preview.total_columns}{" "}
        columns
      </p>
      <div className="max-h-[420px] overflow-auto rounded-lg border">
        <table className="min-w-full text-left text-xs">
          <thead className="sticky top-0 bg-muted/80 backdrop-blur">
            <tr>
              {preview.columns.map((column) => (
                <th key={column} className="whitespace-nowrap px-3 py-2 font-medium">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t">
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="whitespace-nowrap px-3 py-2 text-muted-foreground">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
