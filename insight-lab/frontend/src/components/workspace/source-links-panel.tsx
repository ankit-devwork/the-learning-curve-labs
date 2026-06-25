"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, type DocumentSummary, type SourceLink } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

type SourceLinksPanelProps = {
  setId: string;
  documents: DocumentSummary[];
  accessToken: string | null;
  canEdit: boolean;
};

export function SourceLinksPanel({
  setId,
  documents,
  accessToken,
  canEdit,
}: SourceLinksPanelProps) {
  const [links, setLinks] = useState<SourceLink[]>([]);
  const [excelId, setExcelId] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);

  const spreadsheetDocs = documents.filter((doc) => doc.file_type === "excel");
  const excelDocs = spreadsheetDocs.filter((doc) => doc.status === "ready");
  const textDocs = documents.filter((doc) => doc.file_type === "document" && doc.status === "ready");
  const canLink = excelDocs.length > 0 && textDocs.length > 0;

  const loadLinks = useCallback(async () => {
    if (!accessToken) {
      return;
    }
    const response = await apiFetch(`/workspaces/${setId}/source-links`, accessToken);
    if (response.ok) {
      const data = await response.json();
      setLinks(data.links ?? []);
    }
  }, [accessToken, setId]);

  useEffect(() => {
    void loadLinks();
  }, [loadLinks]);

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    if (!accessToken || !excelId || !documentId) {
      return;
    }
    setBusy(true);
    const response = await apiFetch(`/workspaces/${setId}/source-links`, accessToken, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        excel_document_id: excelId,
        document_id: documentId,
        label: label.trim() || undefined,
      }),
    });
    setBusy(false);
    if (response.ok) {
      setLabel("");
      await loadLinks();
    }
  }

  async function handleDelete(linkId: string) {
    if (!accessToken) {
      return;
    }
    await apiFetch(`/workspaces/${setId}/source-links/${linkId}`, accessToken, { method: "DELETE" });
    await loadLinks();
  }

  if (spreadsheetDocs.length === 0) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-muted-foreground">
          This study sheet has no spreadsheets. Source links are only used when you combine Excel data with
          related PDF or Word notes so spreadsheet chat can cite document excerpts.
        </p>
        <p className="text-sm text-muted-foreground">
          For PDF-only sheets, open each file from <strong className="font-medium text-foreground">Materials</strong>{" "}
          and use chat and quiz there — nothing to set up here.
        </p>
      </div>
    );
  }

  if (!canLink) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">
          Connect a ready spreadsheet to a related PDF or Word file so Excel Q&amp;A can cite document excerpts
          alongside your data.
        </p>
        {excelDocs.length === 0 && textDocs.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Wait until at least one spreadsheet and one PDF/Word file show status <strong className="font-medium text-foreground">Ready</strong>, then return here to create links.
          </p>
        ) : null}
        {excelDocs.length === 0 && textDocs.length > 0 ? (
          <p className="text-sm text-muted-foreground">
            Your PDF/Word files are ready. Spreadsheet analysis is still in progress — links unlock when a
            spreadsheet shows Ready.
          </p>
        ) : null}
        {excelDocs.length > 0 && textDocs.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Your spreadsheet is ready. Upload and process at least one PDF or Word document in this sheet to
            link with it. Excel chat works without links, but document citations need a related file.
          </p>
        ) : null}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Connect spreadsheets to related PDFs or Word docs so Excel Q&amp;A can pull document citations.
      </p>

      {links.length > 0 ? (
        <ul className="space-y-2 text-sm">
          {links.map((link) => (
            <li key={link.id} className="flex items-center justify-between gap-2 rounded-md border px-3 py-2">
              <span className="min-w-0 truncate">
                {link.excel_filename} ↔ {link.document_filename}
                {link.label ? ` · ${link.label}` : ""}
              </span>
              {canEdit ? (
                <Button type="button" variant="ghost" size="sm" onClick={() => void handleDelete(link.id)}>
                  Remove
                </Button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">No links yet.</p>
      )}

      {canEdit ? (
        <form onSubmit={handleCreate} className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="link-excel">Spreadsheet</Label>
            <select
              id="link-excel"
              className="flex h-10 w-full rounded-md border bg-background px-3 text-sm"
              value={excelId}
              onChange={(event) => setExcelId(event.target.value)}
            >
              <option value="">Select…</option>
              {excelDocs.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.filename}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label htmlFor="link-doc">Document</Label>
            <select
              id="link-doc"
              className="flex h-10 w-full rounded-md border bg-background px-3 text-sm"
              value={documentId}
              onChange={(event) => setDocumentId(event.target.value)}
            >
              <option value="">Select…</option>
              {textDocs.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.filename}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <Label htmlFor="link-label">Label (optional)</Label>
            <input
              id="link-label"
              className="flex h-10 w-full rounded-md border bg-background px-3 text-sm"
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              placeholder="e.g. Lab data ↔ theory notes"
            />
          </div>
          <Button type="submit" disabled={busy || !excelId || !documentId}>
            {busy ? "Linking…" : "Create link"}
          </Button>
        </form>
      ) : null}
    </div>
  );
}

/** Show Source links UI when the sheet has spreadsheets or existing links. */
export function shouldShowSourceLinks(documents: DocumentSummary[], linkCount: number): boolean {
  const hasSpreadsheets = documents.some((doc) => doc.file_type === "excel");
  return hasSpreadsheets || linkCount > 0;
}
