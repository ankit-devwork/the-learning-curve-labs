"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type DocumentSummary, type WorkspaceSummary } from "@/lib/api";
import { MultiDocChatPanel } from "@/components/documents/multi-doc-chat-panel";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

const selectClassName = cn(
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
  "ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
);

type CompareWorkspaceClientProps = {
  initialSetId?: string;
};

export function CompareWorkspaceClient({ initialSetId }: CompareWorkspaceClientProps) {
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [selectedSetId, setSelectedSetId] = useState(initialSetId ?? "");
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const loadWorkspaces = useCallback(async () => {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setLoading(false);
      return;
    }
    const response = await apiFetch("/workspaces", session.access_token);
    if (response.ok) {
      const data = await response.json();
      const items = (data.workspaces ?? []) as WorkspaceSummary[];
      setWorkspaces(items);
      setSelectedSetId((current) => current || initialSetId || items[0]?.id || "");
    }
    setLoading(false);
  }, [initialSetId]);

  const loadDocuments = useCallback(async (setId: string) => {
    if (!setId) {
      setDocuments([]);
      return;
    }
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }
    const response = await apiFetch(`/workspaces/${setId}/documents`, session.access_token);
    if (response.ok) {
      const data = await response.json();
      setDocuments(data.documents ?? []);
    }
  }, []);

  useEffect(() => {
    void loadWorkspaces();
  }, [loadWorkspaces]);

  useEffect(() => {
    if (selectedSetId) {
      void loadDocuments(selectedSetId);
    }
  }, [loadDocuments, selectedSetId]);

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading study sets…</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Compare documents</h1>
        <p className="mt-1 text-muted-foreground">
          Pick a study set, then ask one question across two or more ready documents in that set.
        </p>
      </div>

      {workspaces.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Create a study set first on the{" "}
          <Link href="/dashboard/sets" className="text-primary hover:underline">
            Study sets
          </Link>{" "}
          page.
        </p>
      ) : (
        <>
          <div className="max-w-md space-y-2">
            <Label htmlFor="compare-set">Study set</Label>
            <select
              id="compare-set"
              className={selectClassName}
              value={selectedSetId}
              onChange={(event) => setSelectedSetId(event.target.value)}
            >
              {workspaces.map((workspace) => (
                <option key={workspace.id} value={workspace.id}>
                  {workspace.name}
                </option>
              ))}
            </select>
          </div>

          <MultiDocChatPanel documents={documents} workspaceId={selectedSetId || undefined} />
        </>
      )}
    </div>
  );
}
