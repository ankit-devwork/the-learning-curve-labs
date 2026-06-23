"use client";

import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type DocumentSummary } from "@/lib/api";
import { MultiDocChatPanel } from "@/components/documents/multi-doc-chat-panel";

export function CompareWorkspaceClient() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const loadDocuments = useCallback(async () => {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setLoading(false);
      return;
    }
    const response = await apiFetch("/documents", session.access_token);
    if (response.ok) {
      const data = await response.json();
      setDocuments(data.documents ?? []);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading documents…</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Compare documents</h1>
        <p className="mt-1 text-muted-foreground">
          Select two or more ready documents and ask one question across all of them.
        </p>
      </div>
      <MultiDocChatPanel documents={documents} />
    </div>
  );
}
