"use client";

import { useCallback, useEffect, useState } from "react";
import { Plus } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type WorkspaceSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { StudySetCard } from "@/components/workspace/study-set-card";
import { useToast } from "@/components/ui/toast";

export function StudySetsListClient() {
  const { toast } = useToast();
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

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
      setWorkspaces(data.workspaces ?? []);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void loadWorkspaces();
  }, [loadWorkspaces]);

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      return;
    }
    setCreating(true);
    try {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!session?.access_token) {
        return;
      }
      const response = await apiFetch("/workspaces", session.access_token, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({ title: "Could not create study set", description: body.error, variant: "error" });
        return;
      }
      setName("");
      toast({ title: "Notebook created", variant: "success" });
      await loadWorkspaces();
    } finally {
      setCreating(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading notebooks…</p>;
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl font-semibold tracking-tight">Notebooks</h1>
        <p className="mt-2 max-w-2xl text-muted-foreground">
          Organize sources by class or project — chat, generate study tools, and collaborate like a research notebook.
        </p>
      </div>

      <Card className="notebook-surface border-0 shadow-none">
        <CardHeader>
          <CardTitle className="text-lg">New notebook</CardTitle>
          <CardDescription>For example: Biology 101, MetLaw prep, or Q3 sales data.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleCreate} className="flex flex-wrap gap-2" data-tour="create-set">
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Notebook name"
              maxLength={100}
              className="max-w-sm"
            />
            <Button type="submit" disabled={creating} className="gap-2">
              <Plus className="h-4 w-4" aria-hidden />
              {creating ? "Creating…" : "Create"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3" data-tour="sets-list">
        {workspaces.map((workspace) => (
          <StudySetCard key={workspace.id} workspace={workspace} />
        ))}
      </div>

      {workspaces.length === 0 ? (
        <p className="text-sm text-muted-foreground">Create your first notebook to upload sources and start studying.</p>
      ) : null}
    </div>
  );
}
