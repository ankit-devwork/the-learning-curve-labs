"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { Plus } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type WorkspaceSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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
      toast({ title: "Study set created", variant: "success" });
      await loadWorkspaces();
    } finally {
      setCreating(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading study sets…</p>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Study sets</h1>
        <p className="mt-1 text-muted-foreground">
          Organize course materials by class, exam, or project — then upload, chat, quiz, and compare.
        </p>
      </div>

      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg">Create a study set</CardTitle>
          <CardDescription>For example: Biology 101, Midterm prep, or Q3 sales data.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleCreate} className="flex flex-wrap gap-2" data-tour="create-set">
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Study set name"
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

      <div className="grid gap-3 sm:grid-cols-2" data-tour="sets-list">
        {workspaces.map((workspace) => (
          <Link
            key={workspace.id}
            href={`/dashboard/sets/${workspace.id}`}
            className="rounded-xl border bg-card p-5 shadow-sm transition-shadow hover:shadow-md"
          >
            <p className="font-medium">{workspace.name}</p>
            {workspace.description ? (
              <p className="mt-1 text-sm text-muted-foreground">{workspace.description}</p>
            ) : (
              <p className="mt-1 text-sm text-muted-foreground">Open files, upload, and compare documents</p>
            )}
          </Link>
        ))}
      </div>
    </div>
  );
}
