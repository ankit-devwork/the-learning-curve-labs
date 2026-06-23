"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, getApiUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function InviteAcceptClient({ token }: { token: string }) {
  const [preview, setPreview] = useState<{
    workspace_name?: string | null;
    email?: string;
    role?: string;
    accepted_at?: string | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [accepting, setAccepting] = useState(false);
  const [acceptedWorkspaceId, setAcceptedWorkspaceId] = useState<string | null>(null);

  useEffect(() => {
    async function loadPreview() {
      const response = await fetch(getApiUrl(`/invites/${token}`));
      if (!response.ok) {
        setError("Invite not found or expired");
        return;
      }
      setPreview(await response.json());
    }
    void loadPreview();
  }, [token]);

  async function handleAccept() {
    setAccepting(true);
    setError(null);
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setError("Sign in to accept this invite");
      setAccepting(false);
      return;
    }
    const response = await apiFetch(`/invites/${token}/accept`, session.access_token, {
      method: "POST",
    });
    setAccepting(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || "Could not accept invite");
      return;
    }
    const data = await response.json();
    setAcceptedWorkspaceId(data.workspace?.id ?? null);
  }

  return (
    <Card className="mx-auto max-w-lg shadow-sm">
      <CardHeader>
        <CardTitle>Study set invite</CardTitle>
        <CardDescription>
          {preview?.workspace_name
            ? `You’ve been invited to join “${preview.workspace_name}”`
            : "Loading invite…"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {preview?.email ? (
          <p className="text-sm text-muted-foreground">
            For {preview.email} · role: {preview.role}
          </p>
        ) : null}
        {error ? <p className="text-sm text-destructive">{error}</p> : null}
        {acceptedWorkspaceId ? (
          <Link href={`/dashboard/sets/${acceptedWorkspaceId}`}>
            <Button type="button">Open study set</Button>
          </Link>
        ) : (
          <Button type="button" disabled={accepting} onClick={() => void handleAccept()}>
            {accepting ? "Joining…" : "Accept invite"}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
