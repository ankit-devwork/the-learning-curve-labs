"use client";

import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";

const selectClassName = cn(
  "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
);

type Member = {
  user_id: string;
  role: string;
  email?: string | null;
  full_name?: string | null;
};

type Invite = {
  id: string;
  email: string;
  role: string;
  expires_at: string;
};

export function ShareWorkspacePanel({
  setId,
  canManage,
}: {
  setId: string;
  canManage: boolean;
}) {
  const { toast } = useToast();
  const [members, setMembers] = useState<Member[]>([]);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"viewer" | "editor">("viewer");
  const [loading, setLoading] = useState(true);
  const [lastInviteLink, setLastInviteLink] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setLoading(false);
      return;
    }
    const [membersRes, invitesRes] = await Promise.all([
      apiFetch(`/workspaces/${setId}/members`, session.access_token),
      canManage ? apiFetch(`/workspaces/${setId}/invites`, session.access_token) : Promise.resolve(null),
    ]);
    if (membersRes.ok) {
      const data = await membersRes.json();
      setMembers(data.members ?? []);
    }
    if (invitesRes?.ok) {
      const data = await invitesRes.json();
      setInvites(data.invites ?? []);
    }
    setLoading(false);
  }, [canManage, setId]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  async function handleInvite(event: React.FormEvent) {
    event.preventDefault();
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }
    const response = await apiFetch(`/workspaces/${setId}/invites`, session.access_token, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, role }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      toast({ title: "Invite failed", description: body.error, variant: "error" });
      return;
    }
    const data = await response.json();
    const link = `${window.location.origin}/invite/${data.token}`;
    setLastInviteLink(link);
    setEmail("");
    toast({ title: "Invite created", description: "Copy the link below to share.", variant: "success" });
    await loadAll();
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading sharing settings…</p>;
  }

  return (
    <Card className="shadow-sm" data-tour="share-panel">
      <CardHeader>
        <CardTitle>Share study set</CardTitle>
        <CardDescription>
          Invite classmates as viewers (read + study) or editors (upload + generate tools).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="mb-2 text-sm font-medium">Members</p>
          <ul className="space-y-2 text-sm">
            {members.map((member) => (
              <li key={member.user_id} className="flex items-center justify-between rounded-md border px-3 py-2">
                <span>{member.email || member.full_name || member.user_id}</span>
                <span className="text-xs capitalize text-muted-foreground">{member.role}</span>
              </li>
            ))}
          </ul>
        </div>

        {canManage ? (
          <>
            {invites.length > 0 ? (
              <div>
                <p className="mb-2 text-sm font-medium">Pending invites</p>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  {invites.map((invite) => (
                    <li key={invite.id}>
                      {invite.email} · {invite.role}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <form onSubmit={handleInvite} className="space-y-3">
              <div className="space-y-2">
                <Label htmlFor="invite-email">Email</Label>
                <Input
                  id="invite-email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="classmate@school.edu"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="invite-role">Role</Label>
                <select
                  id="invite-role"
                  className={selectClassName}
                  value={role}
                  onChange={(event) => setRole(event.target.value as "viewer" | "editor")}
                >
                  <option value="viewer">Viewer</option>
                  <option value="editor">Editor</option>
                </select>
              </div>
              <Button type="submit">Create invite link</Button>
            </form>

            {lastInviteLink ? (
              <div className="rounded-md border bg-muted/30 p-3 text-xs">
                <p className="font-medium">Invite link</p>
                <p className="mt-1 break-all text-muted-foreground">{lastInviteLink}</p>
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Only owners and editors can invite collaborators.</p>
        )}
      </CardContent>
    </Card>
  );
}
