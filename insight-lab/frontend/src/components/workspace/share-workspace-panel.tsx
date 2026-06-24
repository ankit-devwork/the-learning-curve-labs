"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, generateTrackingId, getTrackingIdFromResponse, parseApiError } from "@/lib/api";
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

function inviteLinkForToken(token: string): string {
  if (typeof window === "undefined") {
    return `/invite/${token}`;
  }
  return `${window.location.origin}/invite/${token}`;
}

async function copyInviteLink(link: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(link);
    return true;
  } catch {
    return false;
  }
}

export function ShareWorkspacePanel({
  setId,
  canManage,
  isOwner = false,
  embedded = false,
}: {
  setId: string;
  canManage: boolean;
  isOwner?: boolean;
  embedded?: boolean;
}) {
  const router = useRouter();
  const { toast } = useToast();
  const [members, setMembers] = useState<Member[]>([]);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"viewer" | "editor">("viewer");
  const [loading, setLoading] = useState(true);
  const [lastInviteLink, setLastInviteLink] = useState<string | null>(null);
  const [removingMemberId, setRemovingMemberId] = useState<string | null>(null);
  const [revokingInviteId, setRevokingInviteId] = useState<string | null>(null);
  const [updatingRoleMemberId, setUpdatingRoleMemberId] = useState<string | null>(null);
  const [leaving, setLeaving] = useState(false);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);

  const currentMember = members.find((member) => member.user_id === currentUserId);
  const canLeave = Boolean(currentMember && currentMember.role !== "owner");

  const loadAll = useCallback(async (trackingId?: string) => {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      setLoading(false);
      return;
    }
    setCurrentUserId(session.user.id);
    const fetchOpts = trackingId ? { trackingId } : undefined;
    const [membersRes, invitesRes] = await Promise.all([
      apiFetch(`/workspaces/${setId}/members`, session.access_token, fetchOpts),
      canManage
        ? apiFetch(`/workspaces/${setId}/invites`, session.access_token, fetchOpts)
        : Promise.resolve(null),
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
    const trackingId = generateTrackingId();
    const response = await apiFetch(`/workspaces/${setId}/invites`, session.access_token, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, role }),
      trackingId,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      toast({
        title: "Invite failed",
        description: parseApiError(
          body,
          "Could not create invite",
          getTrackingIdFromResponse(response) ?? trackingId,
        ),
        variant: "error",
      });
      return;
    }
    const data = await response.json();
    if (!data.token) {
      toast({
        title: "Invite failed",
        description: "Server did not return an invite link — try again or contact support.",
        variant: "error",
      });
      return;
    }
    const invitedEmail = email.trim();
    const link = inviteLinkForToken(data.token);
    setLastInviteLink(link);
    setEmail("");
    toast({
      title: data.email_sent ? "Invite email sent" : "Invite link created",
      description: data.email_sent
        ? `We emailed ${invitedEmail} — you can also copy the link below.`
        : "Email is off — copy the link and send it yourself.",
      variant: "success",
    });
    await loadAll(trackingId);
  }

  async function handleCopyPendingInvite(invite: Invite) {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }
    const response = await apiFetch(
      `/workspaces/${setId}/invites/${invite.id}/link`,
      session.access_token,
    );
    if (!response.ok) {
      toast({ title: "Could not load invite link", variant: "error" });
      return;
    }
    const data = await response.json();
    await handleCopyLink(data.url as string);
  }

  async function handleCopyLink(link: string) {
    const copied = await copyInviteLink(link);
    toast({
      title: copied ? "Link copied" : "Could not copy",
      description: copied ? "Paste it in email, chat, or text to your classmate." : "Select and copy the link manually.",
      variant: copied ? "success" : "error",
    });
  }

  async function handleRemoveMember(member: Member) {
    if (!isOwner || member.role === "owner") {
      return;
    }
    const label = member.email || member.full_name || "this member";
    if (!window.confirm(`Remove ${label} from this study sheet? They will lose access immediately.`)) {
      return;
    }

    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }

    setRemovingMemberId(member.user_id);
    const trackingId = generateTrackingId();
    try {
      const response = await apiFetch(
        `/workspaces/${setId}/members/${member.user_id}`,
        session.access_token,
        { method: "DELETE", trackingId },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Could not remove member",
          description: body.error || body.detail,
          variant: "error",
        });
        return;
      }
      toast({ title: "Member removed", variant: "success" });
      await loadAll(trackingId);
    } finally {
      setRemovingMemberId(null);
    }
  }

  async function handleRoleChange(member: Member, nextRole: "viewer" | "editor") {
    if (!isOwner || member.role === "owner" || member.role === nextRole) {
      return;
    }

    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }

    setUpdatingRoleMemberId(member.user_id);
    const trackingId = generateTrackingId();
    try {
      const response = await apiFetch(
        `/workspaces/${setId}/members/${member.user_id}`,
        session.access_token,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role: nextRole }),
          trackingId,
        },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Could not update role",
          description: body.error || body.detail,
          variant: "error",
        });
        return;
      }
      toast({ title: "Role updated", variant: "success" });
      await loadAll(trackingId);
    } finally {
      setUpdatingRoleMemberId(null);
    }
  }

  async function handleRevokeInvite(invite: Invite) {
    if (!canManage) {
      return;
    }
    if (!window.confirm(`Revoke the invite for ${invite.email}? The link will stop working.`)) {
      return;
    }

    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }

    setRevokingInviteId(invite.id);
    const trackingId = generateTrackingId();
    try {
      const response = await apiFetch(
        `/workspaces/${setId}/invites/${invite.id}`,
        session.access_token,
        { method: "DELETE", trackingId },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Could not revoke invite",
          description: body.error || body.detail,
          variant: "error",
        });
        return;
      }
      toast({ title: "Invite revoked", variant: "success" });
      await loadAll(trackingId);
    } finally {
      setRevokingInviteId(null);
    }
  }

  async function handleLeaveWorkspace() {
    if (!canLeave) {
      return;
    }
    if (
      !window.confirm(
        "Leave this study sheet? You will lose access to its files and shared progress.",
      )
    ) {
      return;
    }

    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (!session?.access_token) {
      return;
    }

    setLeaving(true);
    const trackingId = generateTrackingId();
    try {
      const response = await apiFetch(`/workspaces/${setId}/leave`, session.access_token, {
        method: "POST",
        trackingId,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast({
          title: "Could not leave study sheet",
          description: body.error || body.detail,
          variant: "error",
        });
        return;
      }
      toast({ title: "Left study sheet", variant: "success" });
      router.push("/dashboard/sets");
    } finally {
      setLeaving(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading sharing settings…</p>;
  }

  const content = (
    <div className="space-y-4">
      <div data-tour="share-members">
          <p className="mb-2 text-sm font-medium">Members</p>
          <ul className="space-y-2 text-sm">
            {members.map((member) => (
              <li
                key={member.user_id}
                className="flex items-center justify-between gap-3 rounded-md border px-3 py-2"
              >
                <div className="min-w-0">
                  <p className="truncate">{member.email || member.full_name || member.user_id}</p>
                  {member.user_id === currentUserId ? (
                    <p className="text-xs text-muted-foreground">You</p>
                  ) : null}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {isOwner && member.role !== "owner" && member.user_id !== currentUserId ? (
                    <select
                      className={cn(selectClassName, "h-8 w-auto min-w-[6.5rem] text-xs capitalize")}
                      value={member.role}
                      disabled={updatingRoleMemberId === member.user_id}
                      onChange={(event) =>
                        void handleRoleChange(member, event.target.value as "viewer" | "editor")
                      }
                      aria-label={`Role for ${member.email || member.full_name || "member"}`}
                    >
                      <option value="viewer">Viewer</option>
                      <option value="editor">Editor</option>
                    </select>
                  ) : (
                    <span className="text-xs capitalize text-muted-foreground">{member.role}</span>
                  )}
                  {isOwner && member.role !== "owner" && member.user_id !== currentUserId ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-8 text-destructive hover:text-destructive"
                      disabled={removingMemberId === member.user_id}
                      onClick={() => void handleRemoveMember(member)}
                    >
                      {removingMemberId === member.user_id ? "Removing…" : "Remove"}
                    </Button>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
          {isOwner ? (
            <p className="mt-2 text-xs text-muted-foreground">
              As owner, you can change roles or remove editors and viewers.
            </p>
          ) : null}
        </div>

        {canLeave ? (
          <div data-tour="share-leave">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="text-destructive hover:text-destructive"
              disabled={leaving}
              onClick={() => void handleLeaveWorkspace()}
            >
              {leaving ? "Leaving…" : "Leave study sheet"}
            </Button>
          </div>
        ) : null}

        {canManage ? (
          <>
            {invites.length > 0 ? (
              <div data-tour="share-invites">
                <p className="mb-2 text-sm font-medium">Pending invites</p>
                <ul className="space-y-2 text-sm">
                  {invites.map((invite) => (
                    <li
                      key={invite.id}
                      className="flex flex-col gap-2 rounded-md border px-3 py-2 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <span className="min-w-0 truncate text-muted-foreground">
                        {invite.email} · {invite.role}
                      </span>
                      <div className="flex shrink-0 items-center gap-2">
                        {invite.id ? (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="h-8"
                            onClick={() => void handleCopyPendingInvite(invite)}
                          >
                            Copy link
                          </Button>
                        ) : null}
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-8 text-destructive hover:text-destructive"
                          disabled={revokingInviteId === invite.id}
                          onClick={() => void handleRevokeInvite(invite)}
                        >
                          {revokingInviteId === invite.id ? "Revoking…" : "Revoke"}
                        </Button>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <form onSubmit={handleInvite} className="space-y-3" data-tour="share-invite">
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
                <p className="font-medium">Invite link — send this to your classmate</p>
                <p className="mt-1 break-all text-muted-foreground">{lastInviteLink}</p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="mt-2 h-8"
                  onClick={() => void handleCopyLink(lastInviteLink)}
                >
                  Copy link
                </Button>
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Only owners and editors can invite collaborators.</p>
        )}
    </div>
  );

  if (embedded) {
    return (
      <div data-tour="share-panel">
        {!canManage ? (
          <p className="mb-4 text-sm text-muted-foreground">
            Invite classmates as viewers or editors. When email is configured, we send the invite
            automatically — you can always copy the link below too.
          </p>
        ) : (
          <p className="mb-4 text-sm text-muted-foreground">
            Invite classmates as viewers (read + study) or editors (upload + generate tools). When email
            is configured, we send the invite automatically — you can always copy the link below too.
          </p>
        )}
        {content}
      </div>
    );
  }

  return (
    <Card className="shadow-sm" data-tour="share-panel">
      <CardHeader>
        <CardTitle>Share study sheet</CardTitle>
        <CardDescription>
          Invite classmates as viewers (read + study) or editors (upload + generate tools). When email is
          configured, we send the invite automatically — you can always copy the link too.
        </CardDescription>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  );
}
