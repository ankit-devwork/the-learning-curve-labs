"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type TeamChatMessage, type WorkspaceMessagesResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ProcessingContentSkeleton } from "@/components/ui/loading-skeletons";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 12_000;
const PLAIN_TEXT_PATTERN = /^[A-Za-z0-9\s.,?!'":;\-()]*$/;

type TeamChatPanelProps = {
  setId: string;
  accessToken: string | null;
  isOwner: boolean;
};

function formatMessageTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function validateClientMessage(body: string): string | null {
  const trimmed = body.trim();
  if (!trimmed) {
    return "Enter a message.";
  }
  if (trimmed.length > 2000) {
    return "Message is too long (max 2000 characters).";
  }
  if (!PLAIN_TEXT_PATTERN.test(trimmed)) {
    return "Use plain English letters, numbers, and basic punctuation only.";
  }
  if (/https?:\/\/|www\./i.test(trimmed)) {
    return "Links are not allowed in team chat.";
  }
  return null;
}

export function TeamChatPanel({ setId, accessToken, isOwner }: TeamChatPanelProps) {
  const [messages, setMessages] = useState<TeamChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const scrollToBottom = useCallback(() => {
    const container = scrollRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, []);

  const loadMessages = useCallback(
    async (options?: { silent?: boolean }) => {
      if (!accessToken) {
        return;
      }
      if (!options?.silent) {
        setLoading(true);
      }
      setError(null);
      const response = await apiFetch(`/workspaces/${setId}/messages`, accessToken);
      if (!options?.silent) {
        setLoading(false);
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        if (response.status === 403) {
          setError("You do not have access to this study sheet chat.");
        } else {
          setError(body.error || `Could not load team chat (${response.status})`);
        }
        return;
      }
      const payload = (await response.json()) as WorkspaceMessagesResponse;
      setMessages(payload.messages ?? []);
    },
    [accessToken, setId],
  );

  useEffect(() => {
    async function loadUser() {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      setCurrentUserId(session?.user?.id ?? null);
    }
    void loadUser();
  }, []);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    void loadMessages();
    pollRef.current = setInterval(() => {
      void loadMessages({ silent: true });
    }, POLL_INTERVAL_MS);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
    };
  }, [accessToken, loadMessages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  async function handleSend(event: React.FormEvent) {
    event.preventDefault();
    if (!accessToken) {
      return;
    }
    const validationError = validateClientMessage(draft);
    if (validationError) {
      setError(validationError);
      return;
    }

    setSending(true);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/messages`, accessToken, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: draft.trim() }),
    });
    setSending(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Could not send message (${response.status})`);
      return;
    }
    setDraft("");
    await loadMessages({ silent: true });
  }

  async function handleDelete(messageId: string) {
    if (!accessToken) {
      return;
    }
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/messages/${messageId}`, accessToken, {
      method: "DELETE",
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(body.error || `Could not delete message (${response.status})`);
      return;
    }
    await loadMessages({ silent: true });
  }

  const canDeleteMessage = useCallback(
    (message: TeamChatMessage) => {
      if (message.is_own || message.author_id === currentUserId) {
        return true;
      }
      return isOwner;
    },
    [currentUserId, isOwner],
  );

  const emptyState = !loading && messages.length === 0;

  return (
    <Card className="shadow-sm" data-tour="team-chat">
      <CardHeader>
        <CardTitle>Team chat</CardTitle>
        <CardDescription>
          Plain-text discussion for study sheet members only. No AI, links, files, or emoji.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        <div
          ref={scrollRef}
          className="max-h-80 space-y-3 overflow-y-auto rounded-lg border bg-muted/20 p-3"
          aria-live="polite"
        >
          {loading ? <ProcessingContentSkeleton lines={3} /> : null}
          {emptyState ? (
            <p className="text-sm text-muted-foreground">
              No messages yet. Invite teammates from Share, then coordinate here.
            </p>
          ) : null}
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex flex-col gap-1 rounded-lg px-3 py-2 text-sm",
                message.is_own || message.author_id === currentUserId
                  ? "ml-8 bg-primary/10"
                  : "mr-8 bg-card ring-1 ring-border/60",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-medium text-foreground">
                  {message.is_own || message.author_id === currentUserId ? "You" : message.author_name}
                </p>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground">{formatMessageTime(message.created_at)}</span>
                  {canDeleteMessage(message) ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-[10px] text-muted-foreground"
                      onClick={() => void handleDelete(message.id)}
                    >
                      Delete
                    </Button>
                  ) : null}
                </div>
              </div>
              <p className="whitespace-pre-wrap leading-relaxed">{message.body}</p>
            </div>
          ))}
        </div>

        <form onSubmit={(event) => void handleSend(event)} className="flex gap-2">
          <Input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Type a plain English message for your team..."
            disabled={!accessToken || sending}
            maxLength={2000}
            aria-label="Team chat message"
          />
          <Button type="submit" disabled={!accessToken || sending || !draft.trim()}>
            {sending ? "Sending..." : "Send"}
          </Button>
        </form>
        <p className="text-[11px] text-muted-foreground">
          Messages refresh every 12 seconds. Only members of this study sheet can read or post.
        </p>
      </CardContent>
    </Card>
  );
}
