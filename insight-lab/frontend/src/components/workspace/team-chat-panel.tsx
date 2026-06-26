"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, MessageCircle, Send, Users } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import {
  apiFetch,
  parseApiError,
  type TeamChatMessage,
  type WorkspaceMessagesResponse,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TeamChatThread } from "@/components/workspace/team-chat-thread";
import { useTeamChatTyping } from "@/hooks/use-team-chat-typing";
import { cn } from "@/lib/utils";

const FALLBACK_POLL_INTERVAL_MS = 30_000;
const PLAIN_TEXT_PATTERN = /^[A-Za-z0-9\s.,?!'":;\-()]*$/;

type TeamChatPanelProps = {
  setId: string;
  accessToken: string | null;
  isOwner: boolean;
  workspaceName?: string;
  embedded?: boolean;
  dock?: boolean;
  onReadStateChange?: () => void;
};

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

export function TeamChatPanel({
  setId,
  accessToken,
  isOwner,
  workspaceName,
  embedded = false,
  dock = false,
  onReadStateChange,
}: TeamChatPanelProps) {
  const [messages, setMessages] = useState<TeamChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [nextBefore, setNextBefore] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [accessDenied, setAccessDenied] = useState(false);
  const [migrationNotice, setMigrationNotice] = useState<string | null>(null);
  const [realtimeConnected, setRealtimeConnected] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const skipAutoScrollRef = useRef(false);
  const lastMarkedMessageIdRef = useRef<string | null>(null);

  const applyPayload = useCallback((payload: WorkspaceMessagesResponse, mode: "replace" | "prepend") => {
    setMigrationNotice(null);
    setHasMore(Boolean(payload.has_more));
    setNextBefore(payload.next_before ?? null);
    setMessages((prev) => {
      if (mode === "prepend") {
        const existing = new Set(prev.map((message) => message.id));
        const older = (payload.messages ?? []).filter((message) => !existing.has(message.id));
        return [...older, ...prev];
      }
      return payload.messages ?? [];
    });
  }, []);

  const fetchMessages = useCallback(
    async (options?: { silent?: boolean; before?: string | null }) => {
      if (!accessToken) {
        return null;
      }
      const isOlder = Boolean(options?.before);
      if (!options?.silent && !isOlder) {
        setLoading(true);
      }
      if (isOlder) {
        setLoadingOlder(true);
        skipAutoScrollRef.current = true;
      }
      setError(null);

      const params = new URLSearchParams();
      if (options?.before) {
        params.set("before", options.before);
      }
      const query = params.toString();
      const response = await apiFetch(
        `/workspaces/${setId}/messages${query ? `?${query}` : ""}`,
        accessToken,
      );

      if (!options?.silent && !isOlder) {
        setLoading(false);
      }
      if (isOlder) {
        setLoadingOlder(false);
      }

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        if (response.status === 403) {
          setAccessDenied(true);
          setError("You do not have access to this study sheet chat.");
        } else if (response.status === 503 && body.notice) {
          setMigrationNotice(body.notice);
          setMessages([]);
        } else {
          setError(parseApiError(body, `Could not load team chat (${response.status})`));
        }
        return null;
      }

      setAccessDenied(false);
      const payload = (await response.json()) as WorkspaceMessagesResponse;
      if (payload.migration_required && payload.notice) {
        setMigrationNotice(payload.notice);
        setMessages([]);
        return null;
      }

      if (isOlder) {
        const container = scrollContainerRef.current;
        const previousHeight = container?.scrollHeight ?? 0;
        applyPayload(payload, "prepend");
        requestAnimationFrame(() => {
          if (container) {
            container.scrollTop = container.scrollHeight - previousHeight;
          }
          skipAutoScrollRef.current = false;
        });
      } else {
        applyPayload(payload, "replace");
      }
      return payload;
    },
    [accessToken, applyPayload, setId],
  );

  const markMessagesRead = useCallback(
    async (upToMessageId: string | null) => {
      if (!accessToken || !upToMessageId || lastMarkedMessageIdRef.current === upToMessageId) {
        return;
      }
      lastMarkedMessageIdRef.current = upToMessageId;
      const response = await apiFetch(`/workspaces/${setId}/messages/read`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ up_to_message_id: upToMessageId }),
      });
      if (response.ok) {
        onReadStateChange?.();
      }
    },
    [accessToken, onReadStateChange, setId],
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
    lastMarkedMessageIdRef.current = null;
  }, [setId]);

  useEffect(() => {
    if (!accessToken || accessDenied) {
      return;
    }

    void fetchMessages();

    const supabase = createClient();
    const channel = supabase
      .channel(`workspace-messages-${setId}`)
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "workspace_messages", filter: `workspace_id=eq.${setId}` },
        () => {
          void fetchMessages({ silent: true });
        },
      )
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "workspace_message_reads" },
        () => {
          void fetchMessages({ silent: true });
        },
      )
      .subscribe((status) => {
        if (status === "SUBSCRIBED") {
          setRealtimeConnected(true);
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
        if (status === "CHANNEL_ERROR" || status === "TIMED_OUT" || status === "CLOSED") {
          setRealtimeConnected(false);
          if (!pollRef.current) {
            pollRef.current = setInterval(() => {
              void fetchMessages({ silent: true });
            }, FALLBACK_POLL_INTERVAL_MS);
          }
        }
      });

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      void supabase.removeChannel(channel);
    };
  }, [accessDenied, accessToken, fetchMessages, setId]);

  useEffect(() => {
    if (skipAutoScrollRef.current || loading || messages.length === 0) {
      return;
    }
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [loading, messages]);

  useEffect(() => {
    if (messages.length === 0) {
      return;
    }
    const latest = messages[messages.length - 1];
    void markMessagesRead(latest.id);
  }, [markMessagesRead, messages]);

  const { typingLabel } = useTeamChatTyping({
    workspaceId: setId,
    accessToken,
    currentUserId,
    draft,
    enabled: !accessDenied && !migrationNotice,
  });

  const canDeleteMessage = useCallback(
    (message: TeamChatMessage) => {
      if (message.is_own || message.author_id === currentUserId) {
        return true;
      }
      return isOwner;
    },
    [currentUserId, isOwner],
  );

  const handleSend = async () => {
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
      setError(parseApiError(body, `Could not send message (${response.status})`));
      return;
    }
    setDraft("");
    await fetchMessages({ silent: true });
  };

  const handleDelete = async (messageId: string) => {
    if (!accessToken) {
      return;
    }
    setDeletingId(messageId);
    setError(null);
    const response = await apiFetch(`/workspaces/${setId}/messages/${messageId}`, accessToken, {
      method: "DELETE",
    });
    setDeletingId(null);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setError(parseApiError(body, `Could not delete message (${response.status})`));
      return;
    }
    await fetchMessages({ silent: true });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const composerDisabled = !accessToken || sending || accessDenied || Boolean(migrationNotice);

  const threadArea = (
    <div ref={scrollContainerRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-3 sm:px-4">
      {migrationNotice ? (
        <p className="mb-3 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-100">
          {migrationNotice}
        </p>
      ) : null}

      {hasMore && !loading ? (
        <div className="mb-3 flex justify-center">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 rounded-full text-xs"
            disabled={loadingOlder || !nextBefore}
            onClick={() => void fetchMessages({ silent: true, before: nextBefore })}
          >
            {loadingOlder ? (
              <>
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                Loading…
              </>
            ) : (
              "Load older messages"
            )}
          </Button>
        </div>
      ) : null}

      {loading ? (
        <div className="flex h-40 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : messages.length === 0 ? (
        <div className="flex h-40 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
          <MessageCircle className="h-10 w-10 opacity-40" />
          <p className="text-sm font-medium text-foreground">Start a conversation</p>
          <p className="max-w-xs text-xs">
            {workspaceName
              ? `Invite teammates to ${workspaceName} from Share, then coordinate here.`
              : "Invite teammates from Share, then coordinate study plans and deadlines here."}
          </p>
        </div>
      ) : (
        <TeamChatThread
          messages={messages}
          currentUserId={currentUserId}
          deletingId={deletingId}
          canDelete={canDeleteMessage}
          onDelete={(messageId) => void handleDelete(messageId)}
        />
      )}
      <div ref={bottomRef} />
    </div>
  );

  const composer = (
    <div className="shrink-0 border-t bg-muted/20 px-3 py-3 sm:px-4">
      {typingLabel ? (
        <p className="mb-2 text-xs italic text-muted-foreground" aria-live="polite">
          {typingLabel}
        </p>
      ) : null}
      {error ? (
        <p className="mb-2 text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : null}
      <div className="flex items-end gap-2">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Write a message…"
          rows={1}
          maxLength={2000}
          className={cn(
            "flex min-h-[44px] max-h-32 w-full resize-none rounded-2xl border border-muted-foreground/20 bg-background px-4 py-3 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          )}
          disabled={composerDisabled}
          aria-label="Team chat message"
        />
        <Button
          type="button"
          size="icon"
          className="h-11 w-11 shrink-0 rounded-full"
          onClick={() => void handleSend()}
          disabled={composerDisabled || !draft.trim()}
          aria-label="Send message"
        >
          {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </div>
      <p className="mt-1.5 text-[10px] text-muted-foreground">
        Enter to send · Shift+Enter for new line · Plain text only ·{" "}
        {realtimeConnected ? "Live updates" : "Refreshing every 30s"}
      </p>
    </div>
  );

  const body = (
    <div className="flex min-h-0 flex-1 flex-col" data-tour={embedded || dock ? undefined : "team-chat"}>
      {threadArea}
      {composer}
    </div>
  );

  if (dock) {
    return body;
  }
  if (embedded) {
    return (
      <div className="flex max-h-[min(520px,70vh)] min-h-[280px] flex-col overflow-hidden rounded-lg border bg-card">
        {body}
      </div>
    );
  }

  return (
    <Card className="flex h-[min(640px,75vh)] flex-col overflow-hidden shadow-sm" data-tour="team-chat">
      <CardHeader className="shrink-0 border-b bg-muted/30 pb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10">
            <Users className="h-4 w-4 text-primary" />
          </div>
          <div className="min-w-0 flex-1">
            <CardTitle className="text-base">Team chat</CardTitle>
            <CardDescription>
              Plain-text discussion for study sheet members. No AI, links, files, or emoji.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col p-0">{body}</CardContent>
    </Card>
  );
}
