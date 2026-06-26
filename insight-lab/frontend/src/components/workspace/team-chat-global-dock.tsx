"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, ChevronDown, Loader2, MessageCircle } from "lucide-react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch, type TeamChatConversation, type TeamChatInboxResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { TeamChatInbox } from "@/components/workspace/team-chat-inbox";
import { TeamChatPanel } from "@/components/workspace/team-chat-panel";
import { cn } from "@/lib/utils";

const INBOX_POLL_MS = 30_000;

type TeamChatGlobalDockProps = {
  /** When viewing a study sheet, open that thread directly from the launcher. */
  contextWorkspaceId?: string | null;
};

type DockView = "inbox" | "thread";

export function TeamChatGlobalDock({ contextWorkspaceId = null }: TeamChatGlobalDockProps) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<DockView>("inbox");
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [conversations, setConversations] = useState<TeamChatConversation[]>([]);
  const [totalUnread, setTotalUnread] = useState(0);
  const [inboxLoading, setInboxLoading] = useState(true);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [activeWorkspaceName, setActiveWorkspaceName] = useState("");
  const [activeIsOwner, setActiveIsOwner] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadInbox = useCallback(async (options?: { silent?: boolean }) => {
    if (!accessToken) {
      return;
    }
    if (!options?.silent) {
      setInboxLoading(true);
    }
    const response = await apiFetch("/workspaces/messages/inbox", accessToken);
    if (!options?.silent) {
      setInboxLoading(false);
    }
    if (!response.ok) {
      return;
    }
    const payload = (await response.json()) as TeamChatInboxResponse;
    setConversations(payload.conversations ?? []);
    setTotalUnread(payload.total_unread ?? 0);
  }, [accessToken]);

  useEffect(() => {
    async function loadSession() {
      const supabase = createClient();
      const {
        data: { session },
      } = await supabase.auth.getSession();
      setAccessToken(session?.access_token ?? null);
    }
    void loadSession();
  }, []);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    void loadInbox();

    const supabase = createClient();
    const channel = supabase
      .channel("team-chat-inbox")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "workspace_messages" },
        () => {
          void loadInbox({ silent: true });
        },
      )
      .subscribe();

    pollRef.current = setInterval(() => {
      void loadInbox({ silent: true });
    }, INBOX_POLL_MS);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      void supabase.removeChannel(channel);
    };
  }, [accessToken, loadInbox]);

  const openConversation = useCallback(
    (workspaceId: string) => {
      const conversation = conversations.find((row) => row.workspace_id === workspaceId);
      setActiveWorkspaceId(workspaceId);
      setActiveWorkspaceName(conversation?.workspace_name ?? "Study sheet");
      setActiveIsOwner(Boolean(conversation?.is_owner));
      setView("thread");
    },
    [conversations],
  );

  const handleOpenDock = () => {
    setOpen(true);
    if (contextWorkspaceId) {
      openConversation(contextWorkspaceId);
    } else {
      setView("inbox");
    }
  };

  const handleBackToInbox = () => {
    setView("inbox");
    setActiveWorkspaceId(null);
    void loadInbox({ silent: true });
  };

  if (!open) {
    return (
      <div className="pointer-events-none fixed bottom-0 right-4 z-[110] sm:right-6">
        <button
          type="button"
          data-tour="team-chat"
          onClick={handleOpenDock}
          className={cn(
            "pointer-events-auto flex items-center gap-2 rounded-t-xl border border-b-0 bg-card px-4 py-2.5 shadow-lg",
            "transition-colors hover:bg-muted/50",
          )}
          aria-label="Open team chat"
        >
          <span className="relative flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
            <MessageCircle className="h-4 w-4" />
            {totalUnread > 0 ? (
              <span className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-primary px-1 text-[9px] font-bold text-primary-foreground">
                {totalUnread > 99 ? "99+" : totalUnread}
              </span>
            ) : null}
          </span>
          <span className="text-sm font-semibold text-foreground">Team chat</span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "pointer-events-none fixed bottom-0 right-4 z-[110] sm:right-6",
        "flex w-[min(360px,calc(100vw-2rem))] flex-col overflow-hidden rounded-t-xl border border-b-0 bg-card shadow-2xl",
        "h-[min(520px,calc(100vh-5rem))]",
      )}
      data-tour="team-chat"
      role="dialog"
      aria-label="Team chat"
    >
      <header className="pointer-events-auto flex shrink-0 items-center gap-2 border-b bg-muted/30 px-3 py-2.5">
        {view === "thread" ? (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={handleBackToInbox}
            aria-label="Back to conversations"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
        ) : (
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
            <MessageCircle className="h-4 w-4" />
          </span>
        )}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold leading-tight">
            {view === "thread" ? activeWorkspaceName : "Team chat"}
          </p>
          <p className="truncate text-[11px] text-muted-foreground">
            {view === "thread" ? "Study sheet members" : `${conversations.length} study sheets`}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={() => setOpen(false)}
          aria-label="Minimize team chat"
        >
          <ChevronDown className="h-4 w-4" />
        </Button>
      </header>

      <div className="pointer-events-auto flex min-h-0 flex-1 flex-col">
        {view === "inbox" ? (
          <TeamChatInbox
            conversations={conversations}
            loading={inboxLoading}
            activeWorkspaceId={activeWorkspaceId}
            onSelect={openConversation}
          />
        ) : activeWorkspaceId ? (
          <TeamChatPanel
            setId={activeWorkspaceId}
            accessToken={accessToken}
            isOwner={activeIsOwner}
            workspaceName={activeWorkspaceName}
            dock
            onReadStateChange={() => void loadInbox({ silent: true })}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}
      </div>
    </div>
  );
}
