"use client";

import { Loader2, MessageCircle } from "lucide-react";
import type { TeamChatConversation } from "@/lib/api";
import { cn } from "@/lib/utils";

type TeamChatInboxProps = {
  conversations: TeamChatConversation[];
  loading: boolean;
  activeWorkspaceId: string | null;
  onSelect: (workspaceId: string) => void;
};

function formatPreviewTime(iso: string | null): string {
  if (!iso) {
    return "";
  }
  const date = new Date(iso);
  const today = new Date();
  const sameDay =
    date.getFullYear() === today.getFullYear() &&
    date.getMonth() === today.getMonth() &&
    date.getDate() === today.getDate();
  if (sameDay) {
    return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function previewText(conversation: TeamChatConversation): string {
  const message = conversation.last_message;
  if (!message) {
    return "No messages yet";
  }
  const prefix = message.is_own ? "You: " : `${message.author_name}: `;
  const body = message.body.length > 72 ? `${message.body.slice(0, 72)}…` : message.body;
  return `${prefix}${body}`;
}

export function TeamChatInbox({
  conversations,
  loading,
  activeWorkspaceId,
  onSelect,
}: TeamChatInboxProps) {
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (conversations.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-muted-foreground">
        <MessageCircle className="h-10 w-10 opacity-40" />
        <p className="text-sm font-medium text-foreground">No study sheet chats yet</p>
        <p className="text-xs">Join or create a study sheet to start coordinating with teammates.</p>
      </div>
    );
  }

  return (
    <ul className="divide-y overflow-y-auto">
      {conversations.map((conversation) => {
        const isActive = conversation.workspace_id === activeWorkspaceId;
        const hasUnread = conversation.unread_count > 0;

        return (
          <li key={conversation.workspace_id}>
            <button
              type="button"
              onClick={() => onSelect(conversation.workspace_id)}
              className={cn(
                "flex w-full items-start gap-3 px-3 py-3 text-left transition-colors hover:bg-muted/50",
                isActive && "bg-primary/5",
              )}
            >
              <span className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                <MessageCircle className="h-4 w-4" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-start justify-between gap-2">
                  <span
                    className={cn(
                      "truncate text-sm",
                      hasUnread ? "font-semibold text-foreground" : "font-medium text-foreground",
                    )}
                  >
                    {conversation.workspace_name}
                  </span>
                  <span className="shrink-0 text-[10px] text-muted-foreground">
                    {formatPreviewTime(conversation.last_message_at)}
                  </span>
                </span>
                <span className="mt-0.5 flex items-center justify-between gap-2">
                  <span
                    className={cn(
                      "truncate text-xs",
                      hasUnread ? "font-medium text-foreground" : "text-muted-foreground",
                    )}
                  >
                    {previewText(conversation)}
                  </span>
                  {hasUnread ? (
                    <span className="flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full bg-primary px-1.5 text-[10px] font-semibold text-primary-foreground">
                      {conversation.unread_count > 99 ? "99+" : conversation.unread_count}
                    </span>
                  ) : null}
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
