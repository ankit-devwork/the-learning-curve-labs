"use client";

import { Loader2, MoreHorizontal } from "lucide-react";
import type { TeamChatMessage } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const AVATAR_COLORS = [
  "bg-sky-600 text-white",
  "bg-emerald-600 text-white",
  "bg-violet-600 text-white",
  "bg-amber-600 text-white",
  "bg-rose-600 text-white",
  "bg-cyan-700 text-white",
] as const;

type TeamChatThreadProps = {
  messages: TeamChatMessage[];
  currentUserId: string | null;
  deletingId?: string | null;
  onDelete?: (messageId: string) => void;
  canDelete?: (message: TeamChatMessage) => boolean;
};

type RenderItem =
  | { type: "date"; key: string; label: string }
  | { type: "message"; key: string; message: TeamChatMessage; showHeader: boolean };

function avatarColor(seed: string): string {
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = seed.charCodeAt(index) + ((hash << 5) - hash);
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) {
    return "?";
  }
  if (parts.length === 1) {
    return parts[0].slice(0, 2).toUpperCase();
  }
  return `${parts[0][0] ?? ""}${parts[1][0] ?? ""}`.toUpperCase();
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function dateDividerLabel(iso: string): string {
  const date = new Date(iso);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);

  const sameDay = (left: Date, right: Date) =>
    left.getFullYear() === right.getFullYear() &&
    left.getMonth() === right.getMonth() &&
    left.getDate() === right.getDate();

  if (sameDay(date, today)) {
    return "Today";
  }
  if (sameDay(date, yesterday)) {
    return "Yesterday";
  }
  return date.toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

function buildRenderItems(messages: TeamChatMessage[]): RenderItem[] {
  const items: RenderItem[] = [];
  let lastDateKey = "";
  let lastAuthorId = "";
  let lastTimestamp = 0;

  for (const message of messages) {
    const dateKey = new Date(message.created_at).toDateString();
    if (dateKey !== lastDateKey) {
      items.push({
        type: "date",
        key: `date-${dateKey}`,
        label: dateDividerLabel(message.created_at),
      });
      lastDateKey = dateKey;
      lastAuthorId = "";
      lastTimestamp = 0;
    }

    const timestamp = new Date(message.created_at).getTime();
    const showHeader =
      message.author_id !== lastAuthorId || timestamp - lastTimestamp > 5 * 60 * 1000;

    items.push({
      type: "message",
      key: message.id,
      message,
      showHeader,
    });
    lastAuthorId = message.author_id;
    lastTimestamp = timestamp;
  }

  return items;
}

function MemberAvatar({ message, isOwn }: { message: TeamChatMessage; isOwn: boolean }) {
  const label = isOwn ? "You" : message.author_name;
  if (message.author_avatar_url) {
    return (
      <img
        src={message.author_avatar_url}
        alt=""
        className="h-9 w-9 shrink-0 rounded-full object-cover ring-1 ring-border/60"
      />
    );
  }

  return (
    <span
      className={cn(
        "flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-xs font-semibold ring-1 ring-border/40",
        avatarColor(message.author_id),
      )}
      aria-hidden
    >
      {initials(label)}
    </span>
  );
}

export function TeamChatThread({
  messages,
  currentUserId,
  deletingId = null,
  onDelete,
  canDelete,
}: TeamChatThreadProps) {
  const items = buildRenderItems(messages);

  return (
    <div className="space-y-1" aria-live="polite">
      {items.map((item) => {
        if (item.type === "date") {
          return (
            <div key={item.key} className="flex justify-center py-3">
              <span className="rounded-full bg-muted px-3 py-0.5 text-[11px] font-medium text-muted-foreground">
                {item.label}
              </span>
            </div>
          );
        }

        const message = item.message;
        const isOwn = message.is_own || message.author_id === currentUserId;
        const deletable = canDelete?.(message) ?? false;
        const isDeleting = deletingId === message.id;

        return (
          <div
            key={item.key}
            className={cn(
              "group flex gap-2",
              isOwn ? "flex-row-reverse" : "flex-row",
              !item.showHeader && (isOwn ? "pr-11" : "pl-11"),
            )}
          >
            {item.showHeader ? (
              <MemberAvatar message={message} isOwn={isOwn} />
            ) : (
              <span className="w-9 shrink-0" aria-hidden />
            )}

            <div className={cn("min-w-0 max-w-[min(100%,34rem)]", isOwn ? "items-end" : "items-start")}>
              {item.showHeader ? (
                <div
                  className={cn(
                    "mb-1 flex items-center gap-2",
                    isOwn ? "flex-row-reverse" : "flex-row",
                  )}
                >
                  <p className="text-xs font-semibold text-foreground">
                    {isOwn ? "You" : message.author_name}
                  </p>
                  <span className="text-[10px] text-muted-foreground">{formatTime(message.created_at)}</span>
                  {deletable && onDelete ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 opacity-0 transition-opacity group-hover:opacity-100"
                      aria-label="Delete message"
                      disabled={isDeleting}
                      onClick={() => onDelete(message.id)}
                    >
                      {isDeleting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <MoreHorizontal className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  ) : null}
                </div>
              ) : null}

              <div
                className={cn(
                  "rounded-2xl px-3.5 py-2 text-sm leading-relaxed shadow-sm",
                  isOwn
                    ? "rounded-tr-md bg-primary text-primary-foreground"
                    : "rounded-tl-md border bg-card text-foreground",
                )}
              >
                <p className="whitespace-pre-wrap">{message.body}</p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
