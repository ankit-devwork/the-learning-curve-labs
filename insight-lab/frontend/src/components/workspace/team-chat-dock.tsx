"use client";

import { ChevronDown, MessageCircle } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { TeamChatPanel } from "@/components/workspace/team-chat-panel";
import { cn } from "@/lib/utils";

type TeamChatDockProps = {
  setId: string;
  accessToken: string | null;
  isOwner: boolean;
};

export function TeamChatDock({ setId, accessToken, isOwner }: TeamChatDockProps) {
  const [open, setOpen] = useState(false);

  if (!open) {
    return (
      <div className="pointer-events-none fixed bottom-0 right-4 z-[110] sm:right-6">
        <button
          type="button"
          data-tour="team-chat"
          onClick={() => setOpen(true)}
          className={cn(
            "pointer-events-auto flex items-center gap-2 rounded-t-xl border border-b-0 bg-card px-4 py-2.5 shadow-lg",
            "transition-colors hover:bg-muted/50",
          )}
          aria-label="Open team chat"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
            <MessageCircle className="h-4 w-4" />
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
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary">
          <MessageCircle className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold leading-tight">Team chat</p>
          <p className="truncate text-[11px] text-muted-foreground">Study sheet members</p>
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
        <TeamChatPanel setId={setId} accessToken={accessToken} isOwner={isOwner} dock />
      </div>
    </div>
  );
}
