"use client";

import {
  BookOpen,
  Brain,
  FileText,
  Image,
  Layers,
  MessageSquare,
  Network,
  PlayCircle,
  Presentation,
  Sparkles,
  Volume2,
  Wrench,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DocumentWorkspaceTab, StudioTab } from "@/lib/notebook-utils";
import { STUDIO_TAB_LABELS } from "@/lib/notebook-utils";
import { cn } from "@/lib/utils";

type StudioPanelProps = {
  activeTab: DocumentWorkspaceTab;
  ready: boolean;
  busy?: boolean;
  badges?: Partial<Record<StudioTab, string | number>>;
  onSelectTab: (tab: StudioTab) => void;
  className?: string;
};

const STUDIO_GROUPS: Array<{
  label: string;
  items: Array<{ id: StudioTab; icon: typeof FileText }>;
}> = [
  {
    label: "Read",
    items: [
      { id: "brief", icon: FileText },
      { id: "guide", icon: BookOpen },
    ],
  },
  {
    label: "Practice",
    items: [
      { id: "session", icon: PlayCircle },
      { id: "quiz", icon: Brain },
      { id: "flashcards", icon: Layers },
    ],
  },
  {
    label: "Create",
    items: [
      { id: "infographic", icon: Image },
      { id: "slides", icon: Presentation },
      { id: "homework", icon: Wrench },
      { id: "audio", icon: Volume2 },
    ],
  },
  {
    label: "Explore",
    items: [{ id: "concepts", icon: Network }],
  },
];

export function StudioPanel({
  activeTab,
  ready,
  busy,
  badges,
  onSelectTab,
  className,
}: StudioPanelProps) {
  return (
    <Card className={cn("notebook-surface border-0 shadow-none", className)} data-tour="studio-panel">
      <CardHeader className="pb-2 pt-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" aria-hidden />
          <CardTitle className="text-sm font-semibold">Studio</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pb-3">
        {STUDIO_GROUPS.map((group) => (
          <div key={group.label}>
            <p className="mb-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {group.label}
            </p>
            <div className="grid gap-0.5">
              {group.items.map(({ id, icon: Icon }) => {
                const selected = activeTab === id;
                const badge = badges?.[id];
                return (
                  <button
                    key={id}
                    type="button"
                    disabled={!ready || busy}
                    aria-current={selected ? "true" : undefined}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
                      selected
                        ? "bg-primary/10 text-primary ring-1 ring-primary/20"
                        : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                      (!ready || busy) && "cursor-not-allowed opacity-60",
                    )}
                    onClick={() => onSelectTab(id)}
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden />
                    <span className="min-w-0 flex-1 truncate font-medium">
                      {STUDIO_TAB_LABELS[id]}
                    </span>
                    {badge != null ? (
                      <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] tabular-nums text-primary">
                        {badge}
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function StudioChatShortcut({
  active,
  onSelect,
  className,
}: {
  active: boolean;
  onSelect: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      aria-current={active ? "true" : undefined}
      className={cn(
        "mb-2 flex w-full items-center gap-2 rounded-md border px-2 py-2 text-left text-sm transition-colors",
        active
          ? "border-primary/30 bg-primary/10 text-primary"
          : "border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground",
        className,
      )}
      onClick={onSelect}
    >
      <MessageSquare className="h-3.5 w-3.5 shrink-0" aria-hidden />
      <span className="font-medium">Chat</span>
    </button>
  );
}
