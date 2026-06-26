"use client";

import {
  BookOpen,
  Brain,
  FileText,
  Image,
  Layers,
  Network,
  PlayCircle,
  Presentation,
  Sparkles,
  Volume2,
  Wrench,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { StudioTab } from "@/lib/notebook-utils";
import { cn } from "@/lib/utils";

type StudioPanelProps = {
  activeTab: StudioTab;
  ready: boolean;
  busy?: boolean;
  badges?: Partial<Record<StudioTab, string | number>>;
  onSelectTab: (tab: StudioTab) => void;
  className?: string;
};

const actions: Array<{
  id: StudioTab;
  label: string;
  icon: typeof FileText;
  description: string;
}> = [
  { id: "brief", label: "Brief", icon: FileText, description: "AI summary of this source" },
  { id: "session", label: "Study plan", icon: PlayCircle, description: "Guided brief → cards → quiz" },
  { id: "quiz", label: "Quiz", icon: Brain, description: "Practice questions" },
  { id: "flashcards", label: "Flashcards", icon: Layers, description: "Term and definition cards" },
  { id: "guide", label: "Study guide", icon: BookOpen, description: "Structured overview" },
  { id: "infographic", label: "Infographic", icon: Image, description: "Visual summary card" },
  { id: "slides", label: "Slide deck", icon: Presentation, description: "Presentation outline" },
  { id: "homework", label: "Homework help", icon: Wrench, description: "Step-by-step solver" },
  { id: "audio", label: "Audio overview", icon: Volume2, description: "Narrated summary (MP3)" },
  { id: "concepts", label: "Concept graph", icon: Network, description: "Topics and mastery" },
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
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" aria-hidden />
          <CardTitle className="text-base">Studio</CardTitle>
        </div>
        <CardDescription>Outputs from this source</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-1.5">
        {actions.map(({ id, label, icon: Icon, description }) => {
          const selected = activeTab === id;
          const badge = badges?.[id];
          return (
            <Button
              key={id}
              type="button"
              variant={selected ? "secondary" : "ghost"}
              className={cn(
                "h-auto justify-start gap-3 px-3 py-2.5 text-left",
                selected && "ring-1 ring-primary/20",
              )}
              disabled={!ready || busy}
              aria-current={selected ? "true" : undefined}
              onClick={() => onSelectTab(id)}
            >
              <span
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-md",
                  selected ? "bg-primary text-primary-foreground" : "bg-primary/10 text-primary",
                )}
              >
                <Icon className="h-4 w-4" aria-hidden />
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2">
                  <span className="text-sm font-medium">{label}</span>
                  {badge != null ? (
                    <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
                      {badge}
                    </span>
                  ) : null}
                </span>
                <span className="block text-xs font-normal text-muted-foreground">{description}</span>
              </span>
            </Button>
          );
        })}
      </CardContent>
    </Card>
  );
}
