"use client";

import { Pause, Play, Volume2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type AudioPlayerBarProps = {
  title: string;
  subtitle?: string;
  playing: boolean;
  onPlayPause: () => void;
  onStop: () => void;
  className?: string;
};

export function AudioPlayerBar({
  title,
  subtitle,
  playing,
  onPlayPause,
  onStop,
  className,
}: AudioPlayerBarProps) {
  return (
    <div
      className={cn(
        "fixed inset-x-0 bottom-0 z-50 border-t bg-background/95 px-4 py-3 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/80",
        className,
      )}
    >
      <div className="mx-auto flex max-w-5xl items-center gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Volume2 className="h-5 w-5" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{title}</p>
          {subtitle ? <p className="truncate text-xs text-muted-foreground">{subtitle}</p> : null}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button type="button" size="sm" className="gap-1.5" onClick={onPlayPause}>
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {playing ? "Pause" : "Play"}
          </Button>
          <Button type="button" size="icon" variant="ghost" aria-label="Stop audio" onClick={onStop}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
