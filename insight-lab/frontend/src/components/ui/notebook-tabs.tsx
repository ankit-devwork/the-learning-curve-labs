"use client";

import { cn } from "@/lib/utils";

export type NotebookTab = {
  id: string;
  label: string;
  badge?: string | number;
};

type NotebookTabsProps = {
  tabs: NotebookTab[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
  sticky?: boolean;
};

export function NotebookTabs({ tabs, active, onChange, className, sticky }: NotebookTabsProps) {
  return (
    <div
      className={cn(
        sticky &&
          "sticky top-0 z-20 -mx-1 bg-background/95 px-1 py-2 backdrop-blur supports-[backdrop-filter]:bg-background/80",
        className,
      )}
    >
      <div
        className="flex gap-1 overflow-x-auto rounded-xl border bg-muted/30 p-1"
        role="tablist"
      >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          className={cn(
            "shrink-0 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
            active === tab.id
              ? "bg-background text-foreground shadow-sm ring-1 ring-border/50"
              : "text-muted-foreground hover:text-foreground",
          )}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
          {tab.badge != null ? (
            <span className="ml-1.5 rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
              {tab.badge}
            </span>
          ) : null}
        </button>
      ))}
      </div>
    </div>
  );
}
