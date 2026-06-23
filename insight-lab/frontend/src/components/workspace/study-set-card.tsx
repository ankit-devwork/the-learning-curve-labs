"use client";

import Link from "next/link";
import { Users } from "lucide-react";
import type { WorkspaceSummary } from "@/lib/api";
import { notebookAccentClass } from "@/lib/notebook-utils";
import { cn } from "@/lib/utils";

type StudySetCardProps = {
  workspace: WorkspaceSummary;
  sourceCount?: number;
  readyCount?: number;
};

export function StudySetCard({ workspace, sourceCount, readyCount }: StudySetCardProps) {
  const accent = notebookAccentClass(workspace.name);

  return (
    <Link
      href={`/dashboard/sets/${workspace.id}`}
      className={cn(
        "group block overflow-hidden rounded-xl border bg-gradient-to-br shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md",
        accent,
      )}
    >
      <div className="p-5">
        <div className="flex items-start justify-between gap-2">
          <p className="font-semibold tracking-tight group-hover:text-primary">{workspace.name}</p>
          {workspace.shared ? (
            <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-background/70 px-2 py-0.5 text-[10px] font-medium text-primary">
              <Users className="h-3 w-3" aria-hidden />
              Shared
            </span>
          ) : null}
        </div>
        {workspace.description ? (
          <p className="mt-2 line-clamp-2 text-sm text-muted-foreground">{workspace.description}</p>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">Open notebook to chat, quiz, and study</p>
        )}
        <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
          {sourceCount != null ? (
            <span className="rounded-full bg-background/60 px-2 py-0.5">
              {sourceCount} source{sourceCount === 1 ? "" : "s"}
            </span>
          ) : null}
          {readyCount != null ? (
            <span className="rounded-full bg-background/60 px-2 py-0.5">
              {readyCount} ready
            </span>
          ) : null}
          {workspace.access_role ? (
            <span className="rounded-full bg-background/60 px-2 py-0.5 capitalize">
              {workspace.access_role}
            </span>
          ) : null}
        </div>
      </div>
    </Link>
  );
}
