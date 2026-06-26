"use client";

import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

type CollapsibleSectionProps = {
  title: string;
  description?: string;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
  tourId?: string;
};

export function CollapsibleSection({
  title,
  description,
  defaultOpen = false,
  children,
  className,
  tourId,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section
      className={cn("overflow-hidden rounded-xl border bg-card shadow-sm", className)}
      data-tour={tourId}
    >
      <button
        type="button"
        className="flex w-full items-start justify-between gap-3 px-4 py-3 text-left hover:bg-muted/30"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <div className="min-w-0">
          <h2 className="text-base font-semibold">{title}</h2>
          {description ? <p className="mt-0.5 text-sm text-muted-foreground">{description}</p> : null}
        </div>
        <ChevronDown
          className={cn(
            "mt-0.5 h-5 w-5 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
          aria-hidden
        />
      </button>
      {open ? <div className="border-t px-4 py-4">{children}</div> : null}
    </section>
  );
}
