"use client";

import { cn } from "@/lib/utils";

const STAGES = [
  { key: "queued", label: "Queued" },
  { key: "processing", label: "Processing" },
  { key: "ready", label: "Ready" },
];

type ProcessingStepperProps = {
  stage: string;
  progressPct: number;
  message: string;
  className?: string;
};

export function ProcessingStepper({
  stage,
  progressPct,
  message,
  className,
}: ProcessingStepperProps) {
  const activeIndex = STAGES.findIndex((item) => item.key === stage);

  return (
    <div className={cn("space-y-3 rounded-xl border bg-card p-4", className)}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-medium">Processing status</p>
        <span className="text-xs text-muted-foreground">{progressPct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500"
          style={{ width: `${Math.max(0, Math.min(progressPct, 100))}%` }}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {STAGES.map((item, index) => (
          <span
            key={item.key}
            className={cn(
              "rounded-full px-2.5 py-1 text-xs font-medium",
              index <= activeIndex
                ? "bg-primary/10 text-primary"
                : "bg-muted text-muted-foreground",
            )}
          >
            {item.label}
          </span>
        ))}
      </div>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
