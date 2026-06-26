"use client";

import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export type UploadGuidanceConfig = {
  summary: string;
  points: string[];
  require_acknowledgment: boolean;
};

type UploadGuidanceProps = {
  guidance: UploadGuidanceConfig;
  acknowledged: boolean;
  onAcknowledgedChange: (value: boolean) => void;
  className?: string;
};

type UploadGuidanceNoticeProps = {
  guidance: UploadGuidanceConfig;
  className?: string;
  compact?: boolean;
};

export function UploadGuidanceNotice({
  guidance,
  className,
  compact = false,
}: UploadGuidanceNoticeProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-amber-200/80 bg-amber-50/80 px-4 py-3 text-sm dark:border-amber-900/50 dark:bg-amber-950/30",
        className,
      )}
      role="note"
      aria-label="Upload privacy notice"
    >
      <div className="flex gap-3">
        <AlertTriangle
          className={cn("shrink-0 text-amber-700 dark:text-amber-400", compact ? "mt-0.5 h-4 w-4" : "mt-0.5 h-4 w-4")}
          aria-hidden
        />
        <div className="min-w-0 space-y-2">
          <p className="font-medium text-amber-950 dark:text-amber-100">{guidance.summary}</p>
          {!compact ? (
            <ul className="list-disc space-y-1 pl-4 text-amber-900/90 dark:text-amber-100/80">
              {guidance.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          ) : (
            <ul className="space-y-1 text-xs text-amber-900/90 dark:text-amber-100/80">
              {guidance.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export function UploadGuidance({
  guidance,
  acknowledged,
  onAcknowledgedChange,
  className,
}: UploadGuidanceProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-amber-200/80 bg-amber-50/80 px-4 py-3 text-sm dark:border-amber-900/50 dark:bg-amber-950/30",
        className,
      )}
      role="note"
      aria-label="Upload privacy guidance"
    >
      <div className="flex gap-3">
        <AlertTriangle
          className="mt-0.5 h-4 w-4 shrink-0 text-amber-700 dark:text-amber-400"
          aria-hidden
        />
        <div className="space-y-2">
          <p className="font-medium text-amber-950 dark:text-amber-100">{guidance.summary}</p>
          <ul className="list-disc space-y-1 pl-4 text-amber-900/90 dark:text-amber-100/80">
            {guidance.points.map((point) => (
              <li key={point}>{point}</li>
            ))}
          </ul>
          {guidance.require_acknowledgment ? (
            <label className="flex cursor-pointer items-start gap-2 pt-1">
              <input
                type="checkbox"
                checked={acknowledged}
                onChange={(event) => onAcknowledgedChange(event.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-amber-300"
              />
              <span className="text-amber-950 dark:text-amber-100">
                I confirm this file is appropriate to share with study sheet members.
              </span>
            </label>
          ) : null}
        </div>
      </div>
    </div>
  );
}
