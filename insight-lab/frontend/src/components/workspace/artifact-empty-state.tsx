"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

type ArtifactEmptyStateProps = {
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  actionDisabled?: boolean;
  actionBusy?: boolean;
};

export function ArtifactEmptyState({
  message,
  actionLabel,
  onAction,
  actionDisabled,
  actionBusy,
}: ArtifactEmptyStateProps) {
  return (
    <Card className="notebook-surface border-0 shadow-none">
      <CardContent className="flex flex-col items-center gap-3 py-10 text-center">
        <p className="max-w-sm text-sm text-muted-foreground">{message}</p>
        {actionLabel && onAction ? (
          <Button type="button" disabled={actionDisabled || actionBusy} onClick={onAction}>
            {actionBusy ? "Generating…" : actionLabel}
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}
