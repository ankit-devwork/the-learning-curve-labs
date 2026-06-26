"use client";

import { useEffect, useState } from "react";
import { CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type PanelHeaderProps = {
  hintId: string;
  title: string;
  description?: string;
  className?: string;
  titleClassName?: string;
};

function hintStorageKey(hintId: string): string {
  return `ilab-panel-hint:${hintId}`;
}

export function PanelHeader({
  hintId,
  title,
  description,
  className,
  titleClassName,
}: PanelHeaderProps) {
  const [showDescription, setShowDescription] = useState(false);

  useEffect(() => {
    try {
      setShowDescription(!window.localStorage.getItem(hintStorageKey(hintId)));
    } catch {
      setShowDescription(Boolean(description));
    }
  }, [hintId, description]);

  useEffect(() => {
    if (!showDescription || !description) {
      return;
    }
    try {
      window.localStorage.setItem(hintStorageKey(hintId), "1");
    } catch {
      // ignore storage failures
    }
  }, [showDescription, description, hintId]);

  return (
    <CardHeader className={cn("pb-3", className)}>
      <CardTitle className={cn("text-lg", titleClassName)}>{title}</CardTitle>
      {showDescription && description ? (
        <CardDescription>{description}</CardDescription>
      ) : null}
    </CardHeader>
  );
}
