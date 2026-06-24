"use client";

import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type SlideOverDrawerProps = {
  open: boolean;
  title: string;
  description?: string;
  onClose: () => void;
  children: React.ReactNode;
  className?: string;
  widthClassName?: string;
};

export function SlideOverDrawer({
  open,
  title,
  description,
  onClose,
  children,
  className,
  widthClassName = "w-[min(480px,100vw)]",
}: SlideOverDrawerProps) {
  if (!open) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-[120] bg-black/40"
        aria-label="Close panel"
        onClick={onClose}
      />
      <aside
        className={cn(
          "fixed inset-y-0 right-0 z-[121] flex flex-col border-l bg-background shadow-xl",
          widthClassName,
          className,
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby="slide-over-drawer-title"
      >
        <div className="flex items-start justify-between gap-3 border-b px-4 py-4">
          <div className="min-w-0">
            <h2 id="slide-over-drawer-title" className="text-base font-semibold">
              {title}
            </h2>
            {description ? (
              <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>
            ) : null}
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">{children}</div>
      </aside>
    </>
  );
}
