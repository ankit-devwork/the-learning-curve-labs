"use client";

import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { markOnboardingComplete } from "@/lib/onboarding";

export type TourStep = {
  id: string;
  title: string;
  body: string;
  target?: string;
  placement?: "top" | "bottom" | "left" | "right" | "center";
};

type SpotlightRect = {
  top: number;
  left: number;
  width: number;
  height: number;
};

type TooltipPosition = {
  top: number;
  left: number;
};

const PADDING = 8;
const TOOLTIP_WIDTH = 320;
const TOOLTIP_GAP = 16;

function getTargetRect(selector?: string): SpotlightRect | null {
  if (!selector) {
    return null;
  }
  const element = document.querySelector(selector);
  if (!element) {
    return null;
  }
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top - PADDING,
    left: rect.left - PADDING,
    width: rect.width + PADDING * 2,
    height: rect.height + PADDING * 2,
  };
}

function getTooltipPosition(
  rect: SpotlightRect | null,
  placement: TourStep["placement"],
): TooltipPosition {
  if (!rect || placement === "center") {
    return {
      top: Math.max(24, window.innerHeight / 2 - 120),
      left: Math.max(16, (window.innerWidth - TOOLTIP_WIDTH) / 2),
    };
  }

  const clampLeft = (left: number) =>
    Math.min(Math.max(16, left), window.innerWidth - TOOLTIP_WIDTH - 16);

  switch (placement) {
    case "top":
      return {
        top: Math.max(16, rect.top - TOOLTIP_GAP - 180),
        left: clampLeft(rect.left),
      };
    case "left":
      return {
        top: Math.max(16, rect.top),
        left: clampLeft(rect.left - TOOLTIP_WIDTH - TOOLTIP_GAP),
      };
    case "right":
      return {
        top: Math.max(16, rect.top),
        left: clampLeft(rect.left + rect.width + TOOLTIP_GAP),
      };
    case "bottom":
    default:
      return {
        top: Math.min(window.innerHeight - 220, rect.top + rect.height + TOOLTIP_GAP),
        left: clampLeft(rect.left),
      };
  }
}

type OnboardingTourProps = {
  steps: TourStep[];
  open: boolean;
  onClose: () => void;
  onComplete?: () => void;
};

export function OnboardingTour({ steps, open, onClose, onComplete }: OnboardingTourProps) {
  const [stepIndex, setStepIndex] = useState(0);
  const [spotlight, setSpotlight] = useState<SpotlightRect | null>(null);
  const [tooltip, setTooltip] = useState<TooltipPosition>({ top: 0, left: 0 });

  const step = steps[stepIndex];
  const isLastStep = stepIndex === steps.length - 1;

  const updateLayout = useCallback(() => {
    if (!open || !step) {
      return;
    }
    const rect = getTargetRect(step.target);
    setSpotlight(rect);
    setTooltip(getTooltipPosition(rect, step.placement));
  }, [open, step]);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }
    setStepIndex(0);
  }, [open]);

  useLayoutEffect(() => {
    updateLayout();
  }, [updateLayout, stepIndex]);

  useLayoutEffect(() => {
    if (!open || !step?.target) {
      return;
    }
    const element = document.querySelector(step.target);
    element?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const timer = window.setTimeout(updateLayout, 300);
    return () => window.clearTimeout(timer);
  }, [open, step, updateLayout, stepIndex]);

  useEffect(() => {
    if (!open) {
      return;
    }

    const handleResize = () => updateLayout();
    window.addEventListener("resize", handleResize);
    window.addEventListener("scroll", handleResize, true);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("scroll", handleResize, true);
    };
  }, [open, updateLayout]);

  function handleSkip() {
    markOnboardingComplete();
    onComplete?.();
    onClose();
  }

  function handleNext() {
    if (isLastStep) {
      markOnboardingComplete();
      onComplete?.();
      onClose();
      return;
    }
    setStepIndex((current) => current + 1);
  }

  if (!open || !step) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      {!spotlight ? (
        <div
          className="absolute inset-0 bg-black/55 transition-opacity"
          onClick={handleSkip}
          aria-hidden
        />
      ) : null}

      {spotlight ? (
        <button
          type="button"
          className="absolute inset-0 cursor-default bg-transparent"
          onClick={handleSkip}
          aria-label="Skip tour"
        />
      ) : null}

      {spotlight ? (
        <div
          className="pointer-events-none absolute rounded-xl ring-2 ring-primary ring-offset-2 ring-offset-transparent transition-all duration-200"
          style={{
            top: spotlight.top,
            left: spotlight.left,
            width: spotlight.width,
            height: spotlight.height,
            boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.55)",
          }}
        />
      ) : null}

      <div
        className={cn(
          "absolute z-[101] w-[min(320px,calc(100vw-2rem))] rounded-xl border bg-background p-5 shadow-xl",
          "animate-in fade-in slide-in-from-bottom-2 duration-200",
        )}
        style={{ top: tooltip.top, left: tooltip.left }}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-primary">
              Step {stepIndex + 1} of {steps.length}
            </p>
            <h3 id="tour-title" className="mt-1 text-base font-semibold">
              {step.title}
            </h3>
          </div>
          <button
            type="button"
            onClick={handleSkip}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Close tour"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{step.body}</p>

        <div className="mt-5 flex items-center justify-between gap-2">
          <Button type="button" variant="ghost" size="sm" onClick={handleSkip}>
            Skip tour
          </Button>
          <Button type="button" size="sm" onClick={handleNext}>
            {isLastStep ? "Get started" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  );
}
