"use client";

import { useEffect, useState } from "react";
import { Compass } from "lucide-react";
import { Button } from "@/components/ui/button";
import { OnboardingTour, type TourStep } from "@/components/onboarding/onboarding-tour";
import { isOnboardingComplete, resetOnboarding } from "@/lib/onboarding";

const DASHBOARD_TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to InsightLab",
    body: "Upload lecture notes, PDFs, or spreadsheets and explore them with AI summaries, chat, quizzes, and charts.",
    placement: "center",
  },
  {
    id: "upload",
    title: "Upload your first file",
    body: "Click Upload file to add a PDF, Word doc, Excel sheet, or CSV. Processing usually takes about a minute.",
    target: '[data-tour="upload-btn"]',
    placement: "bottom",
  },
  {
    id: "files",
    title: "Open a ready file",
    body: "When status shows Ready, click the filename to open summaries, Ask, Quiz (documents), or charts (spreadsheets).",
    target: '[data-tour="file-list"]',
    placement: "top",
  },
  {
    id: "compare",
    title: "Compare multiple documents",
    body: "Select two or more ready documents here, ask one question, and get a combined answer with citations.",
    target: '[data-tour="compare-docs"]',
    placement: "top",
  },
];

export function DashboardOnboarding() {
  const [tourOpen, setTourOpen] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!isOnboardingComplete()) {
        setTourOpen(true);
      }
    }, 600);

    return () => window.clearTimeout(timer);
  }, []);

  function restartTour() {
    resetOnboarding();
    setTourOpen(true);
  }

  return (
    <>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="gap-2"
        onClick={restartTour}
      >
        <Compass className="h-4 w-4" aria-hidden />
        Show tour
      </Button>

      <OnboardingTour
        steps={DASHBOARD_TOUR_STEPS}
        open={tourOpen}
        onClose={() => setTourOpen(false)}
      />
    </>
  );
}
