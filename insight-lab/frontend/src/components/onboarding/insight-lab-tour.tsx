"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname } from "next/navigation";
import { OnboardingTour, type TourStep } from "@/components/onboarding/onboarding-tour";
import {
  isOnboardingComplete,
  TOUR_RESTART_EVENT,
} from "@/lib/onboarding";

const SETS_TOUR: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to InsightLab",
    body: "Organize course materials into study sets, then chat, quiz, compare, and export insights.",
    placement: "center",
  },
  {
    id: "create-set",
    title: "Create a study set",
    body: "Start by naming a set for a class, exam, or project. Each set keeps its own files and progress.",
    target: '[data-tour="create-set"]',
    placement: "bottom",
  },
  {
    id: "sets-list",
    title: "Open a study set",
    body: "Click a set to upload PDFs or spreadsheets and use AI tools on your materials.",
    target: '[data-tour="sets-list"]',
    placement: "top",
  },
];

const SET_DETAIL_TOUR: TourStep[] = [
  {
    id: "upload",
    title: "Upload materials",
    body: "Drop PDFs, Word docs, Excel sheets, or CSV files into this study set.",
    target: '[data-tour="upload-btn"]',
    placement: "bottom",
  },
  {
    id: "files",
    title: "Open a ready file",
    body: "When status shows Ready, open a file for chat, quiz, flashcards, and the Studio panel.",
    target: '[data-tour="file-list"]',
    placement: "top",
  },
  {
    id: "set-quiz",
    title: "Set-wide adaptive quiz",
    body: "After taking quizzes on documents in this set, generate a quiz that targets your weak topics across all files.",
    target: '[data-tour="set-quiz"]',
    placement: "top",
  },
  {
    id: "compare",
    title: "Compare documents",
    body: "Select two or more ready documents and ask one question across all of them.",
    target: '[data-tour="compare-docs"]',
    placement: "top",
  },
];

const DOCUMENT_TOUR: TourStep[] = [
  {
    id: "studio",
    title: "Studio tools",
    body: "Generate quizzes, flashcards, study guides, audio overviews, and explore the topic graph.",
    target: '[data-tour="studio-panel"]',
    placement: "left",
  },
];

function stepsForPath(pathname: string): TourStep[] {
  if (/^\/dashboard\/sets\/[^/]+\/documents\/[^/]+$/.test(pathname)) {
    return DOCUMENT_TOUR;
  }
  if (/^\/dashboard\/sets\/[^/]+$/.test(pathname)) {
    return SET_DETAIL_TOUR;
  }
  if (pathname.startsWith("/dashboard/sets") || pathname === "/dashboard") {
    return SETS_TOUR;
  }
  return [];
}

export function InsightLabTourHost() {
  const pathname = usePathname();
  const steps = useMemo(() => stepsForPath(pathname), [pathname]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (steps.length === 0) {
      setOpen(false);
      return;
    }

    const timer = window.setTimeout(() => {
      if (!isOnboardingComplete()) {
        setOpen(true);
      }
    }, 700);

    return () => window.clearTimeout(timer);
  }, [pathname, steps.length]);

  useEffect(() => {
    function handleRestart() {
      if (steps.length > 0) {
        setOpen(true);
      }
    }
    window.addEventListener(TOUR_RESTART_EVENT, handleRestart);
    return () => window.removeEventListener(TOUR_RESTART_EVENT, handleRestart);
  }, [steps.length]);

  if (steps.length === 0) {
    return null;
  }

  return (
    <OnboardingTour
      steps={steps}
      open={open}
      onClose={() => setOpen(false)}
    />
  );
}
