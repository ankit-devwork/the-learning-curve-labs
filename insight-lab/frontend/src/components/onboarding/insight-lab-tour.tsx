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
    body: "Organize course materials into study sets, collaborate with classmates, and use AI tools to study smarter.",
    placement: "center",
  },
  {
    id: "create-set",
    title: "Create a study set",
    body: "Name a set for a class, exam, or project. Each set keeps its own files, progress, and sharing settings.",
    target: '[data-tour="create-set"]',
    placement: "bottom",
  },
  {
    id: "sets-list",
    title: "Open a study set",
    body: "Sets shared with you show a badge. Click any set to upload, chat, quiz, and export.",
    target: '[data-tour="sets-list"]',
    placement: "top",
  },
];

const SET_DETAIL_TOUR: TourStep[] = [
  {
    id: "share",
    title: "Share this study set",
    body: "Open the share panel to invite classmates. Editors and owners can send invite links; viewers can see who has access.",
    target: '[data-tour="share-panel"]',
    placement: "bottom",
  },
  {
    id: "share-members",
    title: "Manage members",
    body: "Owners can change roles (viewer ↔ editor), remove members, or revoke pending invites. Members who are not owners can leave the set.",
    target: '[data-tour="share-members"]',
    placement: "bottom",
  },
  {
    id: "share-invites",
    title: "Pending invites",
    body: "Revoke an invite link anytime before it is accepted — useful if you sent it to the wrong email.",
    target: '[data-tour="share-invites"]',
    placement: "bottom",
  },
  {
    id: "share-leave",
    title: "Leave a shared set",
    body: "Editors and viewers can leave when they no longer need access. Owners delete the set instead.",
    target: '[data-tour="share-leave"]',
    placement: "bottom",
  },
  {
    id: "delete-set",
    title: "Delete study set",
    body: "Owners can permanently delete a study set from the header. You must keep at least one set on your account.",
    target: '[data-tour="delete-set"]',
    placement: "bottom",
  },
  {
    id: "share-invite",
    title: "Invite by email",
    body: "Choose viewer (study only) or editor (upload + generate), then copy the invite link to send to a classmate.",
    target: '[data-tour="share-invite"]',
    placement: "bottom",
  },
  {
    id: "course-pack",
    title: "Course pack",
    body: "One click generates summary, quiz, flashcards, study guide, and audio overview for every ready document.",
    target: '[data-tour="course-pack"]',
    placement: "bottom",
  },
  {
    id: "upload",
    title: "Upload materials",
    body: "Editors can drop PDFs, Word docs, Excel sheets, or CSV files into this study set.",
    target: '[data-tour="upload-btn"]',
    placement: "bottom",
  },
  {
    id: "files",
    title: "Open a ready file",
    body: "When status shows Ready, open a file for chat, quiz, flashcards, exports, and Studio tools.",
    target: '[data-tour="file-list"]',
    placement: "top",
  },
  {
    id: "set-quiz",
    title: "Set-wide adaptive quiz",
    body: "After quizzes on individual documents, generate a set-wide quiz targeting weak topics. Edit and publish before sharing.",
    target: '[data-tour="set-quiz"]',
    placement: "top",
  },
  {
    id: "compare",
    title: "Compare documents",
    body: "Ask one question across multiple files. Similar questions may be answered from cache to save time.",
    target: '[data-tour="compare-docs"]',
    placement: "top",
  },
];

const DOCUMENT_TOUR: TourStep[] = [
  {
    id: "studio",
    title: "Studio tools",
    body: "Generate quizzes, flashcards, study guides, and audio overviews. Export flashcards to Anki CSV and study guides to PDF.",
    target: '[data-tour="studio-panel"]',
    placement: "left",
  },
  {
    id: "quiz-edit",
    title: "Review before you publish",
    body: "Quizzes start as drafts. Edit questions inline, then publish when you are happy with them.",
    target: '[data-tour="quiz-panel"]',
    placement: "top",
  },
];

const EXCEL_TOUR: TourStep[] = [
  {
    id: "excel-canvas",
    title: "Excel canvas",
    body: "Preview data and charts on the left, chat on the right. Ask questions in plain English about your spreadsheet.",
    target: '[data-tour="excel-canvas"]',
    placement: "center",
  },
  {
    id: "excel-export",
    title: "Export charts",
    body: "Download any chart as CSV or PNG for reports and slides.",
    target: '[data-tour="excel-preview"]',
    placement: "top",
  },
];

function stepsForPath(pathname: string): TourStep[] {
  if (/^\/dashboard\/sets\/[^/]+\/documents\/[^/]+$/.test(pathname)) {
    return DOCUMENT_TOUR;
  }
  if (/^\/dashboard\/sets\/[^/]+\/excel\/[^/]+$/.test(pathname)) {
    return EXCEL_TOUR;
  }
  if (/^\/dashboard\/sets\/[^/]+$/.test(pathname)) {
    return SET_DETAIL_TOUR;
  }
  if (pathname.startsWith("/dashboard/sets") || pathname === "/dashboard") {
    return SETS_TOUR;
  }
  return [];
}

/** Steps whose targets may be hidden for viewers — tour skips missing elements gracefully. */
function filterAvailableSteps(steps: TourStep[]): TourStep[] {
  if (typeof document === "undefined") {
    return steps;
  }
  return steps.filter((step) => {
    if (!step.target) {
      return true;
    }
    return Boolean(document.querySelector(step.target));
  });
}

export function InsightLabTourHost() {
  const pathname = usePathname();
  const rawSteps = useMemo(() => stepsForPath(pathname), [pathname]);
  const [steps, setSteps] = useState<TourStep[]>(rawSteps);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setSteps(filterAvailableSteps(rawSteps));
  }, [rawSteps]);

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
      const available = filterAvailableSteps(rawSteps);
      setSteps(available);
      if (available.length > 0) {
        setOpen(true);
      }
    }
    window.addEventListener(TOUR_RESTART_EVENT, handleRestart);
    return () => window.removeEventListener(TOUR_RESTART_EVENT, handleRestart);
  }, [rawSteps]);

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
