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
    body: "Organize course materials into study sheets, collaborate with classmates, and use AI tools to study smarter. Open User guide in the sidebar anytime.",
    placement: "center",
  },
  {
    id: "create-set",
    title: "Create a study sheet",
    body: "Name a sheet for a class, exam, or project. Each sheet keeps its own files, progress, and sharing settings.",
    target: '[data-tour="create-set"]',
    placement: "bottom",
  },
  {
    id: "sets-list",
    title: "Open a study sheet",
    body: "Sheets shared with you show a badge. Click any sheet to upload, chat, quiz, and export.",
    target: '[data-tour="sets-list"]',
    placement: "top",
  },
];

const SET_DETAIL_TOUR: TourStep[] = [
  {
    id: "welcome-set",
    title: "Your study sheet home",
    body: "This page holds all materials and team tools. Upload files first, then use chat, quizzes, and guided plans below.",
    placement: "center",
  },
  {
    id: "upload",
    title: "Upload materials",
    body: "Add PDFs, Word docs, Excel spreadsheets, or CSV files. Wait until status shows Ready before studying.",
    target: '[data-tour="upload-btn"]',
    placement: "bottom",
  },
  {
    id: "files",
    title: "Open a ready file",
    body: "Click a Ready file for chat, quiz, flashcards, and Studio tools. Spreadsheets open the Excel workspace.",
    target: '[data-tour="file-list"]',
    placement: "top",
  },
  {
    id: "progress",
    title: "Your progress",
    body: "See quiz scores, mastery, and what to study next. Suggestions link you to the right tool.",
    target: '[data-tour="progress-dashboard"]',
    placement: "bottom",
  },
  {
    id: "study-session",
    title: "Guided study plan",
    body: "Optional step-by-step plan across all files. Click Start my plan once, then Go and Mark done on each step. You can use the app without starting a plan.",
    target: '[data-tour="study-session"]',
    placement: "bottom",
  },
  {
    id: "learning-path",
    title: "Learning path",
    body: "AI suggests an order to study topics. Generate a path, then start your guided plan to follow it.",
    target: '[data-tour="learning-path"]',
    placement: "bottom",
  },
  {
    id: "concept-graph",
    title: "Concept graph",
    body: "Topics across every file, colored by quiz mastery. Toggle weak concepts to focus review.",
    target: '[data-tour="concept-graph"]',
    placement: "bottom",
  },
  {
    id: "team-chat",
    title: "Team chat",
    body: "Plain-text chat for sheet members. Open the corner panel to coordinate with classmates — messages update live.",
    target: '[data-tour="team-chat"]',
    placement: "left",
  },
  {
    id: "set-quiz",
    title: "Practice quiz (whole sheet)",
    body: "A real quiz on weak topics across all files — different from the guided plan checklist. Take a file quiz first so weak topics appear.",
    target: '[data-tour="set-quiz"]',
    placement: "top",
  },
  {
    id: "compare",
    title: "Compare documents",
    body: "Ask one question across multiple ready files on this sheet. Answers include citations per file.",
    target: '[data-tour="compare-docs"]',
    placement: "top",
  },
  {
    id: "classroom-analytics",
    title: "Classroom analytics",
    body: "Editors see quiz averages and member activity — useful for teachers monitoring a class.",
    target: '[data-tour="classroom-analytics"]',
    placement: "top",
  },
  {
    id: "share",
    title: "Share this study sheet",
    body: "Invite classmates as viewers (study only) or editors (upload + generate). Copy the invite link from the Share drawer.",
    target: '[data-tour="share-btn"]',
    placement: "bottom",
  },
  {
    id: "course-pack",
    title: "Course pack",
    body: "Generate summary, quiz, flashcards, study guide, and audio for every ready document in one batch.",
    target: '[data-tour="course-pack-btn"]',
    placement: "bottom",
  },
];

const DOCUMENT_TOUR: TourStep[] = [
  {
    id: "welcome-doc",
    title: "Document workspace",
    body: "Chat with this source, then use Studio tabs for brief, study plan, quiz, flashcards, and more.",
    placement: "center",
  },
  {
    id: "document-chat",
    title: "Chat with this source",
    body: "Ask questions grounded in the document. Click citation chips to jump to the source excerpt.",
    target: '[data-tour="document-chat"]',
    placement: "bottom",
  },
  {
    id: "studio",
    title: "Studio tools",
    body: "Brief, study plan, quiz, flashcards, study guide, audio, infographic, and concept graph — all from this panel.",
    target: '[data-tour="studio-panel"]',
    placement: "left",
  },
  {
    id: "document-session",
    title: "Guided study plan (this file)",
    body: "Optional checklist: brief → flashcards → quiz. Start my plan to save progress, or open tabs directly without tracking.",
    target: '[data-tour="document-session"]',
    placement: "top",
  },
  {
    id: "quiz-edit",
    title: "Quiz on this file",
    body: "Generate practice questions, edit drafts, publish when ready, and track topic mastery.",
    target: '[data-tour="quiz-panel"]',
    placement: "top",
  },
  {
    id: "document-concepts",
    title: "Concept graph",
    body: "Topics from this document colored by your quiz scores. Focus on weak areas.",
    target: '[data-tour="document-concepts"]',
    placement: "top",
  },
  {
    id: "audio-overview",
    title: "Audio overview",
    body: "Generate a narrated summary to listen while reviewing.",
    target: '[data-tour="audio-overview"]',
    placement: "top",
  },
];

const EXCEL_TOUR: TourStep[] = [
  {
    id: "excel-canvas",
    title: "Excel workspace",
    body: "Chat about your data first, then use tabs for insights, preview, charts, concept graph, and quiz.",
    target: '[data-tour="excel-canvas"]',
    placement: "center",
  },
  {
    id: "excel-tools",
    title: "Spreadsheet tools",
    body: "Quick stats, re-analyze after changes, and a shortcut back to chat.",
    target: '[data-tour="excel-tools"]',
    placement: "left",
  },
  {
    id: "excel-preview",
    title: "Preview & export",
    body: "Browse sample rows and download charts as CSV or PNG for reports.",
    target: '[data-tour="excel-preview"]',
    placement: "top",
  },
  {
    id: "excel-concepts",
    title: "Concept graph",
    body: "Topics extracted from insights and charts. Update the graph after analysis.",
    target: '[data-tour="excel-concepts"]',
    placement: "top",
  },
  {
    id: "excel-quiz",
    title: "Spreadsheet quiz",
    body: "Practice questions based on your data summary and charts.",
    target: '[data-tour="excel-quiz"]',
    placement: "top",
  },
];

const COMPARE_TOUR: TourStep[] = [
  {
    id: "welcome-compare",
    title: "Compare documents",
    body: "Pick a study sheet, select two or more ready files, and ask one question for a combined answer.",
    placement: "center",
  },
  {
    id: "compare-panel",
    title: "Multi-document chat",
    body: "Select files, ask a question, and read citations from each source. Great for cross-reading lectures.",
    target: '[data-tour="compare-docs"]',
    placement: "top",
  },
];

const HELP_TOUR: TourStep[] = [
  {
    id: "welcome-help",
    title: "User guide",
    body: "Plain-language help for every feature. Click Show tour on any page for a walkthrough of what you see there.",
    placement: "center",
  },
];

function stepsForPath(pathname: string): TourStep[] {
  if (pathname === "/dashboard/help") {
    return HELP_TOUR;
  }
  if (pathname === "/dashboard/compare") {
    return COMPARE_TOUR;
  }
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
