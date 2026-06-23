"use client";

import { FileSpreadsheet, FileText, MessagesSquare } from "lucide-react";
import { DashboardOnboarding } from "@/components/onboarding/dashboard-onboarding";

const features = [
  {
    icon: FileText,
    title: "Documents",
    description: "Summaries, Q&A with citations, and quizzes from PDFs and Word files.",
  },
  {
    icon: FileSpreadsheet,
    title: "Spreadsheets",
    description: "Auto charts, insights, and natural-language questions about your data.",
  },
  {
    icon: MessagesSquare,
    title: "Compare",
    description: "Select multiple documents and ask one question across all of them.",
  },
];

export function DashboardHero() {
  return (
    <section className="space-y-6" data-tour="workspace-hero">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Your workspace</h2>
          <p className="mt-1 max-w-2xl text-muted-foreground">
            Upload learning materials or datasets, then explore them with AI-powered summaries,
            chat, and quizzes.
          </p>
        </div>
        <DashboardOnboarding />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {features.map(({ icon: Icon, title, description }) => (
          <div
            key={title}
            className="rounded-xl border bg-card p-4 shadow-sm transition-shadow hover:shadow-md"
          >
            <span className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Icon className="h-4 w-4" aria-hidden />
            </span>
            <p className="font-medium">{title}</p>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
