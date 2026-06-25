"use client";

import Link from "next/link";
import { BookOpen, Compass } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { requestTourRestart } from "@/lib/onboarding";

const sections = [
  {
    title: "Start here",
    items: [
      {
        heading: "Create a study sheet",
        body: "Go to Study sheets, name your class or project, and click Create. Each sheet holds its own files, sharing, and progress.",
      },
      {
        heading: "Upload and wait for Ready",
        body: "Upload PDF, Word, Excel, or CSV files. Most tools unlock when status shows Ready (about a minute).",
      },
      {
        heading: "Open a file",
        body: "Click any Ready file to chat, quiz, and use Studio tools. Spreadsheets open the Excel workspace; documents open the reading workspace.",
      },
    ],
  },
  {
    title: "Guided study plan vs practice quiz",
    items: [
      {
        heading: "Guided study plan",
        body: "A checklist that walks you through steps (brief, flashcards, quiz) and saves progress. Click Start my plan once, then use Go and Mark done on each step. Optional — you can always use files directly without starting a plan.",
      },
      {
        heading: "Practice quiz (whole sheet)",
        body: "An actual quiz covering weak topics across all files. Take a quiz on at least one file first so the app knows what to practice. This is not the same as the guided plan.",
      },
    ],
  },
  {
    title: "Study sheet tools",
    items: [
      {
        heading: "Learning path",
        body: "AI suggests an order to study topics. Generate a path, then start your guided plan to follow it.",
      },
      {
        heading: "Concept graph",
        body: "See topics across all files. Colors show mastery; filter weak concepts to focus review.",
      },
      {
        heading: "Team chat",
        body: "Plain-text messages for sheet members. Coordinate with classmates — no links or attachments.",
      },
      {
        heading: "Compare",
        body: "Pick multiple ready files and ask one question. Get a combined answer with citations.",
      },
      {
        heading: "Course pack",
        body: "Generate summary, quiz, flashcards, study guide, and audio for every ready file at once.",
      },
      {
        heading: "Share",
        body: "Invite viewers (study only) or editors (upload and generate). Copy the invite link and send it yourself.",
      },
    ],
  },
  {
    title: "Document & Excel workspaces",
    items: [
      {
        heading: "Chat",
        body: "Ask questions grounded in your file. Click citations to see source excerpts.",
      },
      {
        heading: "Studio tabs",
        body: "Brief, study plan, quiz, flashcards, study guide, infographic, audio, and concept graph — all from the Studio panel on the right (or tabs on mobile).",
      },
      {
        heading: "Excel tabs",
        body: "Insights, preview, charts, chart builder, concept graph, and quiz — built for spreadsheet study.",
      },
    ],
  },
  {
    title: "Roles",
    items: [
      {
        heading: "Viewer",
        body: "Read, chat, quiz, and study. Cannot upload or invite.",
      },
      {
        heading: "Editor",
        body: "Upload, generate artifacts, edit quizzes, and invite members.",
      },
      {
        heading: "Owner",
        body: "Full control including delete sheet and manage all members.",
      },
    ],
  },
];

export function UserGuidePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-primary">
            <BookOpen className="h-5 w-5" aria-hidden />
            <span className="text-sm font-medium">User guide</span>
          </div>
          <h1 className="font-display text-2xl font-semibold sm:text-3xl">How to use InsightLab</h1>
          <p className="mt-2 text-muted-foreground">
            Simple steps for studying documents and spreadsheets with your class or team.
          </p>
        </div>
        <Button type="button" variant="outline" className="gap-2" onClick={() => requestTourRestart()}>
          <Compass className="h-4 w-4" aria-hidden />
          Show tour
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick start</CardTitle>
          <CardDescription>Most people follow this path</CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="list-decimal space-y-2 pl-5 text-sm text-muted-foreground">
            <li>Create a study sheet and upload files.</li>
            <li>Open a Ready file — read the Brief, then try Chat or Quiz.</li>
            <li>Optional: start a guided study plan to track steps.</li>
            <li>After file quizzes, use Practice quiz (whole sheet) for weak topics.</li>
            <li>Share the sheet and use Team chat to study together.</li>
          </ol>
        </CardContent>
      </Card>

      {sections.map((section) => (
        <section key={section.title} className="space-y-3">
          <h2 className="text-lg font-semibold">{section.title}</h2>
          <div className="space-y-3">
            {section.items.map((item) => (
              <Card key={item.heading}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{item.heading}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-muted-foreground">{item.body}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      ))}

      <p className="text-sm text-muted-foreground">
        For the full manual (exports, troubleshooting, workflows), see{" "}
        <Link href="https://github.com/ankit-devwork/the-learning-curve-labs/blob/main/insight-lab/docs/USER-MANUAL.md" className="text-primary underline-offset-4 hover:underline" target="_blank" rel="noopener noreferrer">
          USER-MANUAL.md
        </Link>{" "}
        in the repository.
      </p>
    </div>
  );
}
