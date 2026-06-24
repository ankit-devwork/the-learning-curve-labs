"use client";

import {
  BookOpen,
  Brain,
  Image,
  Layers,
  MessageSquare,
  Network,
  Sparkles,
  Volume2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type StudioPanelProps = {
  ready: boolean;
  busy?: boolean;
  canEdit?: boolean;
  onGenerateQuiz: () => void;
  onGenerateFlashcards: () => void;
  onGenerateStudyGuide: () => void;
  onGenerateAudioOverview: () => void;
  onGenerateInfographic: () => void;
  onOpenMindMap: () => void;
  onFocusAsk: () => void;
  className?: string;
};

const actions = [
  { id: "ask", label: "Ask", icon: MessageSquare, description: "Chat with this document", requiresEdit: false },
  { id: "quiz", label: "Quiz", icon: Brain, description: "Generate practice questions", requiresEdit: true },
  { id: "flashcards", label: "Flashcards", icon: Layers, description: "Term and definition cards", requiresEdit: true },
  { id: "guide", label: "Study guide", icon: BookOpen, description: "Structured overview", requiresEdit: true },
  { id: "infographic", label: "Infographic", icon: Image, description: "Visual summary card", requiresEdit: true },
  { id: "audio", label: "Audio overview", icon: Volume2, description: "Listen to a narrated summary", requiresEdit: true },
  { id: "mindmap", label: "Mind map", icon: Share2, description: "Explore connected concepts", requiresEdit: false },
] as const;

export function StudioPanel({
  ready,
  busy,
  canEdit = true,
  onGenerateQuiz,
  onGenerateFlashcards,
  onGenerateStudyGuide,
  onGenerateAudioOverview,
  onGenerateInfographic,
  onOpenMindMap,
  onFocusAsk,
  className,
}: StudioPanelProps) {
  function handleAction(actionId: (typeof actions)[number]["id"]) {
    if (!ready || busy) {
      return;
    }
    switch (actionId) {
      case "ask":
        onFocusAsk();
        break;
      case "quiz":
        onGenerateQuiz();
        break;
      case "flashcards":
        onGenerateFlashcards();
        break;
      case "guide":
        onGenerateStudyGuide();
        break;
      case "infographic":
        onGenerateInfographic();
        break;
      case "audio":
        onGenerateAudioOverview();
        break;
      case "mindmap":
        onOpenMindMap();
        break;
    }
  }

  return (
    <Card className={cn("notebook-surface border-0 shadow-none", className)} data-tour="studio-panel">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" aria-hidden />
          <CardTitle className="text-base">Studio</CardTitle>
        </div>
        <CardDescription>One-click learning tools from this document</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-2">
        {actions.map(({ id, label, icon: Icon, description, requiresEdit }) => (
          <Button
            key={id}
            type="button"
            variant="outline"
            className="h-auto justify-start gap-3 px-3 py-3 text-left"
            disabled={!ready || busy || (requiresEdit && !canEdit)}
            onClick={() => handleAction(id)}
          >
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Icon className="h-4 w-4" aria-hidden />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium">{label}</span>
              <span className="block text-xs font-normal text-muted-foreground">{description}</span>
            </span>
          </Button>
        ))}
      </CardContent>
    </Card>
  );
}
