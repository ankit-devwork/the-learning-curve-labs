"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { SlideDeckContent } from "@/lib/api";
import { downloadTextFile } from "@/lib/export-utils";

type SlideDeckViewProps = {
  title: string;
  content: SlideDeckContent;
  slideDeckId?: string;
  accessToken?: string | null;
};

export function SlideDeckView({ title, content, slideDeckId, accessToken }: SlideDeckViewProps) {
  const slides = content.slides ?? [];

  async function exportMarkdown() {
    if (slideDeckId && accessToken) {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "/api-backend"}/slide-decks/${slideDeckId}/export/markdown`,
        { headers: { Authorization: `Bearer ${accessToken}` } },
      );
      if (response.ok) {
        downloadTextFile(await response.text(), `${title.replace(/\s+/g, "-")}-slides.md`);
        return;
      }
    }
    const markdown = [
      `# ${title}`,
      "",
      ...slides.flatMap((slide) => [
        `## Slide ${slide.slide_number}: ${slide.title}`,
        ...slide.bullets.map((bullet) => `- ${bullet}`),
        slide.speaker_notes ? `\n_Speaker notes:_ ${slide.speaker_notes}\n` : "",
      ]),
    ].join("\n");
    downloadTextFile(markdown, `${title.replace(/\s+/g, "-")}-slides.md`);
  }

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription>{slides.length} slides · export to Markdown for Google Slides or PowerPoint</CardDescription>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => void exportMarkdown()}>
          Export .md
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {slides.map((slide) => (
          <div key={slide.slide_number} className="rounded-xl border bg-muted/20 p-4">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Slide {slide.slide_number}
            </p>
            <h3 className="mt-1 text-base font-semibold">{slide.title}</h3>
            <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {slide.bullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
            {slide.speaker_notes ? (
              <p className="mt-3 text-sm italic text-muted-foreground">{slide.speaker_notes}</p>
            ) : null}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
