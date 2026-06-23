"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { StudyGuideContent } from "@/lib/api";
import { downloadStudyGuidePdf } from "@/lib/export-utils";

export function StudyGuideView({
  title,
  content,
}: {
  title: string;
  content: StudyGuideContent;
}) {
  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <CardTitle className="text-lg">{title}</CardTitle>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => downloadStudyGuidePdf(title, content)}
        >
          Export PDF
        </Button>
      </CardHeader>
      <CardContent className="space-y-6 text-sm">
        <section>
          <h4 className="font-medium">Overview</h4>
          <p className="mt-2 whitespace-pre-wrap leading-relaxed text-muted-foreground">
            {content.overview}
          </p>
        </section>

        {content.key_terms.length > 0 ? (
          <section>
            <h4 className="font-medium">Key terms</h4>
            <dl className="mt-2 space-y-2">
              {content.key_terms.map((item) => (
                <div key={item.term} className="rounded-md border px-3 py-2">
                  <dt className="font-medium">{item.term}</dt>
                  <dd className="mt-1 text-muted-foreground">{item.definition}</dd>
                </div>
              ))}
            </dl>
          </section>
        ) : null}

        {content.sections.map((section) => (
          <section key={section.heading}>
            <h4 className="font-medium">{section.heading}</h4>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
              {section.bullets.map((bullet) => (
                <li key={bullet}>{bullet}</li>
              ))}
            </ul>
          </section>
        ))}

        {content.sample_questions.length > 0 ? (
          <section>
            <h4 className="font-medium">Sample questions</h4>
            <ul className="mt-2 list-decimal space-y-1 pl-5 text-muted-foreground">
              {content.sample_questions.map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </CardContent>
    </Card>
  );
}
