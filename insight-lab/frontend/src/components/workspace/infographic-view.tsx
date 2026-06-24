"use client";

import { useRef } from "react";
import { toPng } from "html-to-image";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { InfographicBlock, InfographicContent } from "@/lib/api";
import { cn } from "@/lib/utils";

const THEME_STYLES: Record<
  InfographicContent["theme"],
  { header: string; accent: string; stat: string; quote: string }
> = {
  blue: {
    header: "from-blue-600 to-blue-500",
    accent: "text-blue-600 dark:text-blue-400",
    stat: "border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-950/40",
    quote: "border-blue-300/60 bg-blue-500/5",
  },
  violet: {
    header: "from-violet-600 to-violet-500",
    accent: "text-violet-600 dark:text-violet-400",
    stat: "border-violet-200 bg-violet-50 dark:border-violet-900 dark:bg-violet-950/40",
    quote: "border-violet-300/60 bg-violet-500/5",
  },
  emerald: {
    header: "from-emerald-600 to-emerald-500",
    accent: "text-emerald-600 dark:text-emerald-400",
    stat: "border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/40",
    quote: "border-emerald-300/60 bg-emerald-500/5",
  },
  amber: {
    header: "from-amber-600 to-amber-500",
    accent: "text-amber-600 dark:text-amber-400",
    stat: "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/40",
    quote: "border-amber-300/60 bg-amber-500/5",
  },
  rose: {
    header: "from-rose-600 to-rose-500",
    accent: "text-rose-600 dark:text-rose-400",
    stat: "border-rose-200 bg-rose-50 dark:border-rose-900 dark:bg-rose-950/40",
    quote: "border-rose-300/60 bg-rose-500/5",
  },
  cyan: {
    header: "from-cyan-600 to-cyan-500",
    accent: "text-cyan-600 dark:text-cyan-400",
    stat: "border-cyan-200 bg-cyan-50 dark:border-cyan-900 dark:bg-cyan-950/40",
    quote: "border-cyan-300/60 bg-cyan-500/5",
  },
};

function InfographicBlockView({
  block,
  theme,
}: {
  block: InfographicBlock;
  theme: InfographicContent["theme"];
}) {
  const styles = THEME_STYLES[theme];

  if (block.type === "stat") {
    return (
      <div className={cn("rounded-xl border p-4 text-center", styles.stat)}>
        <p className={cn("text-3xl font-bold tracking-tight", styles.accent)}>{block.value}</p>
        <p className="mt-1 text-sm font-medium">{block.label}</p>
        {block.caption ? <p className="mt-1 text-xs text-muted-foreground">{block.caption}</p> : null}
      </div>
    );
  }

  if (block.type === "bullets") {
    return (
      <div className="rounded-xl border bg-card p-4">
        <h4 className={cn("text-sm font-semibold", styles.accent)}>{block.heading}</h4>
        <ul className="mt-2 space-y-1.5 text-sm text-muted-foreground">
          {block.items.map((item) => (
            <li key={item} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (block.type === "comparison") {
    return (
      <div className="rounded-xl border bg-card p-4">
        <h4 className={cn("mb-3 text-sm font-semibold", styles.accent)}>{block.heading}</h4>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border bg-muted/20 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide">{block.left_title}</p>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {block.left_items.map((item) => (
                <li key={item}>• {item}</li>
              ))}
            </ul>
          </div>
          <div className="rounded-lg border bg-muted/20 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide">{block.right_title}</p>
            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
              {block.right_items.map((item) => (
                <li key={item}>• {item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <blockquote className={cn("rounded-xl border-l-4 p-4 italic", styles.quote)}>
      <p className="text-sm leading-relaxed">&ldquo;{block.text}&rdquo;</p>
      {block.attribution ? (
        <footer className="mt-2 text-xs not-italic text-muted-foreground">— {block.attribution}</footer>
      ) : null}
    </blockquote>
  );
}

export function InfographicView({
  title,
  content,
}: {
  title: string;
  content: InfographicContent;
}) {
  const exportRef = useRef<HTMLDivElement>(null);
  const theme = THEME_STYLES[content.theme] ?? THEME_STYLES.blue;

  async function exportPng() {
    if (!exportRef.current) {
      return;
    }
    const dataUrl = await toPng(exportRef.current, { pixelRatio: 2, cacheBust: true });
    const link = document.createElement("a");
    link.download = `${title.replace(/\s+/g, "-").toLowerCase()}-infographic.png`;
    link.href = dataUrl;
    link.click();
  }

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <CardTitle className="text-lg">{title}</CardTitle>
        <Button type="button" variant="outline" size="sm" onClick={() => void exportPng()}>
          Export PNG
        </Button>
      </CardHeader>
      <CardContent>
        <div ref={exportRef} className="overflow-hidden rounded-xl border bg-background">
          <div className={cn("bg-gradient-to-r px-6 py-8 text-white", theme.header)}>
            <h3 className="text-2xl font-bold tracking-tight">{content.title}</h3>
            {content.subtitle ? <p className="mt-1 text-sm text-white/90">{content.subtitle}</p> : null}
          </div>
          <div className="grid gap-4 p-4 sm:grid-cols-2">
            {content.blocks.map((block, index) => (
              <div
                key={`${block.type}-${index}`}
                className={cn(block.type === "stat" || block.type === "quote" ? "sm:col-span-2" : "")}
              >
                <InfographicBlockView block={block} theme={content.theme} />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
