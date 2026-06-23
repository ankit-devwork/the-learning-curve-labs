import { cn } from "@/lib/utils";
import { Lightbulb } from "lucide-react";

type FeatureGuideProps = {
  title?: string;
  steps: string[];
  className?: string;
  variant?: "default" | "hero";
};

export function FeatureGuide({
  title = "How to use",
  steps,
  className,
  variant = "default",
}: FeatureGuideProps) {
  return (
    <div
      className={cn(
        "rounded-xl border text-sm",
        variant === "hero"
          ? "border-primary/15 bg-gradient-to-br from-primary/[0.06] via-background to-background p-6 shadow-sm"
          : "border-border/80 bg-card p-4 shadow-sm",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        {variant === "hero" ? (
          <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Lightbulb className="h-4 w-4" aria-hidden />
          </span>
        ) : null}
        <div className="min-w-0 flex-1">
          <p className="font-medium text-foreground">{title}</p>
          <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-muted-foreground">
            {steps.map((step, index) => (
              <li key={index} className="leading-relaxed">
                {step}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}
