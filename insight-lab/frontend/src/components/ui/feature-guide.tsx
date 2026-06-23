import { cn } from "@/lib/utils";

type FeatureGuideProps = {
  title?: string;
  steps: string[];
  className?: string;
};

export function FeatureGuide({ title = "How to use", steps, className }: FeatureGuideProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-muted/40 p-4 text-sm",
        className,
      )}
    >
      <p className="font-medium text-foreground">{title}</p>
      <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-muted-foreground">
        {steps.map((step, index) => (
          <li key={index}>{step}</li>
        ))}
      </ol>
    </div>
  );
}
