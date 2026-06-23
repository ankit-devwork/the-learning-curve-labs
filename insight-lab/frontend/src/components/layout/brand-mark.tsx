import { Sparkles } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

type BrandMarkProps = {
  className?: string;
  inverted?: boolean;
  href?: string | null;
};

export function BrandMark({ className, inverted, href = "/dashboard" }: BrandMarkProps) {
  const content = (
    <>
      <span
        className={cn(
          "flex h-9 w-9 items-center justify-center rounded-lg shadow-sm",
          inverted ? "bg-white/15 text-white" : "bg-primary text-primary-foreground",
        )}
      >
        <Sparkles className="h-5 w-5" aria-hidden />
      </span>
      <span className="leading-tight">
        <span className={cn("block text-base font-semibold tracking-tight", inverted && "text-white")}>
          InsightLab
        </span>
        <span
          className={cn(
            "block text-xs",
            inverted ? "text-white/75" : "text-muted-foreground",
          )}
        >
          Documents · Data · Quizzes
        </span>
      </span>
    </>
  );

  if (href != null) {
    return (
      <Link href={href} className={cn("flex items-center gap-3", className)}>
        {content}
      </Link>
    );
  }

  return <div className={cn("flex items-center gap-3", className)}>{content}</div>;
}
