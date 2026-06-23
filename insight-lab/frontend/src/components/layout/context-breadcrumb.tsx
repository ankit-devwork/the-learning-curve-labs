import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

export type BreadcrumbItem = {
  label: string;
  href?: string;
};

type ContextBreadcrumbProps = {
  items: BreadcrumbItem[];
  className?: string;
};

export function ContextBreadcrumb({ items, className }: ContextBreadcrumbProps) {
  if (items.length === 0) {
    return null;
  }

  return (
    <nav aria-label="Breadcrumb" className={cn("flex flex-wrap items-center gap-1 text-sm", className)}>
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        return (
          <span key={`${item.label}-${index}`} className="inline-flex items-center gap-1">
            {index > 0 ? <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/60" aria-hidden /> : null}
            {item.href && !isLast ? (
              <Link href={item.href} className="text-muted-foreground hover:text-primary">
                {item.label}
              </Link>
            ) : (
              <span className={isLast ? "font-medium text-foreground" : "text-muted-foreground"}>
                {item.label}
              </span>
            )}
          </span>
        );
      })}
    </nav>
  );
}
