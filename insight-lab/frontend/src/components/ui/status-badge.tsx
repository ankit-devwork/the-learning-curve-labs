import { cn } from "@/lib/utils";

const styles: Record<string, string> = {
  ready: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200",
  pending: "bg-amber-100 text-amber-900 dark:bg-amber-950/50 dark:text-amber-100",
  processing: "bg-blue-100 text-blue-800 dark:bg-blue-950/50 dark:text-blue-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-950/50 dark:text-red-200",
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const key = status.toLowerCase();
  const label = status.charAt(0).toUpperCase() + status.slice(1);

  return (
    <span
      className={cn(
        "inline-flex shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium capitalize",
        styles[key] ?? "bg-muted text-muted-foreground",
        className,
      )}
    >
      {label}
    </span>
  );
}
