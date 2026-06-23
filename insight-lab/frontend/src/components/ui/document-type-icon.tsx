import { FileSpreadsheet, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

export function DocumentTypeIcon({
  fileType,
  className,
}: {
  fileType: string;
  className?: string;
}) {
  const Icon = fileType === "excel" ? FileSpreadsheet : FileText;
  return (
    <span
      className={cn(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary",
        className,
      )}
    >
      <Icon className="h-4 w-4" aria-hidden />
    </span>
  );
}
