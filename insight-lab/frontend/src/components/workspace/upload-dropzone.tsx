"use client";

import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

type UploadDropzoneProps = {
  accept?: string;
  disabled?: boolean;
  uploading?: boolean;
  onFileSelected: (file: File) => void;
  className?: string;
};

export function UploadDropzone({
  accept,
  disabled,
  uploading,
  onFileSelected,
  className,
}: UploadDropzoneProps) {
  return (
    <label
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed bg-muted/20 px-6 py-10 text-center transition-colors",
        !disabled && "hover:border-primary/40 hover:bg-primary/[0.03]",
        disabled && "cursor-not-allowed opacity-60",
        className,
      )}
    >
      <Upload className="h-8 w-8 text-muted-foreground/70" aria-hidden />
      <p className="mt-3 font-medium">{uploading ? "Uploading…" : "Drop a file here"}</p>
      <p className="mt-1 text-sm text-muted-foreground">
        PDF, Word, Excel, or CSV — or click to browse
      </p>
      <input
        type="file"
        accept={accept}
        disabled={disabled || uploading}
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onFileSelected(file);
            event.target.value = "";
          }
        }}
      />
    </label>
  );
}
