"use client";

import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { UploadGuidance, type UploadGuidanceConfig } from "@/components/workspace/upload-guidance";

type UploadDropzoneProps = {
  accept?: string;
  disabled?: boolean;
  uploading?: boolean;
  guidance?: UploadGuidanceConfig | null;
  acknowledged?: boolean;
  onAcknowledgedChange?: (value: boolean) => void;
  onFileSelected: (file: File) => void;
  className?: string;
};

export function UploadDropzone({
  accept,
  disabled,
  uploading,
  guidance,
  acknowledged = false,
  onAcknowledgedChange,
  onFileSelected,
  className,
}: UploadDropzoneProps) {
  const requiresAck = Boolean(guidance?.require_acknowledgment);
  const uploadBlocked = Boolean(disabled || uploading || (requiresAck && !acknowledged));

  return (
    <div className={cn("space-y-4", className)}>
      {guidance ? (
        <UploadGuidance
          guidance={guidance}
          acknowledged={acknowledged}
          onAcknowledgedChange={onAcknowledgedChange ?? (() => undefined)}
        />
      ) : null}
      <label
        data-tour="upload-btn"
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed bg-muted/20 px-6 py-10 text-center transition-colors",
          !uploadBlocked && "hover:border-primary/40 hover:bg-primary/[0.03]",
          uploadBlocked && "cursor-not-allowed opacity-60",
        )}
      >
        <Upload className="h-8 w-8 text-muted-foreground/70" aria-hidden />
        <p className="mt-3 font-medium">{uploading ? "Uploading…" : "Drop a file here"}</p>
        <p className="mt-1 text-sm text-muted-foreground">
          PDF, Word, Excel, or CSV — or click to browse
        </p>
        {requiresAck && !acknowledged ? (
          <p className="mt-2 text-xs text-muted-foreground">
            Confirm the privacy notice above before uploading.
          </p>
        ) : null}
        <input
          type="file"
          accept={accept}
          disabled={uploadBlocked}
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
    </div>
  );
}
