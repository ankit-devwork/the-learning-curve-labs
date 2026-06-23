import { ExcelDetailClient } from "@/components/excel/excel-detail-client";
import Link from "next/link";

export default async function ExcelWorkspacePage({
  params,
}: {
  params: Promise<{ setId: string; docId: string }>;
}) {
  const { setId, docId } = await params;
  return (
    <div className="space-y-4">
      <Link
        href={`/dashboard/sets/${setId}`}
        className="text-sm text-muted-foreground hover:text-primary"
      >
        ← Back to study set
      </Link>
      <ExcelDetailClient documentId={docId} />
    </div>
  );
}
