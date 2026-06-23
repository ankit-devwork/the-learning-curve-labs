import { ExcelDetailClient } from "@/components/excel/excel-detail-client";

export default async function ExcelWorkspacePage({
  params,
}: {
  params: Promise<{ setId: string; docId: string }>;
}) {
  const { setId, docId } = await params;
  return <ExcelDetailClient documentId={docId} setId={setId} />;
}
