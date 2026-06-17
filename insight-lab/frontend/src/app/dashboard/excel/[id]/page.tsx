import { ExcelDetailClient } from "@/components/excel/excel-detail-client";

export default async function ExcelDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <main className="container mx-auto max-w-4xl px-4 py-8">
      <ExcelDetailClient documentId={id} />
    </main>
  );
}
