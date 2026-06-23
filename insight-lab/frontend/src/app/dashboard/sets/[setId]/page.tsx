import { StudySetDetailClient } from "@/components/workspace/study-set-detail-client";

export default async function StudySetPage({
  params,
}: {
  params: Promise<{ setId: string }>;
}) {
  const { setId } = await params;
  return <StudySetDetailClient setId={setId} />;
}
