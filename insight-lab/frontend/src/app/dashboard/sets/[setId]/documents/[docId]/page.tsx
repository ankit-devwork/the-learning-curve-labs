import { DocumentWorkspaceClient } from "@/components/workspace/document-workspace-client";

export default async function DocumentWorkspacePage({
  params,
}: {
  params: Promise<{ setId: string; docId: string }>;
}) {
  const { setId, docId } = await params;
  return <DocumentWorkspaceClient setId={setId} documentId={docId} />;
}
