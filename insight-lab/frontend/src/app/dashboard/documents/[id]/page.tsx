import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { DocumentDetailClient } from "@/components/documents/document-detail-client";

export default async function DocumentDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  return (
    <div className="min-h-screen bg-muted/40">
      <main className="mx-auto max-w-6xl p-4 py-6">
        <DocumentDetailClient documentId={id} />
      </main>
    </div>
  );
}
