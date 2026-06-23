import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { ExcelDetailClient } from "@/components/excel/excel-detail-client";
import { AppShell } from "@/components/layout/app-shell";

export default async function ExcelDetailPage({
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
    <AppShell userEmail={user.email} wide>
      <ExcelDetailClient documentId={id} />
    </AppShell>
  );
}
