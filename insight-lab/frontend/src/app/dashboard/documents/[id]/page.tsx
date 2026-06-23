import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";

export default async function DocumentDetailRedirectPage({
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

  const { data: document } = await supabase
    .from("documents")
    .select("workspace_id")
    .eq("id", id)
    .eq("owner_id", user.id)
    .maybeSingle();

  if (!document?.workspace_id) {
    redirect("/dashboard/sets");
  }

  redirect(`/dashboard/sets/${document.workspace_id}/documents/${id}`);
}
