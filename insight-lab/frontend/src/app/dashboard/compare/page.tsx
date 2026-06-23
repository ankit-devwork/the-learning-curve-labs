import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { CompareWorkspaceClient } from "@/components/workspace/compare-workspace-client";

export default async function ComparePage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    redirect("/login");
  }
  return <CompareWorkspaceClient />;
}
