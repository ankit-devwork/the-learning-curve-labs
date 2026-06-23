import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { CompareWorkspaceClient } from "@/components/workspace/compare-workspace-client";

type ComparePageProps = {
  searchParams: Promise<{ setId?: string }>;
};

export default async function ComparePage({ searchParams }: ComparePageProps) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    redirect("/login");
  }

  const params = await searchParams;
  return <CompareWorkspaceClient initialSetId={params.setId} />;
}
