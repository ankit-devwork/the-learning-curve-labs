import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { UserGuidePage } from "@/components/help/user-guide-page";

export default async function HelpPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    redirect("/login");
  }

  return <UserGuidePage />;
}
