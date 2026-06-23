import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { DevBackendMeCard } from "@/components/auth/dev-backend-me-card";
import { FileUploadCard } from "@/components/documents/file-upload-card";
import { DashboardHero } from "@/components/dashboard/dashboard-hero";
import { AppShell } from "@/components/layout/app-shell";
import { FeatureGuide } from "@/components/ui/feature-guide";

export default async function DashboardPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  return (
    <AppShell userEmail={user.email}>
      <div className="space-y-8">
        <DashboardHero />
        <FeatureGuide
          variant="hero"
          title="Quick start"
          steps={[
            "Choose a file to upload — PDF, Word, Excel, or CSV.",
            "When status shows Ready, click the filename to open it.",
            "On the dashboard, select two or more documents to compare them in chat.",
          ]}
        />
        <DevBackendMeCard />
        <FileUploadCard />
      </div>
    </AppShell>
  );
}
