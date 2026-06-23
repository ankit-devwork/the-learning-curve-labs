import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { SignOutButton } from "@/components/auth/sign-out-button";
import { DevBackendMeCard } from "@/components/auth/dev-backend-me-card";
import { FileUploadCard } from "@/components/documents/file-upload-card";
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
    <div className="min-h-screen bg-muted/40">
      <header className="border-b bg-background">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-lg font-semibold">InsightLab</h1>
            <p className="text-sm text-muted-foreground">{user.email}</p>
          </div>
          <SignOutButton />
        </div>
      </header>

      <main className="mx-auto max-w-4xl space-y-4 px-4 py-6">
        <FeatureGuide
          title="Welcome to InsightLab"
          steps={[
            "Upload a PDF, Word doc, or spreadsheet using Choose file below.",
            "Click a filename to open it — PDFs and docs get a summary, Q&A, and quiz; spreadsheets get charts and data chat.",
            "Select two or more ready documents on this page to ask one question across all of them.",
          ]}
        />
        <DevBackendMeCard />
        <FileUploadCard />
      </main>
    </div>
  );
}
