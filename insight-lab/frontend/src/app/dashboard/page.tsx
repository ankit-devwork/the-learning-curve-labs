import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { SignOutButton } from "@/components/auth/sign-out-button";
import { DevBackendMeCard } from "@/components/auth/dev-backend-me-card";
import { FileUploadCard } from "@/components/documents/file-upload-card";

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
        <p className="text-sm text-muted-foreground">
          Upload a PDF or spreadsheet, then open it to summarize, ask questions, or take a quiz.
          Select multiple documents below to compare them in chat.
        </p>
        <DevBackendMeCard />
        <FileUploadCard />
      </main>
    </div>
  );
}
