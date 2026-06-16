import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { SignOutButton } from "@/components/auth/sign-out-button";
import { BackendMeCard } from "@/components/auth/backend-me-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

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
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4">
          <div>
            <h1 className="text-xl font-semibold">InsightLab</h1>
            <p className="text-sm text-muted-foreground">{user.email}</p>
          </div>
          <SignOutButton />
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 p-4 py-8">
        <BackendMeCard />
        <Card>
          <CardHeader>
            <CardTitle>Dashboard</CardTitle>
            <CardDescription>
              Upload Excel or documents, chat, and generate quizzes — coming in Phase 1 features.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-3">
            <Card className="border-dashed">
              <CardHeader>
                <CardTitle className="text-base">Excel insights</CardTitle>
                <CardDescription>Charts and narrative from spreadsheets</CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-dashed">
              <CardHeader>
                <CardTitle className="text-base">Document chat</CardTitle>
                <CardDescription>Summarize and ask questions with citations</CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-dashed">
              <CardHeader>
                <CardTitle className="text-base">Quiz generator</CardTitle>
                <CardDescription>MCQ and single-choice from your docs</CardDescription>
              </CardHeader>
            </Card>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
