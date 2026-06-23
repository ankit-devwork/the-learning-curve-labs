import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { AppSidebarLayout } from "@/components/layout/app-sidebar";
import { ToastProvider } from "@/components/ui/toast";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  return (
    <ToastProvider>
      <AppSidebarLayout userEmail={user.email} wide>
        {children}
      </AppSidebarLayout>
    </ToastProvider>
  );
}
