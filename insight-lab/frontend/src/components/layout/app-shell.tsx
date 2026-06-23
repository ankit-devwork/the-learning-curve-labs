import { BrandMark } from "@/components/layout/brand-mark";
import { SignOutButton } from "@/components/auth/sign-out-button";

type AppShellProps = {
  userEmail?: string | null;
  children: React.ReactNode;
  wide?: boolean;
};

export function AppShell({ userEmail, children, wide }: AppShellProps) {
  return (
    <div className="min-h-screen bg-[hsl(var(--shell))]">
      <header className="sticky top-0 z-40 border-b bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
        <div
          className={`mx-auto flex items-center justify-between gap-4 px-4 py-3 ${wide ? "max-w-6xl" : "max-w-5xl"}`}
        >
          <BrandMark href="/dashboard" />
          <div className="flex items-center gap-3">
            {userEmail && (
              <span className="hidden max-w-[220px] truncate text-sm text-muted-foreground sm:inline">
                {userEmail}
              </span>
            )}
            <SignOutButton />
          </div>
        </div>
      </header>
      <main className={`mx-auto px-4 py-8 ${wide ? "max-w-6xl" : "max-w-5xl"}`}>{children}</main>
    </div>
  );
}
