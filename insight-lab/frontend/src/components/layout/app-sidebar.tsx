"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  Compass,
  FolderOpen,
  HelpCircle,
  MessagesSquare,
} from "lucide-react";
import { BrandMark } from "@/components/layout/brand-mark";
import { SignOutButton } from "@/components/auth/sign-out-button";
import { InsightLabTourHost } from "@/components/onboarding/insight-lab-tour";
import { TeamChatDockShell } from "@/components/workspace/team-chat-dock-shell";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme/theme-toggle";
import { requestTourRestart } from "@/lib/onboarding";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard/sets", label: "Study sheets", icon: FolderOpen },
  { href: "/dashboard/compare", label: "Compare", icon: MessagesSquare },
  { href: "/dashboard/help", label: "User guide", icon: HelpCircle },
];

type AppSidebarProps = {
  userEmail?: string | null;
  children: React.ReactNode;
  wide?: boolean;
};

export function AppSidebarLayout({ userEmail, children, wide }: AppSidebarProps) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-[hsl(var(--shell))]">
      <div className="flex min-h-screen">
        <aside className="hidden w-64 shrink-0 self-start border-r bg-background md:sticky md:top-0 md:flex md:h-screen md:max-h-screen md:flex-col">
          <div className="border-b px-4 py-4">
            <BrandMark href="/dashboard" />
          </div>
          <nav className="flex-1 space-y-1 overflow-y-auto p-3">
            {navItems.map(({ href, label, icon: Icon }) => {
              const active = pathname === href || pathname.startsWith(`${href}/`);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" aria-hidden />
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="shrink-0 border-t p-4 text-xs text-muted-foreground">
            <div className="mb-3 flex items-center gap-2">
              <BookOpen className="h-4 w-4" aria-hidden />
              <span>Learning workspace</span>
            </div>
            <div className="mb-3 flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="flex-1 gap-2"
                onClick={() => requestTourRestart()}
              >
                <Compass className="h-4 w-4" aria-hidden />
                Show tour
              </Button>
              <ThemeToggle />
            </div>
            {userEmail ? <p className="truncate">{userEmail}</p> : null}
            <div className="mt-3">
              <SignOutButton />
            </div>
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-40 border-b bg-background/90 backdrop-blur md:hidden">
            <div className="flex items-center justify-between gap-3 px-4 py-3">
              <BrandMark href="/dashboard" />
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="gap-1"
                  onClick={() => requestTourRestart()}
                >
                  <Compass className="h-4 w-4" aria-hidden />
                  Tour
                </Button>
                <ThemeToggle />
                <SignOutButton />
              </div>
            </div>
            <nav className="flex gap-1 overflow-x-auto px-3 pb-3">
              {navItems.map(({ href, label }) => {
                const active = pathname === href || pathname.startsWith(`${href}/`);
                return (
                  <Link
                    key={href}
                    href={href}
                    className={cn(
                      "shrink-0 rounded-full px-3 py-1.5 text-xs font-medium",
                      active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
                    )}
                  >
                    {label}
                  </Link>
                );
              })}
            </nav>
          </header>
          <main className={cn("flex-1 px-4 py-6", wide ? "max-w-none" : "mx-auto w-full max-w-6xl")}>
            {children}
          </main>
        </div>
      </div>
      <InsightLabTourHost />
      <TeamChatDockShell />
    </div>
  );
}
