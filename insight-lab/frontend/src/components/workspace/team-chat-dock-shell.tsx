"use client";

import { usePathname } from "next/navigation";
import { TeamChatGlobalDock } from "@/components/workspace/team-chat-global-dock";

export function TeamChatDockShell() {
  const pathname = usePathname();
  const match = pathname.match(/\/dashboard\/sets\/([^/]+)/);
  const contextWorkspaceId = match?.[1] ?? null;

  return <TeamChatGlobalDock contextWorkspaceId={contextWorkspaceId} />;
}
