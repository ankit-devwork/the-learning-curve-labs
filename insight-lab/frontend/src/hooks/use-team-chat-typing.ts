"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { apiFetch } from "@/lib/api";

const TYPING_HEARTBEAT_MS = 1_200;
const TYPING_TTL_MS = 5_000;

export type TypingUser = {
  user_id: string;
  name: string;
};

type TeamChatTypingResponse = {
  typers?: TypingUser[];
  migration_required?: boolean;
};

type UseTeamChatTypingOptions = {
  workspaceId: string | null;
  accessToken: string | null;
  currentUserId: string | null;
  draft: string;
  enabled?: boolean;
};

export function useTeamChatTyping({
  workspaceId,
  accessToken,
  currentUserId,
  draft,
  enabled = true,
}: UseTeamChatTypingOptions) {
  const [typingUsers, setTypingUsers] = useState<TypingUser[]>([]);
  const lastHeartbeatRef = useRef(0);
  const expiryTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const clearTypingUser = useCallback((userId: string) => {
    setTypingUsers((prev) => prev.filter((user) => user.user_id !== userId));
    const timer = expiryTimersRef.current.get(userId);
    if (timer) {
      clearTimeout(timer);
      expiryTimersRef.current.delete(userId);
    }
  }, []);

  const applyTypers = useCallback(
    (typers: TypingUser[]) => {
      const filtered = typers.filter((user) => user.user_id !== currentUserId);
      setTypingUsers(filtered);
      expiryTimersRef.current.forEach((timer) => clearTimeout(timer));
      expiryTimersRef.current.clear();
      for (const user of filtered) {
        expiryTimersRef.current.set(
          user.user_id,
          setTimeout(() => clearTypingUser(user.user_id), TYPING_TTL_MS),
        );
      }
    },
    [clearTypingUser, currentUserId],
  );

  const loadTypers = useCallback(async () => {
    if (!accessToken || !workspaceId) {
      return;
    }
    const response = await apiFetch(`/workspaces/${workspaceId}/typing`, accessToken);
    if (!response.ok) {
      return;
    }
    const payload = (await response.json()) as TeamChatTypingResponse;
    applyTypers(payload.typers ?? []);
  }, [accessToken, applyTypers, workspaceId]);

  const sendTypingState = useCallback(
    async (active: boolean) => {
      if (!accessToken || !workspaceId) {
        return;
      }
      await apiFetch(`/workspaces/${workspaceId}/typing`, accessToken, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active }),
      });
    },
    [accessToken, workspaceId],
  );

  useEffect(() => {
    if (!enabled || !workspaceId || !accessToken) {
      setTypingUsers([]);
      return;
    }

    void loadTypers();

    const supabase = createClient();
    const channel = supabase
      .channel(`workspace-typing-${workspaceId}`)
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "workspace_typing_presence",
          filter: `workspace_id=eq.${workspaceId}`,
        },
        () => {
          void loadTypers();
        },
      )
      .subscribe();

    return () => {
      const timers = expiryTimersRef.current;
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
      void sendTypingState(false);
      void supabase.removeChannel(channel);
    };
  }, [accessToken, enabled, loadTypers, sendTypingState, workspaceId]);

  useEffect(() => {
    if (!enabled || !workspaceId || !accessToken || !currentUserId) {
      return;
    }

    if (!draft.trim()) {
      void sendTypingState(false);
      return;
    }

    const now = Date.now();
    if (now - lastHeartbeatRef.current < TYPING_HEARTBEAT_MS) {
      return;
    }
    lastHeartbeatRef.current = now;
    void sendTypingState(true);
  }, [accessToken, currentUserId, draft, enabled, sendTypingState, workspaceId]);

  const typingLabel =
    typingUsers.length === 0
      ? null
      : typingUsers.length === 1
        ? `${typingUsers[0].name} is typing…`
        : `${typingUsers.length} people are typing…`;

  return { typingUsers, typingLabel };
}
