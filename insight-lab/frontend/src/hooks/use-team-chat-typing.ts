"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createClient } from "@/lib/supabase/client";

const TYPING_TTL_MS = 3_500;
const TYPING_BROADCAST_MS = 1_200;

export type TypingUser = {
  user_id: string;
  name: string;
};

type UseTeamChatTypingOptions = {
  workspaceId: string | null;
  accessToken: string | null;
  currentUserId: string | null;
  currentUserName: string;
  draft: string;
  enabled?: boolean;
};

export function useTeamChatTyping({
  workspaceId,
  accessToken,
  currentUserId,
  currentUserName,
  draft,
  enabled = true,
}: UseTeamChatTypingOptions) {
  const [typingUsers, setTypingUsers] = useState<TypingUser[]>([]);
  const lastBroadcastRef = useRef(0);
  const expiryTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const clearTypingUser = useCallback((userId: string) => {
    setTypingUsers((prev) => prev.filter((user) => user.user_id !== userId));
    const timer = expiryTimersRef.current.get(userId);
    if (timer) {
      clearTimeout(timer);
      expiryTimersRef.current.delete(userId);
    }
  }, []);

  const noteTypingUser = useCallback(
    (user: TypingUser) => {
      if (!currentUserId || user.user_id === currentUserId) {
        return;
      }
      setTypingUsers((prev) => {
        if (prev.some((entry) => entry.user_id === user.user_id)) {
          return prev;
        }
        return [...prev, user];
      });
      const existing = expiryTimersRef.current.get(user.user_id);
      if (existing) {
        clearTimeout(existing);
      }
      expiryTimersRef.current.set(
        user.user_id,
        setTimeout(() => clearTypingUser(user.user_id), TYPING_TTL_MS),
      );
    },
    [clearTypingUser, currentUserId],
  );

  useEffect(() => {
    if (!enabled || !workspaceId || !accessToken) {
      setTypingUsers([]);
      return;
    }

    const supabase = createClient();
    const channel = supabase.channel(`team-chat-typing:${workspaceId}`, {
      config: { broadcast: { self: false } },
    });

    channel.on("broadcast", { event: "typing" }, ({ payload }) => {
      const data = payload as TypingUser | null;
      if (data?.user_id && data.name) {
        noteTypingUser(data);
      }
    });

    void channel.subscribe();

    return () => {
      const timers = expiryTimersRef.current;
      timers.forEach((timer) => clearTimeout(timer));
      timers.clear();
      void supabase.removeChannel(channel);
    };
  }, [accessToken, enabled, noteTypingUser, workspaceId]);

  useEffect(() => {
    if (!enabled || !workspaceId || !accessToken || !currentUserId || !draft.trim()) {
      return;
    }

    const now = Date.now();
    if (now - lastBroadcastRef.current < TYPING_BROADCAST_MS) {
      return;
    }
    lastBroadcastRef.current = now;

    const supabase = createClient();
    const channel = supabase.channel(`team-chat-typing:${workspaceId}`);
    void channel.subscribe((status) => {
      if (status === "SUBSCRIBED") {
        void channel.send({
          type: "broadcast",
          event: "typing",
          payload: { user_id: currentUserId, name: currentUserName },
        });
        void supabase.removeChannel(channel);
      }
    });
  }, [accessToken, currentUserId, currentUserName, draft, enabled, workspaceId]);

  const typingLabel =
    typingUsers.length === 0
      ? null
      : typingUsers.length === 1
        ? `${typingUsers[0].name} is typing…`
        : `${typingUsers.length} people are typing…`;

  return { typingUsers, typingLabel };
}
