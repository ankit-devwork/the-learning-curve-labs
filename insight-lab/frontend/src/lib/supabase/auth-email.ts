import type { SupabaseClient } from "@supabase/supabase-js";

export function authEmailRedirectTo(): string {
  if (typeof window === "undefined") {
    return "/auth/callback";
  }
  return `${window.location.origin}/auth/callback`;
}

export function isEmailNotConfirmedError(message: string): boolean {
  const normalized = message.toLowerCase();
  return (
    normalized.includes("email not confirmed") ||
    normalized.includes("email_not_confirmed")
  );
}

export async function resendSignupConfirmation(
  supabase: SupabaseClient,
  email: string
): Promise<{ error: string | null }> {
  const { error } = await supabase.auth.resend({
    type: "signup",
    email: email.trim(),
    options: {
      emailRedirectTo: authEmailRedirectTo(),
    },
  });

  if (error) {
    return { error: error.message };
  }
  return { error: null };
}
