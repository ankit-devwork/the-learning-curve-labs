import type { SupabaseClient, User } from "@supabase/supabase-js";

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

/** Supabase built-in mail is capped (~2/hour). Custom SMTP is required for production. */
export function isAuthEmailRateLimitError(message: string): boolean {
  const normalized = message.toLowerCase();
  return normalized.includes("rate limit") && normalized.includes("email");
}

/**
 * signUp returns success with empty identities when the email is already registered
 * (anti-enumeration). No new confirmation email is sent in that case.
 */
export function isSignupDuplicateResponse(user: User | null, session: unknown): boolean {
  if (session || !user) {
    return false;
  }
  return !user.identities || user.identities.length === 0;
}

export const SUPABASE_BUILTIN_EMAIL_NOTE =
  "Supabase’s default mail sends about 2 emails per hour and often never reaches Gmail. " +
  "Your project owner must enable custom SMTP in the Supabase dashboard, or confirm your account manually under Authentication → Users.";

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

export function formatAuthEmailError(message: string): string {
  if (isAuthEmailRateLimitError(message)) {
    return `${message} Supabase’s built-in email limit is ~2 per hour. Enable custom SMTP in Project Settings → Authentication → SMTP, or ask the project owner to confirm your user manually.`;
  }
  return message;
}
