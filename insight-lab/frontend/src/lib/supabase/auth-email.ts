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
 * and confirmed (anti-enumeration). Unconfirmed duplicates may still return identities,
 * so callers should not rely on this alone for UX — use SIGNUP_INCOMPLETE_HYBRID_MESSAGE
 * whenever signUp succeeds without a session.
 */
export function isSignupDuplicateResponse(user: User | null, session: unknown): boolean {
  if (session || !user) {
    return false;
  }
  return !user.identities || user.identities.length === 0;
}

/** Shown whenever signUp succeeds without a session (new or existing email). */
export const SIGNUP_INCOMPLETE_HYBRID_MESSAGE =
  "If you already have an account, sign in below. Otherwise check your email (and spam) for a confirmation link.";

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

export function formatAuthEmailError(message: string, audience: "user" | "admin" = "user"): string {
  if (isAuthEmailRateLimitError(message)) {
    if (audience === "admin") {
      return `${message} Enable custom SMTP in Project Settings → Authentication → SMTP, or confirm the user manually under Authentication → Users.`;
    }
    return (
      "Email sending is temporarily unavailable (Supabase limit reached). " +
      "Try Sign in with Google, ask the app owner to activate your account, or try again in about an hour."
    );
  }
  return message;
}
