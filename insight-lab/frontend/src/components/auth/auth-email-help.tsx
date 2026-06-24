"use client";

import { useState } from "react";

export function AuthEmailHelp({ className, variant = "user" }: { className?: string; variant?: "user" | "admin" }) {
  const [open, setOpen] = useState(false);

  if (variant === "user") {
    return (
      <div className={className}>
        <p className="text-xs text-muted-foreground">
          Check Gmail <strong>Spam</strong> and <strong>Promotions</strong>. No email after a few minutes? Try{" "}
          <strong>Resend</strong> above, or sign in if you already registered.
        </p>
        <button
          type="button"
          className="mt-2 text-xs font-medium text-primary underline-offset-4 hover:underline"
          onClick={() => setOpen((value) => !value)}
        >
          {open ? "Hide troubleshooting" : "Still stuck?"}
        </button>
        {open ? (
          <ul className="mt-2 list-inside list-disc space-y-1 text-xs text-muted-foreground">
            <li>Sign in on the login page — use Resend there if you see “Email not confirmed”.</li>
            <li>Try Google sign-in if you used that originally.</li>
            <li>Project owner: confirm the user or configure SMTP in Supabase (see repo docs).</li>
          </ul>
        ) : null}
      </div>
    );
  }

  return (
    <div className={className}>
      <p className="text-xs leading-relaxed text-muted-foreground">
        Supabase built-in mail: ~2/hour, often blocked by Gmail. Enable custom SMTP or confirm users in
        Authentication → Users.
      </p>
      <ul className="mt-2 list-inside list-disc space-y-1 text-xs text-muted-foreground">
        <li>Authentication → Logs — rate limits / SMTP errors</li>
        <li>
          <a
            href="https://supabase.com/docs/guides/auth/auth-smtp"
            className="font-medium text-primary underline-offset-4 hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            Custom SMTP setup
          </a>
        </li>
      </ul>
    </div>
  );
}
