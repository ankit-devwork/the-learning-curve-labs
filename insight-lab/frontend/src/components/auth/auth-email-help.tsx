import { SUPABASE_BUILTIN_EMAIL_NOTE } from "@/lib/supabase/auth-email";

export function AuthEmailHelp({ className }: { className?: string }) {
  return (
    <div className={className}>
      <p className="text-xs leading-relaxed text-muted-foreground">{SUPABASE_BUILTIN_EMAIL_NOTE}</p>
      <ul className="mt-2 list-inside list-disc space-y-1 text-xs text-muted-foreground">
        <li>Check Gmail Spam and Promotions.</li>
        <li>
          Project owner: Supabase → <strong>Authentication → Logs</strong> for mail or rate-limit errors.
        </li>
        <li>
          Fastest unblock: Supabase → <strong>Authentication → Users</strong> → Confirm user.
        </li>
        <li>
          Production fix:{" "}
          <a
            href="https://supabase.com/docs/guides/auth/auth-smtp"
            className="font-medium text-primary underline-offset-4 hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            configure custom SMTP
          </a>{" "}
          (Resend, SendGrid, etc.).
        </li>
      </ul>
    </div>
  );
}
