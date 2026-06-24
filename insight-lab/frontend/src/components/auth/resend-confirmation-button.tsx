"use client";

import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { resendSignupConfirmation, formatAuthEmailError } from "@/lib/supabase/auth-email";
import { Button } from "@/components/ui/button";

type ResendConfirmationButtonProps = {
  email: string;
  className?: string;
};

export function ResendConfirmationButton({ email, className }: ResendConfirmationButtonProps) {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleResend = async () => {
    const trimmed = email.trim();
    if (!trimmed) {
      setError("Enter your email address first.");
      return;
    }

    setLoading(true);
    setError(null);
    setMessage(null);

    const supabase = createClient();
    const { error: resendError } = await resendSignupConfirmation(supabase, trimmed);

    if (resendError) {
      setError(formatAuthEmailError(resendError));
    } else {
      setMessage(
        "Supabase accepted the resend request. Check your inbox and spam in a few minutes."
      );
    }
    setLoading(false);
  };

  return (
    <div className={className}>
      <Button type="button" variant="outline" className="w-full" disabled={loading} onClick={handleResend}>
        {loading ? "Sending..." : "Resend confirmation email"}
      </Button>
      {message ? <p className="mt-2 text-sm text-muted-foreground">{message}</p> : null}
      {error ? <p className="mt-2 text-sm text-destructive">{error}</p> : null}
    </div>
  );
}
