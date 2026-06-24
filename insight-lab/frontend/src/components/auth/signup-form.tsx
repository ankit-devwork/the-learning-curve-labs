"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { GoogleSignInButton } from "@/components/auth/google-sign-in-button";
import { ResendConfirmationButton } from "@/components/auth/resend-confirmation-button";
import { AuthEmailHelp } from "@/components/auth/auth-email-help";
import {
  authEmailRedirectTo,
  formatAuthEmailError,
  isAuthEmailRateLimitError,
  SIGNUP_INCOMPLETE_HYBRID_MESSAGE,
} from "@/lib/supabase/auth-email";

export function SignUpForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    setAwaitingConfirmation(false);

    const supabase = createClient();
    const { data, error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: authEmailRedirectTo(),
      },
    });

    if (signUpError) {
      setError(formatAuthEmailError(signUpError.message));
      if (isAuthEmailRateLimitError(signUpError.message)) {
        setAwaitingConfirmation(true);
      }
      setLoading(false);
      return;
    }

    if (data.session) {
      router.push("/dashboard");
      router.refresh();
      return;
    }

    // Same message for new signups and re-signups (Supabase does not always expose duplicates).
    setAwaitingConfirmation(true);
    setMessage(SIGNUP_INCOMPLETE_HYBRID_MESSAGE);
    setLoading(false);
  };

  return (
    <Card className="border-border/80 shadow-lg">
      <CardHeader className="space-y-1 text-center">
        <CardTitle className="text-2xl">Create account</CardTitle>
        <CardDescription>Start uploading documents and spreadsheets</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <GoogleSignInButton label="Sign up with Google" />
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-card px-2 text-muted-foreground">Or sign up with email</span>
          </div>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              autoComplete="new-password"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {message && <p className="text-sm text-muted-foreground">{message}</p>}
          {awaitingConfirmation ? (
            <>
              <Button type="button" variant="outline" className="w-full" asChild>
                <Link href="/login">Go to sign in</Link>
              </Button>
              <ResendConfirmationButton email={email} />
              <AuthEmailHelp className="rounded-md border bg-muted/30 p-3" />
            </>
          ) : (
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Creating account..." : "Sign up"}
            </Button>
          )}
        </form>
      </CardContent>
      <CardFooter className="justify-center">
        <p className="text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="font-medium text-primary underline-offset-4 hover:underline">
            Sign in
          </Link>
        </p>
      </CardFooter>
    </Card>
  );
}
