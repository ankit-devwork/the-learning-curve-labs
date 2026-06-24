"use client";

import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { GoogleSignInButton } from "@/components/auth/google-sign-in-button";

export function LoginForm() {
  return (
    <Card className="border-border/80 shadow-lg">
      <CardHeader className="space-y-1 text-center">
        <CardTitle className="text-2xl">Sign in</CardTitle>
        <CardDescription>Use your Google account to access your workspace</CardDescription>
      </CardHeader>
      <CardContent>
        <GoogleSignInButton label="Sign in with Google" />
        <p className="mt-4 text-center text-xs text-muted-foreground">
          Email and password sign-in is temporarily unavailable.
        </p>
      </CardContent>
      <CardFooter className="justify-center">
        <p className="text-sm text-muted-foreground">
          No account?{" "}
          <Link href="/signup" className="font-medium text-primary underline-offset-4 hover:underline">
            Sign up
          </Link>
        </p>
      </CardFooter>
    </Card>
  );
}
