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

export function SignUpForm() {
  return (
    <Card className="border-border/80 shadow-lg">
      <CardHeader className="space-y-1 text-center">
        <CardTitle className="text-2xl">Create account</CardTitle>
        <CardDescription>Sign up with Google to start uploading documents and spreadsheets</CardDescription>
      </CardHeader>
      <CardContent>
        <GoogleSignInButton label="Sign up with Google" />
        <p className="mt-4 text-center text-xs text-muted-foreground">
          Email and password sign-up is temporarily unavailable.
        </p>
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
