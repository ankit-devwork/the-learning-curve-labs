# Supabase auth email — why nothing arrives and how to fix it

InsightLab signup/login uses **Supabase Auth**. Confirmation emails are sent by **your Supabase project**, not the InsightLab API or EC2 backend.

If signup shows “check your email” but **Gmail is empty**, the app is usually working — Supabase mail is not.

## What you see in the app vs what Supabase does

| Layer | Sends email? |
|-------|----------------|
| InsightLab API (EC2) | No |
| Study sheet invites | No (copy link manually) |
| **Supabase Auth** (signup confirm, password reset) | **Yes — if SMTP is configured** |

`auth.signUp()` can return **success with no error** even when no email is delivered (built-in rate limit or duplicate signup).

## Root cause (most common)

Supabase’s **built-in email service** is for development only:

- About **2 emails per hour** per project ([Supabase docs](https://supabase.com/docs/guides/auth/auth-smtp))
- Best-effort delivery — often **does not reach Gmail**
- After a few signups/resends, further emails are **silently dropped** until the hour resets
- `signUp` / `resend` still return success — check **Authentication → Logs** for the real story

**Production requires custom SMTP** (Resend, SendGrid, AWS SES, etc.).

## Fix now — unblock one user (no SMTP)

1. Supabase dashboard → **Authentication → Users**
2. Find the user (e.g. `ankitsrivastava4u@gmail.com`)
3. **⋯ → Confirm user**
4. User signs in on https://insight-lab-pi.vercel.app/login with their password

## Fix for production — custom SMTP (Resend example)

### 1. Resend account

1. Sign up at [resend.com](https://resend.com)
2. Add and verify a domain **or** use their test sender for development
3. Create an **API key**

### 2. Supabase SMTP settings

**Project Settings → Authentication → SMTP Settings** (or **Authentication → SMTP**)

| Field | Example (Resend) |
|-------|------------------|
| Enable custom SMTP | On |
| Sender email | `onboarding@yourdomain.com` (must be allowed by provider) |
| Sender name | `InsightLab` |
| Host | `smtp.resend.com` |
| Port | `465` (SSL) or `587` (TLS) |
| Username | `resend` |
| Password | Your Resend API key |

Save, then send a test from the dashboard if available.

### 3. Auth URL configuration

**Authentication → URL configuration**

| Field | Value |
|-------|-------|
| Site URL | `https://insight-lab-pi.vercel.app` |
| Redirect URLs | `https://insight-lab-pi.vercel.app/**` |

### 4. Rate limits after custom SMTP

**Authentication → Rate Limits** — built-in 2/hour cap applies to default mail only. After SMTP, limits follow your provider (Supabase may still cap auth emails at ~30/h until you raise it).

### 5. Verify in Auth logs

**Authentication → Logs** — after signup or resend, look for:

- `user.signup` / mail sent — good
- `Email rate limit exceeded` — still on built-in mail or hit provider cap
- SMTP authentication errors — wrong host/port/API key

## Other gotchas

| Issue | What to do |
|-------|------------|
| Email already registered | Signup shows **“An account with this email already exists”** — sign in, or use **Resend** on login if unconfirmed |
| Confirm email disabled | Signup returns a session immediately — no mail needed |
| Gmail spam | Check Spam + Promotions |
| Wrong Supabase project | Vercel `NEXT_PUBLIC_SUPABASE_URL` must match the project where SMTP is configured |

## Optional — disable confirmation (internal testing only)

**Authentication → Providers → Email** → turn off **Confirm email**

Users can sign in right after signup. **Do not use in production** unless you accept unverified addresses.

## InsightLab app behavior (after PR deploy)

- Signup/login explain built-in Supabase limits
- **Resend confirmation email** calls `auth.resend({ type: 'signup' })` — still subject to Supabase SMTP
- Google sign-in bypasses email confirmation entirely

## Related docs

- [DEPLOY-ECR.md — Supabase Auth section](DEPLOY-ECR.md#supabase-auth-fix-localhost-redirect)
- [Supabase: Send emails with custom SMTP](https://supabase.com/docs/guides/auth/auth-smtp)
