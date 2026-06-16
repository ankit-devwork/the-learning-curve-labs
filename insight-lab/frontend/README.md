# InsightLab Frontend

Next.js 15 app with Supabase Auth (email + Google), shadcn/ui, and Tailwind.

## Setup

### 1. Environment

```powershell
cd frontend
copy .env.local.example .env.local
```

Edit `.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Use the **anon public** key from Supabase → Settings → API (not service_role).

### 2. Supabase Auth configuration

**Authentication → URL configuration**

| Setting | Value |
|---------|-------|
| Site URL | `http://localhost:3000` |
| Redirect URLs | `http://localhost:3000/auth/callback` |

**Authentication → Providers → Email** — enabled (default)

**Authentication → Providers → Google**

1. Enable Google provider
2. Create OAuth credentials in [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
3. Authorized redirect URI: `https://YOUR_PROJECT.supabase.co/auth/v1/callback`
4. Paste Client ID and Client Secret into Supabase

### 3. Install and run

```powershell
npm install
npm run dev
```

Open http://localhost:3000

## Routes

| Route | Description |
|-------|-------------|
| `/login` | Email + Google sign in |
| `/signup` | Email + Google sign up |
| `/auth/callback` | OAuth / email confirm callback |
| `/dashboard` | Protected home (requires login) |

## Stack

- Next.js 15 (App Router)
- Supabase Auth (`@supabase/ssr`)
- shadcn/ui + Tailwind
- TypeScript

## Backend

Run FastAPI separately on port 8000:

```powershell
conda activate insightlab
cd ../backend
uvicorn app.main:app --reload
```
