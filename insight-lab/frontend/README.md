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

**Authentication → Providers → Email** — enabled (default). For production, configure **SMTP** under Project Settings → Authentication so confirmation emails reach Gmail reliably.

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

## Deploy on Vercel (EC2 HTTP backend)

When the API runs on EC2 **without HTTPS**, use the built-in proxy rewrite:

1. EC2 nginx on port **8080** → `127.0.0.1:8000` (see [docs/DEPLOY-ECR.md](../docs/DEPLOY-ECR.md))
2. Vercel env:
   - `NEXT_PUBLIC_API_URL=/api-backend`
   - `BACKEND_PROXY_URL=http://YOUR_EC2_PUBLIC_IP:8080`
3. Redeploy on Vercel
4. Test: `https://your-app.vercel.app/api-backend/health`

Local dev with proxy (optional):

```env
NEXT_PUBLIC_API_URL=/api-backend
BACKEND_PROXY_URL=http://localhost:8000
```

Or call the API directly: `NEXT_PUBLIC_API_URL=http://localhost:8000`

## Routes

| Route | Description |
|-------|-------------|
| `/login` | Email + Google sign in |
| `/signup` | Email + Google sign up |
| `/auth/callback` | OAuth / email confirm callback |
| `/invite/[token]` | Accept workspace invite |
| `/dashboard` | Redirects to `/dashboard/sets` |
| `/dashboard/sets` | Notebook gallery |
| `/dashboard/sets/[setId]` | Study sheet home — upload, course pack, share |
| `/dashboard/sets/[setId]/documents/[docId]` | Document notebook workspace |
| `/dashboard/sets/[setId]/excel/[docId]` | Excel notebook workspace |
| `/dashboard/compare` | Multi-document chat (documents only) |

Workspace pages support hash tab links (e.g. `#brief`, `#quiz`, `#charts`).

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
