# InsightLab Frontend

Next.js 14+ app with Supabase Auth, shadcn/ui, and Tailwind.

## Planned pages

| Route | Purpose |
|-------|---------|
| `/login`, `/signup` | Authentication |
| `/dashboard` | Home |
| `/workspace/[id]/upload` | File upload |
| `/workspace/[id]/excel/[docId]` | Charts & insights |
| `/workspace/[id]/document/[docId]` | Summary, chat, quiz |
| `/workspace/[id]/graph` | Knowledge graph |
| `/settings` | Profile |

## Scaffold (Phase 1)

```bash
npx create-next-app@latest . \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir \
  --import-alias "@/*" \
  --use-npm

npm install @supabase/supabase-js @supabase/ssr
npx shadcn@latest init
```

## Environment

Copy from repo root or create `frontend/.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Run

```bash
npm install
npm run dev
```

Open http://localhost:3000
