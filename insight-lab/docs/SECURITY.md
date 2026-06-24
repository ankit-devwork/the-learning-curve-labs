# InsightLab — Security

Security model, known risks, and production checklist for InsightLab.

## Architecture

| Layer | Mechanism |
|-------|-----------|
| **Frontend** | Supabase Auth (Google OAuth); session JWT on API calls |
| **Backend** | Verifies JWT (`role === authenticated`); service-role Supabase for DB |
| **Database** | RLS on all tables; backend is primary enforcement boundary |
| **Storage** | Private bucket; signed URLs via backend |

Team chat messages are **never** exposed across study sheets: Postgres RLS (`is_workspace_member`) plus backend `require_workspace_role()` on every list/post/delete. Message bodies are validated server-side (ASCII English, no links/files/HTML/emoji).

The backend uses the **Supabase service role**, which bypasses RLS. Every route must call `require_workspace_role()` or `get_accessible_document()` before reading or writing user data.

## Authentication

- JWT verified via HS256 (`SUPABASE_JWT_SECRET`) or JWKS (`SUPABASE_URL`)
- Audience must be `authenticated`
- Token `role` claim must be `authenticated` (rejects anon/service tokens)
- Client-facing auth errors are generic (no exception details leaked)

## Authorization matrix

| Action | Minimum role |
|--------|----------------|
| View study sheet / documents | viewer |
| Upload, generate artifacts, exports | editor |
| Share, invites, classroom analytics | editor |
| Remove members, change roles | owner |
| Delete any team chat message | owner |
| Post team chat message | viewer (all members) |
| Course pack / LMS / Markdown export | editor |
| QTI quiz export | editor |

## Public endpoints (no JWT)

| Route | Mitigation |
|-------|------------|
| `GET /health`, `GET /ready` | No sensitive data in production |
| `GET /upload/config` | Upload limits only |
| `GET /invites/{token}/preview` | Rate limited; minimal metadata |
| `GET/POST /public/quizzes/{token}` | Rate limited by IP + token; answer validation |

## Rate limits

Configured in `backend/config.yaml`:

- Chat, quiz generate/submit, Excel, upload, sharing invites, team chat (post + list)
- Public quiz: `public_get_rate_limit_per_min`, `public_submit_rate_limit_per_min`
- Redis / Upstash required for distributed rate limiting

## Production checklist

### Backend (`backend/.env`)

```bash
APP_ENV=production
APP_APP__ENV=production          # yaml override — hides /docs
APP_DEBUG=false

SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...    # never expose to frontend
SUPABASE_JWT_SECRET=...          # HS256 projects

CORS_ALLOW_ORIGINS=https://your-app.vercel.app   # HTTPS only, no wildcard

RESEND_API_KEY=...               # optional invite email
FRONTEND_BASE_URL=https://your-app.vercel.app
```

### Frontend (Vercel)

```bash
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
NEXT_PUBLIC_API_URL=https://your-api.example.com
NEXT_PUBLIC_SHOW_DEV_PANEL=false
```

### Supabase

Run migrations **001–015** (see `supabase/README.md`).

Migration **015** adds:

- Deny-all RLS on `quiz_public_attempts` for anon/authenticated
- Trigger to prevent self-elevation of `profiles.role`

### Deploy verification

1. `GET /docs` returns 404 in production
2. `GET /` returns minimal JSON (no endpoint map)
3. Invalid JWT returns 401 with generic message
4. Viewer cannot export course pack Markdown
5. Invite list API does not return raw tokens (use `GET .../invites/{id}/link`)

## Reporting issues

Do not commit secrets. Rotate keys if exposed. Report security concerns to the repository owner.

## Related docs

- [DEPLOY-EC2.md](DEPLOY-EC2.md) — manual EC2 deploy
- [DEPLOY-ECR.md](DEPLOY-ECR.md) — Docker / ECR deploy
- [SUPABASE-AUTH-EMAIL.md](SUPABASE-AUTH-EMAIL.md) — auth email / SMTP
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — feature checklist
