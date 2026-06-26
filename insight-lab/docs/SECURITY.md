# InsightLab — Security

Security model, known risks, and production checklist for InsightLab.

## Architecture

| Layer | Mechanism |
|-------|-----------|
| **Frontend** | Supabase Auth (Google OAuth); session JWT on API calls |
| **Backend** | Verifies JWT (`role === authenticated`); service-role Supabase for DB |
| **Database** | RLS on all tables; backend is primary enforcement boundary |
| **Storage** | Private bucket; optional app-level Fernet encryption at rest; signed URLs via backend |

Team chat messages are **never** exposed across study sheets: Postgres RLS (`is_workspace_member`) plus backend `require_workspace_role()` on every list/post/delete/read/inbox route. Message bodies are validated server-side (ASCII English, no links/files/HTML/emoji). Read cursors and read receipts are user-scoped (RLS); message authors may see who read their messages only — not other members' read state.

Storage objects in the `uploads` bucket use path `{owner_id}/{document_id}/{filename}`. When `DOCUMENT_STORAGE_ENCRYPTION_KEY` is set, new uploads are encrypted with Fernet before storage (`documents.storage_encrypted = true`). Legacy plaintext blobs remain readable until re-uploaded.

Editors can delete individual documents via `DELETE /documents/{id}` — storage blobs, derived artifacts (DB cascade), Redis caches, and Neo4j nodes are cleaned up. Study sheet deletion also purges all workspace storage paths.

Migration **018** adds a SELECT policy so workspace members can read files for documents in shared study sheets (defense-in-depth; the backend still serves content via the service role).

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
| Delete uploaded files | editor |
| Share, invites, classroom analytics | editor |
| Remove members, change roles | owner |
| Delete any team chat message | owner |
| Post team chat message | viewer (all members) |
| Mark team chat read / inbox / typing | viewer (own read state; members only) |
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

- Chat, quiz generate/submit, Excel, upload, sharing invites, team chat (post, list, delete, inbox, mark-read, typing)
- Explain (`explain.rate_limit_per_min`), homework solver (`homework.rate_limit_per_min`), artifact generation
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

DOCUMENT_STORAGE_ENCRYPTION_KEY=...   # Fernet key — encrypt uploads at rest (recommended in production)

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

Run migrations **001–023** (see [supabase/README.md](../supabase/README.md)).

| Migration | Purpose |
|-----------|---------|
| **015** | Deny-all RLS on `quiz_public_attempts`; block `profiles.role` self-elevation |
| **016** | Tracked study sessions and learning paths |
| **017** | Member-only team chat (`workspace_messages` RLS) |
| **018** | Storage read policies for workspace members on shared documents |
| **019** | Supabase Realtime publication for `workspace_messages` (member RLS still applies to subscribers) |
| **020** | Chat history, flashcard SRS, persisted audio/slides, homework solutions (user-scoped RLS) |
| **021** | `storage_encrypted` flag; editor document delete RLS |
| **022** | Team chat read cursors + per-message read receipts; Realtime on `workspace_message_reads` |
| **023** | Member-scoped typing presence (`workspace_typing_presence` RLS + Realtime) |

Team chat posts, deletes, inbox, and mark-read remain **backend-only** (JWT + rate-limited FastAPI routes). Realtime message and read-receipt subscriptions are read-only on the client — members only receive rows their RLS policies allow.

**Typing indicators** use the `workspace_typing_presence` table with member RLS and backend heartbeats (`POST/GET /workspaces/{id}/typing`). Only authenticated study sheet members can publish or subscribe via Realtime; knowing a workspace UUID alone is not sufficient.

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
