# Supabase

SQL migrations and RLS policies for InsightLab.

## Setup

1. Create a project at [supabase.com](https://supabase.com).
2. Run `001_initial.sql` in the SQL Editor (or use Supabase CLI).
3. Create a Storage bucket named `uploads` (private).
4. Copy project URL and keys to `.env` files.

## Local CLI (optional)

```bash
npx supabase init
npx supabase link --project-ref your-project-ref
npx supabase db push
```

RLS policies will be added in Phase 1 when auth flows are implemented.
