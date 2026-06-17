# Supabase

SQL migrations and RLS policies for InsightLab.

## Setup

1. Create a project at [supabase.com](https://supabase.com).
2. Run `001_initial.sql` in the SQL Editor (or use Supabase CLI).
3. Run `002_document_chunks.sql` for document summary + chat (Step 1.7).
4. Run `003_pgvector_embeddings.sql` for semantic chunk search (Step 1.7b).
5. Run `004_excel_charts.sql` for Excel analysis results (Step 1.8).
6. Create a Storage bucket named `uploads` (private).
7. Copy project URL and keys to `.env` files.

## Local CLI (optional)

```bash
npx supabase init
npx supabase link --project-ref your-project-ref
npx supabase db push
```

RLS policies will be added in Phase 1 when auth flows are implemented.
