# Supabase

SQL migrations and RLS policies for InsightLab.

## Setup

1. Create a project at [supabase.com](https://supabase.com).
2. Run `001_initial.sql` in the SQL Editor (or use Supabase CLI).
3. Run `002_document_chunks.sql` for document summary + chat (Step 1.7).
4. Run `003_pgvector_embeddings.sql` for semantic chunk search (Step 1.7b).
5. Run `004_excel_charts.sql` for Excel analysis results (Step 1.8).
6. Run `005_rls_policies.sql` for table RLS policies.
7. Run `006_storage_and_rpc_security.sql` for Storage policies and RPC lockdown.
8. Run `007_phase2_graph_mastery_multi_doc.sql` for concept graph, mastery, multi-doc search.
9. Create a Storage bucket named `uploads` (private).
10. Copy project URL and keys to `.env` files.

## Local CLI (optional)

```bash
npx supabase init
npx supabase link --project-ref your-project-ref
npx supabase db push
```

RLS policies are defined in `005_rls_policies.sql` and `006_storage_and_rpc_security.sql`.
The backend uses the service role key and enforces ownership in application code.
