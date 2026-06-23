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
9. Run `008_phase3_4_study_features.sql` for flashcards and study guides.
10. Run `009_phase6_sharing_quiz_edit.sql` for workspace members, invites, and quiz publish.
11. Run `010_security_hardening.sql` for RLS fixes and RPC lockdown updates.
12. Run `011_phase8_member_rls.sql` for member-aware artifact policies (chunks, quizzes, flashcards, study guides).
13. Create a Storage bucket named `uploads` (private).
14. Copy project URL and keys to `.env` files.

## Local CLI (optional)

```bash
npx supabase init
npx supabase link --project-ref your-project-ref
npx supabase db push
```

RLS policies are defined in `005_rls_policies.sql`, `006_storage_and_rpc_security.sql`, and updated in `009`–`011`.
The backend uses the service role key and enforces ownership in application code.
