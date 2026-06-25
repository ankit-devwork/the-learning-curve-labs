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
13. Run `012_profiles_email.sql` for profile email sync (required for study sheet invites).
14. Run `013_document_infographics.sql` for document infographics.
15. Run `014_tier1_tier2_features.sql` for public quiz sharing and source links.
16. Run `015_security_hardening.sql` for quiz_public_attempts RLS and profiles.role lock.
17. Run `016_study_sessions_learning_paths.sql` for study session progress and learning paths.
18. Run `017_workspace_team_chat.sql` for member-only team chat.
19. Run `018_storage_member_read.sql` for workspace member Storage read access.
20. Run `019_workspace_messages_realtime.sql` for team chat Realtime (member RLS unchanged).
21. Create a Storage bucket named `uploads` (private).
22. Copy project URL and keys to `.env` files.

## Local CLI (optional)

```bash
npx supabase init
npx supabase link --project-ref your-project-ref
npx supabase db push
```

RLS policies are defined in `005_rls_policies.sql`, `006_storage_and_rpc_security.sql`, and updated in `009`–`011`.
The backend uses the service role key and enforces ownership in application code.
