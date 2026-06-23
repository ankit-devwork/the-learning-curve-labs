-- Phase 8: Member-aware RLS for artifacts/chunks (defense-in-depth)

-- Workspace members: owner may promote/demote editors and viewers
create policy workspace_members_update on public.workspace_members
  for update using (
    public.is_workspace_member(workspace_id, array['owner'])
  )
  with check (
    public.is_workspace_member(workspace_id, array['owner'])
    and role in ('editor', 'viewer')
  );

-- Document chunks: workspace members can read chunks for shared documents
create policy document_chunks_select_member on public.document_chunks
  for select using (
    exists (
      select 1
      from public.documents d
      where d.id = document_chunks.document_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Quizzes and questions: workspace members can read shared quizzes
create policy quizzes_select_member on public.quizzes
  for select using (
    exists (
      select 1
      from public.documents d
      where d.id = quizzes.document_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

create policy quiz_questions_select_member on public.quiz_questions
  for select using (
    exists (
      select 1
      from public.quizzes q
      join public.documents d on d.id = q.document_id
      where q.id = quiz_questions.quiz_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Quiz attempts: members may record attempts on shared workspace quizzes
create policy quiz_attempts_insert_member on public.quiz_attempts
  for insert with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.quizzes q
      join public.documents d on d.id = q.document_id
      where q.id = quiz_attempts.quiz_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Study guides: workspace members can read shared guides
create policy study_guides_select_member on public.study_guides
  for select using (
    exists (
      select 1
      from public.documents d
      where d.id = study_guides.document_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Flashcard sets and cards: workspace members can read shared flashcards
create policy flashcard_sets_select_member on public.flashcard_sets
  for select using (
    exists (
      select 1
      from public.documents d
      where d.id = flashcard_sets.document_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

create policy flashcards_select_member on public.flashcards
  for select using (
    exists (
      select 1
      from public.flashcard_sets s
      join public.documents d on d.id = s.document_id
      where s.id = flashcards.set_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Flashcard reviews: members may review cards in shared sets
create policy flashcard_reviews_insert_member on public.flashcard_reviews
  for insert with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.flashcards fc
      join public.flashcard_sets s on s.id = fc.set_id
      join public.documents d on d.id = s.document_id
      where fc.id = flashcard_reviews.flashcard_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );
