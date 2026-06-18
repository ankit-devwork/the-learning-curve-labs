-- Step 1.9b: Row Level Security policies (defense in depth; backend uses service role)

-- Profiles: users manage their own row
create policy profiles_select_own on public.profiles
  for select using (auth.uid() = id);

create policy profiles_update_own on public.profiles
  for update using (auth.uid() = id);

create policy profiles_insert_own on public.profiles
  for insert with check (auth.uid() = id);

-- Workspaces: owner access only
create policy workspaces_select_own on public.workspaces
  for select using (auth.uid() = owner_id);

create policy workspaces_insert_own on public.workspaces
  for insert with check (auth.uid() = owner_id);

create policy workspaces_update_own on public.workspaces
  for update using (auth.uid() = owner_id);

create policy workspaces_delete_own on public.workspaces
  for delete using (auth.uid() = owner_id);

-- Documents: owner access only
create policy documents_select_own on public.documents
  for select using (auth.uid() = owner_id);

create policy documents_insert_own on public.documents
  for insert with check (auth.uid() = owner_id);

create policy documents_update_own on public.documents
  for update using (auth.uid() = owner_id);

create policy documents_delete_own on public.documents
  for delete using (auth.uid() = owner_id);

-- Document chunks: via document ownership
create policy document_chunks_select_own on public.document_chunks
  for select using (
    exists (
      select 1 from public.documents d
      where d.id = document_chunks.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy document_chunks_insert_own on public.document_chunks
  for insert with check (
    exists (
      select 1 from public.documents d
      where d.id = document_chunks.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy document_chunks_delete_own on public.document_chunks
  for delete using (
    exists (
      select 1 from public.documents d
      where d.id = document_chunks.document_id
        and d.owner_id = auth.uid()
    )
  );

-- Quizzes: via document ownership
create policy quizzes_select_own on public.quizzes
  for select using (
    exists (
      select 1 from public.documents d
      where d.id = quizzes.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy quizzes_insert_own on public.quizzes
  for insert with check (
    exists (
      select 1 from public.documents d
      where d.id = quizzes.document_id
        and d.owner_id = auth.uid()
    )
  );

-- Quiz questions: via quiz -> document ownership
create policy quiz_questions_select_own on public.quiz_questions
  for select using (
    exists (
      select 1
      from public.quizzes q
      join public.documents d on d.id = q.document_id
      where q.id = quiz_questions.quiz_id
        and d.owner_id = auth.uid()
    )
  );

create policy quiz_questions_insert_own on public.quiz_questions
  for insert with check (
    exists (
      select 1
      from public.quizzes q
      join public.documents d on d.id = q.document_id
      where q.id = quiz_questions.quiz_id
        and d.owner_id = auth.uid()
    )
  );

-- Quiz attempts: users see/create their own attempts on owned documents
create policy quiz_attempts_select_own on public.quiz_attempts
  for select using (auth.uid() = user_id);

create policy quiz_attempts_insert_own on public.quiz_attempts
  for insert with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.quizzes q
      join public.documents d on d.id = q.document_id
      where q.id = quiz_attempts.quiz_id
        and d.owner_id = auth.uid()
    )
  );
