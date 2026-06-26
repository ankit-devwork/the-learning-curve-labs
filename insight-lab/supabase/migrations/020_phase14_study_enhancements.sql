-- Phase 14: chat history, flashcard SRS, audio files, slide decks, homework solutions

-- Document RAG chat history (per user)
create table if not exists public.document_chat_messages (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  question text not null check (char_length(question) <= 2000),
  answer text not null check (char_length(answer) <= 16000),
  sources jsonb not null default '[]',
  retrieval_method text,
  cached boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists document_chat_messages_doc_user_idx
  on public.document_chat_messages (document_id, user_id, created_at desc);

-- Multi-document compare chat history (per user, per workspace)
create table if not exists public.workspace_compare_chat_messages (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  document_ids jsonb not null default '[]',
  question text not null check (char_length(question) <= 2000),
  answer text not null check (char_length(answer) <= 16000),
  sources jsonb not null default '[]',
  cached boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists workspace_compare_chat_ws_user_idx
  on public.workspace_compare_chat_messages (workspace_id, user_id, created_at desc);

-- Spaced repetition state for flashcards
create table if not exists public.flashcard_srs_states (
  user_id uuid not null references public.profiles (id) on delete cascade,
  flashcard_id uuid not null references public.flashcards (id) on delete cascade,
  interval_days int not null default 1 check (interval_days >= 1),
  due_at timestamptz not null default now(),
  ease_factor numeric(4, 2) not null default 2.50 check (ease_factor >= 1.30),
  repetitions int not null default 0 check (repetitions >= 0),
  updated_at timestamptz not null default now(),
  primary key (user_id, flashcard_id)
);

create index if not exists flashcard_srs_due_idx
  on public.flashcard_srs_states (user_id, due_at);

-- Persisted audio overviews with optional MP3 in Storage
create table if not exists public.document_audio_overviews (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  title text not null,
  script text not null,
  storage_path text,
  estimated_minutes int,
  created_at timestamptz not null default now()
);

create index if not exists document_audio_overviews_document_idx
  on public.document_audio_overviews (document_id, created_at desc);

-- Slide deck artifacts
create table if not exists public.document_slide_decks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  title text not null,
  content jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists document_slide_decks_document_idx
  on public.document_slide_decks (document_id);

-- Homework solution history (grounded step-by-step)
create table if not exists public.document_homework_solutions (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  question text not null check (char_length(question) <= 4000),
  answer jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists document_homework_doc_user_idx
  on public.document_homework_solutions (document_id, user_id, created_at desc);

-- RLS
alter table public.document_chat_messages enable row level security;
alter table public.workspace_compare_chat_messages enable row level security;
alter table public.flashcard_srs_states enable row level security;
alter table public.document_audio_overviews enable row level security;
alter table public.document_slide_decks enable row level security;
alter table public.document_homework_solutions enable row level security;

-- Chat: users see only their own messages; backend uses service role for writes
create policy document_chat_messages_select_own on public.document_chat_messages
  for select using (auth.uid() = user_id);

create policy workspace_compare_chat_select_own on public.workspace_compare_chat_messages
  for select using (auth.uid() = user_id);

-- SRS: own state only
create policy flashcard_srs_select_own on public.flashcard_srs_states
  for select using (auth.uid() = user_id);

create policy flashcard_srs_insert_own on public.flashcard_srs_states
  for insert with check (auth.uid() = user_id);

create policy flashcard_srs_update_own on public.flashcard_srs_states
  for update using (auth.uid() = user_id);

-- Audio / slides: owner + workspace members (read)
create policy document_audio_overviews_select_own on public.document_audio_overviews
  for select using (auth.uid() = owner_id);

create policy document_audio_overviews_insert_own on public.document_audio_overviews
  for insert with check (auth.uid() = owner_id);

create policy document_audio_overviews_select_member on public.document_audio_overviews
  for select using (
    exists (
      select 1 from public.documents d
      where d.id = document_audio_overviews.document_id
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

create policy document_slide_decks_select_own on public.document_slide_decks
  for select using (auth.uid() = owner_id);

create policy document_slide_decks_insert_own on public.document_slide_decks
  for insert with check (auth.uid() = owner_id);

create policy document_slide_decks_delete_own on public.document_slide_decks
  for delete using (auth.uid() = owner_id);

create policy document_slide_decks_select_member on public.document_slide_decks
  for select using (
    exists (
      select 1 from public.documents d
      where d.id = document_slide_decks.document_id
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Homework: own solutions only
create policy document_homework_select_own on public.document_homework_solutions
  for select using (auth.uid() = user_id);

create policy document_homework_insert_own on public.document_homework_solutions
  for insert with check (auth.uid() = user_id);
