-- Tier 1+2: public quiz sharing, excel-document links, public quiz attempts

alter table public.quizzes
  add column if not exists public_share_token text unique,
  add column if not exists public_share_enabled_at timestamptz;

create index if not exists quizzes_public_share_token_idx on public.quizzes (public_share_token)
  where public_share_token is not null;

create table if not exists public.quiz_public_attempts (
  id uuid primary key default gen_random_uuid(),
  quiz_id uuid not null references public.quizzes (id) on delete cascade,
  display_name text not null default 'Guest',
  score int not null,
  total int not null,
  answers jsonb not null default '{}',
  completed_at timestamptz not null default now()
);

create index if not exists quiz_public_attempts_quiz_id_idx on public.quiz_public_attempts (quiz_id);

create table if not exists public.document_source_links (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  excel_document_id uuid not null references public.documents (id) on delete cascade,
  document_id uuid not null references public.documents (id) on delete cascade,
  label text,
  created_by uuid not null references public.profiles (id) on delete cascade,
  created_at timestamptz not null default now(),
  unique (excel_document_id, document_id)
);

create index if not exists document_source_links_workspace_idx on public.document_source_links (workspace_id);
create index if not exists document_source_links_excel_idx on public.document_source_links (excel_document_id);

alter table public.quiz_public_attempts enable row level security;
alter table public.document_source_links enable row level security;

-- Source links: workspace members can read
create policy document_source_links_select_member on public.document_source_links
  for select using (
    public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

create policy document_source_links_insert_editor on public.document_source_links
  for insert with check (
    auth.uid() = created_by
    and public.is_workspace_member(workspace_id, array['owner', 'editor'])
  );

create policy document_source_links_delete_editor on public.document_source_links
  for delete using (
    public.is_workspace_member(workspace_id, array['owner', 'editor'])
  );
