-- Document infographics (structured JSON artifacts from LLM)

create table if not exists public.document_infographics (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  title text not null,
  content jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists document_infographics_document_id_idx on public.document_infographics (document_id);
create index if not exists document_infographics_owner_id_idx on public.document_infographics (owner_id);

alter table public.document_infographics enable row level security;

create policy document_infographics_select_own on public.document_infographics
  for select using (auth.uid() = owner_id);

create policy document_infographics_insert_own on public.document_infographics
  for insert with check (auth.uid() = owner_id);

create policy document_infographics_delete_own on public.document_infographics
  for delete using (auth.uid() = owner_id);

-- Workspace members can read shared infographics
create policy document_infographics_select_member on public.document_infographics
  for select using (
    exists (
      select 1
      from public.documents d
      where d.id = document_infographics.document_id
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );
