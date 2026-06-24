-- Workspace team chat (human-only, member-scoped)

create table if not exists public.workspace_messages (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  author_id uuid not null references public.profiles (id) on delete cascade,
  body text not null check (char_length(body) >= 1 and char_length(body) <= 2000),
  created_at timestamptz not null default now(),
  deleted_at timestamptz,
  deleted_by uuid references public.profiles (id) on delete set null
);

create index if not exists workspace_messages_workspace_created_idx
  on public.workspace_messages (workspace_id, created_at desc);

create index if not exists workspace_messages_author_idx
  on public.workspace_messages (author_id);

alter table public.workspace_messages enable row level security;

-- Members may read non-deleted messages in their study sheets only.
create policy workspace_messages_select_member on public.workspace_messages
  for select using (
    deleted_at is null
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

-- All members may post (owner, editor, viewer).
create policy workspace_messages_insert_member on public.workspace_messages
  for insert with check (
    author_id = auth.uid()
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

-- Authors may soft-delete their own messages; owners may soft-delete any message in their sheet.
create policy workspace_messages_update_delete on public.workspace_messages
  for update using (
    deleted_at is null
    and (
      author_id = auth.uid()
      or exists (
        select 1 from public.workspaces w
        where w.id = workspace_id and w.owner_id = auth.uid()
      )
    )
  )
  with check (
    deleted_at is not null
    and (
      author_id = auth.uid()
      or exists (
        select 1 from public.workspaces w
        where w.id = workspace_id and w.owner_id = auth.uid()
      )
    )
  );
