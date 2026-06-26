-- Team chat UX: read cursors, per-message read receipts, Realtime on reads

create table if not exists public.workspace_chat_read_state (
  user_id uuid not null references public.profiles (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  last_read_at timestamptz not null default now(),
  last_read_message_id uuid references public.workspace_messages (id) on delete set null,
  updated_at timestamptz not null default now(),
  primary key (user_id, workspace_id)
);

create index if not exists workspace_chat_read_state_workspace_idx
  on public.workspace_chat_read_state (workspace_id);

create table if not exists public.workspace_message_reads (
  message_id uuid not null references public.workspace_messages (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  read_at timestamptz not null default now(),
  primary key (message_id, user_id)
);

create index if not exists workspace_message_reads_user_idx
  on public.workspace_message_reads (user_id);

alter table public.workspace_chat_read_state enable row level security;
alter table public.workspace_message_reads enable row level security;

-- Users manage their own read cursor for sheets they belong to.
create policy workspace_chat_read_state_select_own on public.workspace_chat_read_state
  for select using (auth.uid() = user_id);

create policy workspace_chat_read_state_insert_own on public.workspace_chat_read_state
  for insert with check (
    auth.uid() = user_id
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

create policy workspace_chat_read_state_update_own on public.workspace_chat_read_state
  for update using (auth.uid() = user_id)
  with check (
    auth.uid() = user_id
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

-- Message authors can see who read their messages; users see their own reads.
create policy workspace_message_reads_select on public.workspace_message_reads
  for select using (
    auth.uid() = user_id
    or exists (
      select 1
      from public.workspace_messages wm
      where wm.id = message_id
        and wm.author_id = auth.uid()
        and wm.deleted_at is null
    )
  );

create policy workspace_message_reads_insert_own on public.workspace_message_reads
  for insert with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.workspace_messages wm
      where wm.id = message_id
        and wm.deleted_at is null
        and wm.author_id <> auth.uid()
        and public.is_workspace_member(wm.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );

-- Realtime for read receipts (optional client refresh)
alter publication supabase_realtime add table public.workspace_message_reads;
