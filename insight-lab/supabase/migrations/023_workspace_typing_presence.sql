-- Member-scoped typing presence (RLS-protected; no public Realtime broadcast channels)

create table if not exists public.workspace_typing_presence (
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  display_name text not null default 'Member',
  updated_at timestamptz not null default now(),
  primary key (workspace_id, user_id)
);

create index if not exists workspace_typing_presence_workspace_updated_idx
  on public.workspace_typing_presence (workspace_id, updated_at desc);

alter table public.workspace_typing_presence enable row level security;

-- Members may see who is typing in their study sheets.
create policy workspace_typing_presence_select_member on public.workspace_typing_presence
  for select using (
    public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

-- Members may upsert/delete only their own typing row.
create policy workspace_typing_presence_insert_own on public.workspace_typing_presence
  for insert with check (
    auth.uid() = user_id
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

create policy workspace_typing_presence_update_own on public.workspace_typing_presence
  for update using (
    auth.uid() = user_id
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  )
  with check (
    auth.uid() = user_id
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

create policy workspace_typing_presence_delete_own on public.workspace_typing_presence
  for delete using (
    auth.uid() = user_id
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

alter table public.workspace_typing_presence replica identity full;

do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    if not exists (
      select 1
      from pg_publication_tables
      where pubname = 'supabase_realtime'
        and schemaname = 'public'
        and tablename = 'workspace_typing_presence'
    ) then
      alter publication supabase_realtime add table public.workspace_typing_presence;
    end if;
  end if;
end $$;
