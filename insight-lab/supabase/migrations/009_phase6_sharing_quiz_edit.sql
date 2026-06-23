-- Phase 6: Shared study sets, invites, quiz edit support, member-aware access

create table if not exists public.workspace_members (
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  role text not null check (role in ('owner', 'editor', 'viewer')),
  invited_by uuid references public.profiles (id),
  joined_at timestamptz not null default now(),
  primary key (workspace_id, user_id)
);

create index if not exists workspace_members_user_id_idx on public.workspace_members (user_id);

create table if not exists public.workspace_invites (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  email text not null,
  role text not null default 'viewer' check (role in ('editor', 'viewer')),
  token text not null unique default encode(gen_random_bytes(32), 'hex'),
  invited_by uuid not null references public.profiles (id),
  expires_at timestamptz not null default (now() + interval '7 days'),
  accepted_at timestamptz,
  created_at timestamptz not null default now(),
  unique (workspace_id, email)
);

create index if not exists workspace_invites_token_idx on public.workspace_invites (token);

-- Backfill owners as members for uniform membership queries
insert into public.workspace_members (workspace_id, user_id, role)
select id, owner_id, 'owner' from public.workspaces
on conflict do nothing;

alter table public.workspace_members enable row level security;
alter table public.workspace_invites enable row level security;

create or replace function public.is_workspace_member(
  p_workspace_id uuid,
  p_roles text[] default array['owner', 'editor', 'viewer']
)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
    from public.workspace_members m
    where m.workspace_id = p_workspace_id
      and m.user_id = auth.uid()
      and m.role = any (p_roles)
  );
$$;

-- Members: see own memberships and co-members on shared sets
create policy workspace_members_select on public.workspace_members
  for select using (
    user_id = auth.uid()
    or public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

create policy workspace_members_insert on public.workspace_members
  for insert with check (
    public.is_workspace_member(workspace_id, array['owner', 'editor'])
    or user_id = auth.uid()
  );

create policy workspace_members_delete on public.workspace_members
  for delete using (
    public.is_workspace_member(workspace_id, array['owner'])
    or user_id = auth.uid()
  );

-- Invites: owners/editors manage; invitee can read by token via backend
create policy workspace_invites_select on public.workspace_invites
  for select using (
    public.is_workspace_member(workspace_id, array['owner', 'editor'])
  );

create policy workspace_invites_insert on public.workspace_invites
  for insert with check (
    public.is_workspace_member(workspace_id, array['owner', 'editor'])
  );

create policy workspace_invites_delete on public.workspace_invites
  for delete using (
    public.is_workspace_member(workspace_id, array['owner', 'editor'])
  );

-- Member-aware read policies (additive — owner policies from 005 remain)
create policy workspaces_select_member on public.workspaces
  for select using (public.is_workspace_member(id, array['owner', 'editor', 'viewer']));

create policy documents_select_member on public.documents
  for select using (
    workspace_id is not null
    and public.is_workspace_member(workspace_id, array['owner', 'editor', 'viewer'])
  );

create policy quiz_questions_update_editor on public.quiz_questions
  for update using (
    exists (
      select 1
      from public.quizzes q
      join public.documents d on d.id = q.document_id
      where q.id = quiz_questions.quiz_id
        and (
          d.owner_id = auth.uid()
          or (
            d.workspace_id is not null
            and public.is_workspace_member(d.workspace_id, array['owner', 'editor'])
          )
        )
    )
  );

create policy quizzes_update_editor on public.quizzes
  for update using (
    exists (
      select 1
      from public.documents d
      where d.id = quizzes.document_id
        and (
          d.owner_id = auth.uid()
          or (
            d.workspace_id is not null
            and public.is_workspace_member(d.workspace_id, array['owner', 'editor'])
          )
        )
    )
  );
