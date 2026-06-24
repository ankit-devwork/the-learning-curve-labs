-- Study session progress + learning paths

create table if not exists public.learning_paths (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  created_by uuid not null references public.profiles (id) on delete cascade,
  title text not null,
  path_type text not null default 'generated'
    check (path_type in ('generated', 'manual')),
  created_at timestamptz not null default now()
);

create index if not exists learning_paths_workspace_idx on public.learning_paths (workspace_id);

create table if not exists public.learning_path_nodes (
  id uuid primary key default gen_random_uuid(),
  path_id uuid not null references public.learning_paths (id) on delete cascade,
  sort_order int not null,
  node_kind text not null check (node_kind in ('concept', 'document')),
  document_id uuid references public.documents (id) on delete cascade,
  concept_id text,
  concept_name text,
  topic text,
  metadata jsonb not null default '{}',
  unique (path_id, sort_order)
);

create index if not exists learning_path_nodes_path_idx on public.learning_path_nodes (path_id);

create table if not exists public.study_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles (id) on delete cascade,
  workspace_id uuid references public.workspaces (id) on delete cascade,
  document_id uuid references public.documents (id) on delete cascade,
  learning_path_id uuid references public.learning_paths (id) on delete set null,
  session_type text not null check (session_type in ('document', 'workspace')),
  status text not null default 'active'
    check (status in ('active', 'completed', 'abandoned')),
  plan_snapshot jsonb not null default '{}',
  current_step_index int not null default 0,
  started_at timestamptz not null default now(),
  last_activity_at timestamptz not null default now(),
  completed_at timestamptz,
  check (
    (session_type = 'document' and document_id is not null)
    or (session_type = 'workspace' and workspace_id is not null)
  )
);

create index if not exists study_sessions_user_active_idx
  on public.study_sessions (user_id, status, last_activity_at desc);

create table if not exists public.study_session_steps (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.study_sessions (id) on delete cascade,
  step_index int not null,
  step_type text not null,
  label text not null,
  payload jsonb not null default '{}',
  status text not null default 'pending'
    check (status in ('pending', 'in_progress', 'completed', 'skipped')),
  started_at timestamptz,
  completed_at timestamptz,
  unique (session_id, step_index)
);

create index if not exists study_session_steps_session_idx on public.study_session_steps (session_id);

alter table public.quiz_attempts
  add column if not exists study_session_step_id uuid
    references public.study_session_steps (id) on delete set null;

alter table public.study_sessions enable row level security;
alter table public.study_session_steps enable row level security;
alter table public.learning_paths enable row level security;
alter table public.learning_path_nodes enable row level security;

-- Study sessions: owner of session row
create policy study_sessions_select_own on public.study_sessions
  for select using (auth.uid() = user_id);

create policy study_sessions_insert_own on public.study_sessions
  for insert with check (auth.uid() = user_id);

create policy study_sessions_update_own on public.study_sessions
  for update using (auth.uid() = user_id);

-- Steps: via session ownership
create policy study_session_steps_select_own on public.study_session_steps
  for select using (
    exists (
      select 1 from public.study_sessions s
      where s.id = study_session_steps.session_id and s.user_id = auth.uid()
    )
  );

create policy study_session_steps_update_own on public.study_session_steps
  for update using (
    exists (
      select 1 from public.study_sessions s
      where s.id = study_session_steps.session_id and s.user_id = auth.uid()
    )
  );

create policy study_session_steps_insert_own on public.study_session_steps
  for insert with check (
    exists (
      select 1 from public.study_sessions s
      where s.id = study_session_steps.session_id and s.user_id = auth.uid()
    )
  );

-- Learning paths: workspace members can read; editors can insert
create policy learning_paths_select_member on public.learning_paths
  for select using (
    exists (
      select 1 from public.workspaces w
      where w.id = learning_paths.workspace_id
        and (w.owner_id = auth.uid() or exists (
          select 1 from public.workspace_members m
          where m.workspace_id = w.id and m.user_id = auth.uid()
        ))
    )
  );

create policy learning_paths_insert_editor on public.learning_paths
  for insert with check (
    exists (
      select 1 from public.workspaces w
      where w.id = learning_paths.workspace_id
        and (w.owner_id = auth.uid() or exists (
          select 1 from public.workspace_members m
          where m.workspace_id = w.id and m.user_id = auth.uid()
            and m.role in ('owner', 'editor')
        ))
    )
  );

create policy learning_path_nodes_select_member on public.learning_path_nodes
  for select using (
    exists (
      select 1 from public.learning_paths p
      join public.workspaces w on w.id = p.workspace_id
      where p.id = learning_path_nodes.path_id
        and (w.owner_id = auth.uid() or exists (
          select 1 from public.workspace_members m
          where m.workspace_id = w.id and m.user_id = auth.uid()
        ))
    )
  );

create policy learning_path_nodes_insert_editor on public.learning_path_nodes
  for insert with check (
    exists (
      select 1 from public.learning_paths p
      join public.workspaces w on w.id = p.workspace_id
      where p.id = learning_path_nodes.path_id
        and (w.owner_id = auth.uid() or exists (
          select 1 from public.workspace_members m
          where m.workspace_id = w.id and m.user_id = auth.uid()
            and m.role in ('owner', 'editor')
        ))
    )
  );
