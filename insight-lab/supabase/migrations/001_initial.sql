-- InsightLab initial schema (Phase 1)
-- Apply via Supabase CLI: supabase db push
-- Or paste into Supabase SQL Editor

-- Profiles (extends auth.users)
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  avatar_url text,
  role text not null default 'user' check (role in ('user', 'teacher', 'admin')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Workspaces
create table if not exists public.workspaces (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.profiles (id) on delete cascade,
  name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Documents metadata (files live in Supabase Storage)
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  filename text not null,
  storage_path text not null,
  file_type text not null check (file_type in ('excel', 'document')),
  mime_type text,
  file_hash text,
  status text not null default 'pending' check (status in ('pending', 'processing', 'ready', 'failed')),
  neo4j_synced_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists documents_workspace_id_idx on public.documents (workspace_id);
create index if not exists documents_file_hash_idx on public.documents (file_hash);

-- Quizzes
create table if not exists public.quizzes (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  title text not null,
  question_type text not null default 'scq' check (question_type in ('scq', 'mcq', 'true_false')),
  difficulty text not null default 'medium',
  published boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Quiz questions
create table if not exists public.quiz_questions (
  id uuid primary key default gen_random_uuid(),
  quiz_id uuid not null references public.quizzes (id) on delete cascade,
  question_text text not null,
  options jsonb not null default '[]',
  correct_option_index int not null,
  explanation text,
  source_chunk_id text,
  sort_order int not null default 0,
  created_at timestamptz not null default now()
);

-- Quiz attempts
create table if not exists public.quiz_attempts (
  id uuid primary key default gen_random_uuid(),
  quiz_id uuid not null references public.quizzes (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  score int not null,
  total int not null,
  answers jsonb not null default '{}',
  completed_at timestamptz not null default now()
);

-- RLS (enable — policies added in Phase 1 implementation)
alter table public.profiles enable row level security;
alter table public.workspaces enable row level security;
alter table public.documents enable row level security;
alter table public.quizzes enable row level security;
alter table public.quiz_questions enable row level security;
alter table public.quiz_attempts enable row level security;
