-- Phase 3/4: Study sets artifacts (flashcards, study guides) + workspace metadata

alter table public.workspaces
  add column if not exists description text;

create table if not exists public.study_guides (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  title text not null,
  content jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create index if not exists study_guides_document_id_idx on public.study_guides (document_id);
create index if not exists study_guides_owner_id_idx on public.study_guides (owner_id);

create table if not exists public.flashcard_sets (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  workspace_id uuid not null references public.workspaces (id) on delete cascade,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  title text not null,
  created_at timestamptz not null default now()
);

create index if not exists flashcard_sets_document_id_idx on public.flashcard_sets (document_id);
create index if not exists flashcard_sets_owner_id_idx on public.flashcard_sets (owner_id);

create table if not exists public.flashcards (
  id uuid primary key default gen_random_uuid(),
  set_id uuid not null references public.flashcard_sets (id) on delete cascade,
  front text not null,
  back text not null,
  sort_order int not null default 0,
  source_chunk_index int,
  created_at timestamptz not null default now()
);

create index if not exists flashcards_set_id_idx on public.flashcards (set_id);

create table if not exists public.flashcard_reviews (
  id uuid primary key default gen_random_uuid(),
  flashcard_id uuid not null references public.flashcards (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  knew boolean not null,
  reviewed_at timestamptz not null default now()
);

create index if not exists flashcard_reviews_user_id_idx on public.flashcard_reviews (user_id);

alter table public.study_guides enable row level security;
alter table public.flashcard_sets enable row level security;
alter table public.flashcards enable row level security;
alter table public.flashcard_reviews enable row level security;

-- Study guides: owner only
create policy study_guides_select_own on public.study_guides
  for select using (auth.uid() = owner_id);

create policy study_guides_insert_own on public.study_guides
  for insert with check (auth.uid() = owner_id);

create policy study_guides_delete_own on public.study_guides
  for delete using (auth.uid() = owner_id);

-- Flashcard sets: owner only
create policy flashcard_sets_select_own on public.flashcard_sets
  for select using (auth.uid() = owner_id);

create policy flashcard_sets_insert_own on public.flashcard_sets
  for insert with check (auth.uid() = owner_id);

create policy flashcard_sets_delete_own on public.flashcard_sets
  for delete using (auth.uid() = owner_id);

-- Flashcards: via set ownership
create policy flashcards_select_own on public.flashcards
  for select using (
    exists (
      select 1 from public.flashcard_sets s
      where s.id = flashcards.set_id and s.owner_id = auth.uid()
    )
  );

create policy flashcards_insert_own on public.flashcards
  for insert with check (
    exists (
      select 1 from public.flashcard_sets s
      where s.id = flashcards.set_id and s.owner_id = auth.uid()
    )
  );

-- Flashcard reviews: own reviews on owned sets
create policy flashcard_reviews_select_own on public.flashcard_reviews
  for select using (auth.uid() = user_id);

create policy flashcard_reviews_insert_own on public.flashcard_reviews
  for insert with check (
    auth.uid() = user_id
    and exists (
      select 1
      from public.flashcards fc
      join public.flashcard_sets s on s.id = fc.set_id
      where fc.id = flashcard_reviews.flashcard_id
        and s.owner_id = auth.uid()
    )
  );
