-- Phase 2: concept graph metadata, mastery tracking, multi-doc vector search

-- Extracted concepts per document (Postgres source of truth; Neo4j mirrors when configured)
create table if not exists public.document_concepts (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  concept_id text not null,
  name text not null,
  topic text,
  chunk_indexes int[] not null default '{}',
  created_at timestamptz not null default now(),
  unique (document_id, concept_id)
);

create index if not exists document_concepts_document_id_idx
  on public.document_concepts (document_id);

-- Concept relationships within a document
create table if not exists public.concept_relationships (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  source_concept_id text not null,
  target_concept_id text not null,
  relationship_type text not null check (
    relationship_type in ('related_to', 'prerequisite_for', 'belongs_to')
  ),
  created_at timestamptz not null default now()
);

create index if not exists concept_relationships_document_id_idx
  on public.concept_relationships (document_id);

-- Per-user concept mastery from quiz attempts
create table if not exists public.concept_mastery (
  user_id uuid not null references public.profiles (id) on delete cascade,
  document_id uuid not null references public.documents (id) on delete cascade,
  concept_id text not null,
  attempts int not null default 0,
  correct int not null default 0,
  last_attempt_at timestamptz,
  primary key (user_id, document_id, concept_id)
);

create index if not exists concept_mastery_document_user_idx
  on public.concept_mastery (document_id, user_id);

-- Link quiz questions to concepts for adaptive targeting
alter table public.quiz_questions
  add column if not exists concept_id text;

-- Multi-document semantic chunk search (owned documents only — enforced in application layer)
create or replace function public.match_workspace_chunks(
  filter_document_ids uuid[],
  query_embedding vector(384),
  match_count int default 10
)
returns table (
  document_id uuid,
  chunk_index int,
  content text,
  similarity float
)
language sql
stable
as $$
  select
    dc.document_id,
    dc.chunk_index,
    dc.content,
    1 - (dc.embedding <=> query_embedding) as similarity
  from public.document_chunks dc
  where dc.document_id = any (filter_document_ids)
    and dc.embedding is not null
  order by dc.embedding <=> query_embedding
  limit match_count;
$$;

-- RLS for new tables
alter table public.document_concepts enable row level security;
alter table public.concept_relationships enable row level security;
alter table public.concept_mastery enable row level security;

create policy document_concepts_select_own on public.document_concepts
  for select using (
    exists (
      select 1 from public.documents d
      where d.id = document_concepts.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy document_concepts_insert_own on public.document_concepts
  for insert with check (
    exists (
      select 1 from public.documents d
      where d.id = document_concepts.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy document_concepts_delete_own on public.document_concepts
  for delete using (
    exists (
      select 1 from public.documents d
      where d.id = document_concepts.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy concept_relationships_select_own on public.concept_relationships
  for select using (
    exists (
      select 1 from public.documents d
      where d.id = concept_relationships.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy concept_relationships_insert_own on public.concept_relationships
  for insert with check (
    exists (
      select 1 from public.documents d
      where d.id = concept_relationships.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy concept_relationships_delete_own on public.concept_relationships
  for delete using (
    exists (
      select 1 from public.documents d
      where d.id = concept_relationships.document_id
        and d.owner_id = auth.uid()
    )
  );

create policy concept_mastery_select_own on public.concept_mastery
  for select using (auth.uid() = user_id);

create policy concept_mastery_insert_own on public.concept_mastery
  for insert with check (auth.uid() = user_id);

create policy concept_mastery_update_own on public.concept_mastery
  for update using (auth.uid() = user_id);
