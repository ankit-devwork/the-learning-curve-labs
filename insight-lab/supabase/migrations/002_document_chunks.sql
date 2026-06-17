-- Step 1.7: document processing metadata + text chunks for RAG chat

alter table public.documents
  add column if not exists summary text,
  add column if not exists error_message text,
  add column if not exists processed_at timestamptz;

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents (id) on delete cascade,
  chunk_index int not null,
  content text not null,
  token_count int,
  created_at timestamptz not null default now(),
  unique (document_id, chunk_index)
);

create index if not exists document_chunks_document_id_idx
  on public.document_chunks (document_id);

alter table public.document_chunks enable row level security;
