-- Step 1.7b: pgvector embeddings for semantic chunk retrieval

create extension if not exists vector;

alter table public.document_chunks
  add column if not exists embedding vector(384);

create index if not exists document_chunks_embedding_idx
  on public.document_chunks
  using hnsw (embedding vector_cosine_ops);

create or replace function public.match_document_chunks(
  filter_document_id uuid,
  query_embedding vector(384),
  match_count int default 6
)
returns table (
  id uuid,
  chunk_index int,
  content text,
  similarity float
)
language sql
stable
as $$
  select
    dc.id,
    dc.chunk_index,
    dc.content,
    1 - (dc.embedding <=> query_embedding) as similarity
  from public.document_chunks dc
  where dc.document_id = filter_document_id
    and dc.embedding is not null
  order by dc.embedding <=> query_embedding
  limit match_count;
$$;
