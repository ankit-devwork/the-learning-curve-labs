-- Enable Supabase Realtime for workspace team chat (member RLS applies to subscribers)

alter table public.workspace_messages replica identity full;

do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    if not exists (
      select 1
      from pg_publication_tables
      where pubname = 'supabase_realtime'
        and schemaname = 'public'
        and tablename = 'workspace_messages'
    ) then
      alter publication supabase_realtime add table public.workspace_messages;
    end if;
  end if;
end $$;
