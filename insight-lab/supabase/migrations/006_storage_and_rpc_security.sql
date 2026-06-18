-- Storage policies for private uploads bucket (path: {user_id}/{document_id}/{filename})
-- Run after bucket `uploads` exists in Supabase Storage.

create policy storage_uploads_select_own on storage.objects
  for select to authenticated
  using (
    bucket_id = 'uploads'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy storage_uploads_insert_own on storage.objects
  for insert to authenticated
  with check (
    bucket_id = 'uploads'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy storage_uploads_update_own on storage.objects
  for update to authenticated
  using (
    bucket_id = 'uploads'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy storage_uploads_delete_own on storage.objects
  for delete to authenticated
  using (
    bucket_id = 'uploads'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

-- Restrict vector search RPC to service role (backend validates ownership)
revoke all on function public.match_document_chunks(uuid, vector, int) from public;
revoke all on function public.match_document_chunks(uuid, vector, int) from anon;
revoke all on function public.match_document_chunks(uuid, vector, int) from authenticated;
grant execute on function public.match_document_chunks(uuid, vector, int) to service_role;
