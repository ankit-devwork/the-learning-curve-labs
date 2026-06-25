-- Storage read access for workspace members on shared study sheet documents
-- Path layout: uploads/{owner_user_id}/{document_id}/{filename}

create policy storage_uploads_select_workspace_member on storage.objects
  for select to authenticated
  using (
    bucket_id = 'uploads'
    and exists (
      select 1
      from public.documents d
      where d.id::text = (storage.foldername(name))[2]
        and d.workspace_id is not null
        and public.is_workspace_member(d.workspace_id, array['owner', 'editor', 'viewer'])
    )
  );
