-- Document storage encryption flag and editor delete policy

alter table public.documents
  add column if not exists storage_encrypted boolean not null default false;

comment on column public.documents.storage_encrypted is
  'True when the storage blob is encrypted with DOCUMENT_STORAGE_ENCRYPTION_KEY before upload.';

-- Workspace editors may delete documents they can edit (defense-in-depth; backend enforces editor+).
create policy documents_delete_editor on public.documents
  for delete using (
    auth.uid() = owner_id
    or exists (
      select 1
      from public.workspace_members wm
      where wm.workspace_id = documents.workspace_id
        and wm.user_id = auth.uid()
        and wm.role in ('owner', 'editor')
    )
  );
