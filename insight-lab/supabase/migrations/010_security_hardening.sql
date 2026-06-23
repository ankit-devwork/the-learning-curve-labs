-- Security hardening: close member self-insert escalation and lock down workspace chunk RPC

-- Remove privilege escalation: users must not insert themselves into arbitrary workspaces
drop policy if exists workspace_members_insert on public.workspace_members;

create policy workspace_members_insert on public.workspace_members
  for insert with check (
    public.is_workspace_member(workspace_id, array['owner', 'editor'])
  );

-- Allow members to leave a shared set (delete own membership only)
drop policy if exists workspace_members_delete on public.workspace_members;

create policy workspace_members_delete on public.workspace_members
  for delete using (
    public.is_workspace_member(workspace_id, array['owner'])
    or (user_id = auth.uid() and role <> 'owner')
  );

-- match_workspace_chunks: service role only (same pattern as match_document_chunks)
revoke all on function public.match_workspace_chunks(uuid[], vector, int) from public;
revoke all on function public.match_workspace_chunks(uuid[], vector, int) from anon;
revoke all on function public.match_workspace_chunks(uuid[], vector, int) from authenticated;
grant execute on function public.match_workspace_chunks(uuid[], vector, int) to service_role;
