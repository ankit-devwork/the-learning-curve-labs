-- Security hardening: RLS gaps, profile role lock, quiz public attempts

-- quiz_public_attempts: deny all direct client access (backend uses service role)
create policy quiz_public_attempts_deny_authenticated on public.quiz_public_attempts
  for all to authenticated
  using (false)
  with check (false);

create policy quiz_public_attempts_deny_anon on public.quiz_public_attempts
  for all to anon
  using (false)
  with check (false);

-- Prevent self-elevation of profiles.role via Supabase client
create or replace function public.profiles_lock_role_on_update()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  if new.role is distinct from old.role then
    new.role := old.role;
  end if;
  return new;
end;
$$;

drop trigger if exists profiles_lock_role on public.profiles;
create trigger profiles_lock_role
  before update on public.profiles
  for each row
  execute function public.profiles_lock_role_on_update();
