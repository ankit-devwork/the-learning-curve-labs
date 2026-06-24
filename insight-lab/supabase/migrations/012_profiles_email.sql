-- Sync auth email onto profiles for sharing / invites

alter table public.profiles
  add column if not exists email text;

create index if not exists profiles_email_idx on public.profiles (lower(email));

-- Backfill from Supabase Auth
update public.profiles p
set email = u.email
from auth.users u
where u.id = p.id
  and (p.email is null or p.email is distinct from u.email);

-- Keep profiles.email in sync when users sign up or change email
create or replace function public.sync_profile_email_from_auth()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, role, email)
  values (new.id, 'user', new.email)
  on conflict (id) do update
    set email = excluded.email,
        updated_at = now();
  return new;
end;
$$;

drop trigger if exists on_auth_user_created_sync_profile on auth.users;
create trigger on_auth_user_created_sync_profile
  after insert or update of email on auth.users
  for each row execute function public.sync_profile_email_from_auth();
