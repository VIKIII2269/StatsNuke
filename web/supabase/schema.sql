-- Stats Nuke: accounts, access requests and the site snapshot.
-- Run once in the Supabase SQL editor (safe to re-run).
--
-- Roles: admin (the owner), friend (approved), pending (requested), rejected (declined).
-- Every sign-up becomes a pending member; only an admin can change a role. The snapshot
-- is readable only by admins and friends and written only with the service-role key
-- (the Live workflow), which bypasses row-level security.

create table if not exists public.members (
  user_id      uuid primary key references auth.users (id) on delete cascade,
  email        text not null,
  name         text not null default '',
  note         text not null default '',
  role         text not null default 'pending'
               check (role in ('admin', 'friend', 'pending', 'rejected')),
  requested_at timestamptz not null default now(),
  decided_at   timestamptz,
  decided_by   uuid references auth.users (id)
);

create table if not exists public.snapshots (
  id         text primary key,
  data       jsonb not null,
  updated_at timestamptz not null default now()
);

alter table public.members enable row level security;
alter table public.snapshots enable row level security;

-- role checks run as the table owner so the policies below do not recurse
create or replace function public.is_admin() returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.members where user_id = auth.uid() and role = 'admin');
$$;

create or replace function public.is_member() returns boolean
language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.members where user_id = auth.uid() and role in ('admin', 'friend')
  );
$$;

-- a new account is a pending access request, with the name and note given at sign-up
create or replace function public.on_signup() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  insert into public.members (user_id, email, name, note)
  values (
    new.id,
    new.email,
    left(coalesce(new.raw_user_meta_data ->> 'name', ''), 80),
    left(coalesce(new.raw_user_meta_data ->> 'note', ''), 280)
  )
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists statsnuke_on_signup on auth.users;
create trigger statsnuke_on_signup
  after insert on auth.users
  for each row execute function public.on_signup();

-- only admins may change a role, and nobody may change their own
create or replace function public.guard_role() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  if new.user_id = auth.uid() and new.role is distinct from old.role then
    raise exception 'you cannot change your own access';
  end if;
  new.decided_by := auth.uid();
  return new;
end;
$$;

drop trigger if exists statsnuke_guard_role on public.members;
create trigger statsnuke_guard_role
  before update on public.members
  for each row execute function public.guard_role();

drop policy if exists "members: read own or all as admin" on public.members;
create policy "members: read own or all as admin" on public.members
  for select to authenticated
  using (user_id = auth.uid() or public.is_admin());

drop policy if exists "members: admins decide" on public.members;
create policy "members: admins decide" on public.members
  for update to authenticated
  using (public.is_admin()) with check (public.is_admin());

drop policy if exists "snapshots: members read" on public.snapshots;
create policy "snapshots: members read" on public.snapshots
  for select to authenticated
  using (public.is_member());

-- no insert/delete policies: members rows come from the trigger, snapshots from the
-- service role.

-- After signing up on the site yourself, make yourself the owner (replace the email):
--   update public.members set role = 'admin' where email = 'you@example.com';
-- (run it here in the SQL editor: the guard above only applies to signed-in users,
--  and here auth.uid() is null.)
