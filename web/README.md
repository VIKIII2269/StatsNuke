# Stats Nuke website

A static site (no build step) on GitHub Pages. Accounts and data live in a free Supabase
project.

## Pages and who sees them

| App | What it shows | Who |
|---|---|---|
| Gameweek | The model team's plan for the next deadline, top picks by position, captain shortlist, team news | Friends + owner |
| Players | Every player: expected points for the next five gameweeks, P(start), P(goal), price, ownership, flags | Friends + owner |
| Model team | The model team's season: points per gameweek, running total against real managers | Friends + owner |
| Benchmarks | Against manager panels at p50 / p90 / p99, against the OpenFPL replica live, and the backtests | Friends + owner |
| Fixtures | Fixture ticker shaded by expected FPL points, fixtures and results by gameweek | Friends + owner |
| Ledger | The consensus-value paper bets with CLV (paper only) | Friends + owner |
| Scorers | The anytime-scorer forward test | Friends + owner |
| Lab | The experiment log, kept and rejected | Friends + owner |
| Health | GitHub Actions runs per workflow, live from the GitHub API | Owner |
| Data | Pipeline checks, feed freshness, silver table sizes | Owner |
| Access | Pending requests, approve / decline, revoke members | Owner |

Anyone can open the site. They see the sign-in screen, or **I'm a friend**, which creates an
account as a pending request. The owner approves or declines it under Access. An approved
friend sees the eight apps; Health, Data and Access stay with the owner. The database
enforces this with row-level security, not only the page.

## Data flow

The Live workflow (every 3 h) runs `fplh web export` (one JSON snapshot: forecasts, plan,
team, bets, benchmarks, checks, experiment log). A shell step then upserts it into the
`snapshots` table with the service-role key; the Python package itself makes no write calls. The page reads it with the
signed-in user's session. Run health refreshes live from the public GitHub API.

## One-time setup (about 10 minutes)

1. **Supabase.** Create a free project at supabase.com. Then:
   - In the SQL editor, run `web/supabase/schema.sql`.
   - Under Authentication → Providers → Email, turn **Confirm email** off. Friends can
     then sign up without email delivery; approval is the gate. If you keep it on, the site
     asks new accounts to confirm first.
2. **Repository variables** (Settings → Secrets and variables → Actions):
   - Variable `SUPABASE_URL`: the project URL.
   - Variable `SUPABASE_ANON_KEY`: the anon public key. It is public by design.
   - Secret `SUPABASE_SERVICE_ROLE_KEY`: the service-role key. Keep it secret: it is used
     only by the Live workflow to upload the snapshot.
3. **Pages.** Under Settings → Pages → Source, choose **GitHub Actions**. Then run the
   **Site** workflow, or push a change under `web/`.
4. **Become the owner.** On the site, use **I'm a friend** with your own email. Then, in
   the Supabase SQL editor, run:
   `update public.members set role = 'admin' where email = 'you@example.com';`
5. **Load the data.** Run the **Live** workflow once to upload the first snapshot.

## Local preview

`uv run fplh web export --out site.json --no-network` builds a snapshot from your local lake.
`uv run python scripts/build_site_preview.py site.json preview.html --document` then
bundles a single-file preview with the access flow simulated in memory.
