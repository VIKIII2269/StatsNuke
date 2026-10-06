// Filled in at deploy time from the repository variables SUPABASE_URL and
// SUPABASE_ANON_KEY (.github/workflows/site.yml). The anon key is public by design:
// row-level security in web/supabase/schema.sql decides what each account can read.
window.STATSNUKE_CONFIG = { supabaseUrl: "", supabaseAnonKey: "", repo: "VIKIII2269/StatsNuke" };
