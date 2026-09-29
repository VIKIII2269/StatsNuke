# Contract samples

Saved FPL API payloads validated by `tests/contracts` against `src/fplh/collectors/fpl_schema.py`.

| File | Provenance |
|---|---|
| `element-summary.json` | `history` rows are real 2026/27 GW1 values for element 1 (from vaastav `merged_gw.csv`, which mirrors the API's history fields). |
| `bootstrap-static.json`, `fixtures.json`, `event-live.json` | Hand-built from the documented API shape, trimmed to a few entries. The build sandbox could not reach the FPL API. |

**To do after the first live collector run:** replace the hand-built files with trimmed copies of real
bronze payloads, then keep them in sync whenever a contract test fails because FPL changed a field.

`understat/league.json` and `understat/match.json` are trimmed real responses from
`getLeagueData/EPL/2025` and `getMatchData/26602` (captured 29 Sep 2026; the rendered `tmpl`
HTML is dropped). The first hand-built versions used `datesData`/`rostersData` key names
taken from another scraper's code; the live API uses `dates`/`teams`/`players` and
`rosters`/`shots`, which these contract tests now pin.
