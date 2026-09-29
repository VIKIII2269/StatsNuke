# Contract samples

Saved FPL API payloads validated by `tests/contracts` against `src/fplh/collectors/fpl_schema.py`.

| File | Provenance |
|---|---|
| `element-summary.json` | `history` rows are real 2026/27 GW1 values for element 1 (from vaastav `merged_gw.csv`, which mirrors the API's history fields). |
| `bootstrap-static.json`, `fixtures.json`, `event-live.json` | Hand-built from the documented API shape, trimmed to a few entries. The build sandbox could not reach the FPL API. |

**To do after the first live collector run:** replace the hand-built files with trimmed copies of real
bronze payloads, then keep them in sync whenever a contract test fails because FPL changed a field.
