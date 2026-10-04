# API-Football free-plan probe (Premier League)

Date: 2026-10-04. Host: `https://v3.football.api-sports.io`, auth via the
`x-apisports-key` header (key taken from `FPLH_API_FOOTBALL_KEY`, not stored
here). 29 requests were sent in total. JSON samples are in
`docs/api_football_samples/`, with the key and account name/email removed.

## Summary

- **Verdict: not usable for historical backtests; usable only live.** The
  injury list has no timestamp showing when each record was created. Its
  "Questionable" entries also look reconciled after the match: 0 of 15 such
  players played or were on the bench.
- **The account is now suspended.** Every call, `/status` included, returns
  `"Your account is suspended"`. It started right after a burst that went over
  the 10-requests-a-minute limit (details at the end). Live use is blocked until
  the account is reinstated in the API-Football dashboard.

## Plan

`GET /status` (first call): plan `Free`, active, subscription end
2027-10-04, requests `current: 0`, `limit_day: 100`. A later `/status` showed
`current: 9` even though 25 requests had been sent by then. The counter seems
not to count every call, for example `/status` itself and calls refused by the
rate limit.

**Per-minute limit: 10 requests.** Going over it returns HTTP 200 with
`errors.rateLimit` and an empty `response`.

## Seasons covered

`GET /leagues?id=39` lists the seasons 2010–2026 (2026 is current). These are
the `coverage` flags the API reports for each season. They describe the API's
data, not what the free plan can access.

| Seasons | injuries | lineups | statistics_players | odds | predictions |
|---|---|---|---|---|---|
| 2010–2013 | false | true | false | false | true |
| 2014–2019 | false | true | true | false | true |
| 2020–2025 | true | true | true | false | true |
| 2026 (current) | true | true | true | true | true |

**What the free plan can access:** the API's own error message says
`"Free plans do not have access to this season, try from 2022 to 2024."` I got
that message for 2025 and 2026. 2023 worked. The 2024 call came back
"account suspended", so its access was not tested directly, but the message
says 2024 is included. **The free plan covers 2022–2024 only, so it gets no
current-season data.**

## Endpoint availability (season 2023, free plan)

| Endpoint | Result |
|---|---|
| `/injuries?league=39&season=2023` | OK, 3,853 records, one page (`paging.total = 1`) |
| `/fixtures?league=39&season=2023&round=Regular Season - 20` | OK, 10 fixtures |
| `/fixtures/lineups?fixture=ID` | OK, 12 fixtures (starting XI plus 8–9 substitutes per team) |
| `/fixtures/players?fixture=1035370` | OK, full per-player match stats (minutes, rating, shots, passes, tackles, duels, cards, …) |
| `/odds?league=39&season=2023&fixture=1035370` | No error but **0 results**. Odds are not kept for past fixtures (`coverage.odds = false`). |

## Injuries: fields and timing

Fields in each record:

- `player.{id, name, photo, type, reason}`
- `team.{id, name, logo}`
- `fixture.{id, timezone, date, timestamp}`
- `league.{id, season, name, country, logo, flag}`

**There is no field saying when a record was reported, created or updated.** The
only time fields are the fixture's kickoff time (`fixture.date` and
`fixture.timestamp`). Each record is tied to a fixture, not to a report date.

- `type` values: `Missing Fixture` (3,228) and `Questionable` (625).
- `reason`: 44 distinct values.
  - Most common: Knee Injury 784, Thigh Injury 607, Ankle Injury 426, Injury
    391, Muscle Injury 379, Calf Injury 234, Groin Injury 106, Knock 90.
  - Red Card 75, Illness 66, Suspended 61, National selection 53, Yellow Cards
    45, International duty 38.
  - Also present: Coach's decision 11, Loan agreement 14, Rest 4, Inactive 7.
- Coverage: all 380 fixtures, 1–20 records per fixture, 425 distinct players.

Sample records:

| Player | Team | type | reason | Fixture (kickoff UTC) |
|---|---|---|---|---|
| B. Mee | Brentford | Missing Fixture | Knock | 1035103 (2023-10-01 13:00) |
| N. Redmond | Burnley | Missing Fixture | Muscle Injury | 1035407 (2024-02-10 15:00) |
| N. Zaniolo | Aston Villa | Missing Fixture | Muscle Injury | 1035515 (2024-04-27 19:00) |
| L. Dobbin | Everton | Missing Fixture | Ankle Injury | 1035507 (2024-04-21 12:30) |
| J. Egan | Sheffield Utd | Missing Fixture | Ankle Injury | 1035491 (2024-04-07 16:30) |

## Leakage test

**Method.** I fetched lineups for 12 fixtures from 2023: all 10 in round 20
(30 Dec 2023 – 2 Jan 2024) and 2 in round 1 (1035042, 1035043). I matched each
injury record for those fixtures to the starting XI and substitutes by
`player.id`. For fixture 1035370 I also checked `/fixtures/players`.

| Injury type | Records | Started | On bench | Not in squad |
|---|---|---|---|---|
| Missing Fixture | 124 | 0 | 0 | 124 |
| Questionable | 15 | 0 | 0 | **15** |

**Evidence**

- **No "Missing Fixture" player ever appears in a lineup** (0 of 124). In
  fixture 1035370 none of them appears in `/fixtures/players` either.
- **Every "Questionable" player missed the match** (15 of 15). Examples:
  - Shaw, Martial and Amrabat for Man Utd at Forest;
  - Zouma, Aguerd and Kudus for West Ham against Brighton;
  - Partey for Arsenal at Fulham;
  - Lavia for Chelsea at Luton.

  A list drawn up before the match would include some doubtful players who
  then play or make the bench. A 100% miss rate suggests the stored list was
  pruned or rewritten after the match, keeping only players who did not play.
- **No timestamp on any record**, so there is no way to rebuild what the list
  showed at the FPL deadline (about 90 minutes before the round's first
  kickoff).

**Caveat.** The sample is small: 12 fixtures and 15 Questionable records. Even
so, the chance of 15 of 15 doubtful players all missing out is very low if the
list was genuinely made before the match. Suspensions (Red Card, Yellow Cards,
Suspended) are known in advance anyway, but they make up only about 5% of the
records.

**Conclusion.** The stored historical injury list cannot be trusted to reflect
what was known before the FPL deadline. Treat it as **post-hoc**. Used as a
feature in backtests, it would leak who actually played.

## Recommendation

**Usable only live; not usable for historical backtests.**

- **Historical injuries:** do not use them as forecast features. Every record
  is keyed to its fixture and none is timestamped. The Questionable entries
  look reconciled after the match.
- **Live injuries:** these would be usable only if we snapshot
  `/injuries?league=39&season=CURRENT` ourselves before each deadline, with our
  own timestamps. **The free plan rules this out**: it cannot access the
  current season (2025 and 2026 are refused).
- **Lineups and `/fixtures/players` for 2022–2024:** these are post-match facts.
  They are fine as targets or labels, and as lagged features from earlier
  gameweeks, but they add little over the FPL API's own history.
- **Odds:** not available for past seasons. Not usable for backtests.
- **Net:** on the free plan, API-Football adds no leakage-safe signal for our
  forecasts. A paid plan with live snapshotting would be the only route.

## Incident: account suspended during the probe

Sequence of events:

1. 14 requests were sent within about a minute. 6 of them were refused with
   `errors.rateLimit` (limit: 10 a minute).
2. The 6 refused requests were retried, spaced 7 seconds apart after a
   65-second pause. All succeeded.
3. The next calls (`/injuries` for 2024, then `/status`) returned
   `errors.access = "Your account is suspended, check on https://dashboard.api-football.com."`.

The `/injuries` calls for 2025 and 2026, sent in between, still returned the
free-plan season error. The suspension most likely comes from going over the
per-minute limit. I sent no further requests after that.

**To do:** check and reinstate the account in the API-Football dashboard. Any
future client must throttle to at most 10 requests a minute.
