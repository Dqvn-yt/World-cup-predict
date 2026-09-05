# Historical World Cups

This directory is a normalized historical archive and result-only score training input.
`python -m src.data.collect_worldcup` reuses the retained raw JSON and regenerates
the CSVs and audit. No 2026 CSV, clean data, feature table, or model is modified.

## Identity

- 40 distinct teams across 128 matches (64 per tournament).
- `team_mapping.csv` has a complete `team_id` and an optional `team_id_2026`.
  The 30 shared teams retain their existing IDs, joined by FIFA code, not name.
  This handles Iran / IR Iran without fuzzy matching.
- `historical_team_registry.csv` persists ten additional IDs: CMR 10000,
  CRC 10001, DEN 10002, ISL 10003, NGA 10004, PER 10005, POL 10006,
  RUS 10007, SRB 10008, WAL 10009. Do not discard this registry when refreshing.
  Duplicate identities and conflicts with current IDs fail normalization.
- Match IDs are `year * 1000 + match_no`: 2018001-2018064 and
  2022001-2022064. `match_no` is the position in the retained source JSON,
  not an official FIFA fixture number. IDs are stable for these pinned inputs;
  replacing/reordering upstream JSON requires an explicit identity migration.
  Normalization rejects collisions with existing 2026 match IDs (1-104).
- Goal event IDs identify source events, not people. All 341 `player_id` values
  remain empty: source names alone do not establish a match to 2026 players.

## Score And Time Contract

- `team*_ft` is the score after 90 minutes including stoppage time.
- `team*_et` is the cumulative score after extra time, empty when not played.
- `team*_extra_time_goals` is the extra-time-only increment.
- `team*_final` excludes shootout kicks. `team*_pens` is separate shootout scoring.
- `winner` includes shootout resolution. The 2022 final is 2-2 at 90 minutes,
  3-3 after extra time, and 4-2 on penalties to Argentina.
- Goal events exclude shootout kicks; `penalty` marks an in-match penalty goal.
  `team` and `credited_team_id` indicate the beneficiary of a goal.
  `scorer_team_id` is the opposing side for own goals. There are 14 own goals.
- `kickoff_source_time` preserves the supplied clock time. Source timezone is
  unverified and `kickoff_utc` is empty, not falsely labeled or guessed.
- `neutral` is false for matches involving the tournament host (Russia/Qatar);
  team1/team2 ordering does not establish home advantage.

## Provenance And Audits

Source URLs are recorded on matches/events and in `worldcup_audit.json`, alongside
SHA-256 hashes of retained raw inputs. The audit timestamp is normalization time,
not an invented retrieval timestamp; the original retrieval time is unknown.
The source is openfootball/worldcup.json (public-domain data).

Audits validate stage counts, champions, nonnegative scores, shootout/extra-time
semantics, match ID uniqueness/collisions, and goal counts per match and credited
team. There are 169 goal events in 2018 and 172 in 2022, with no score mismatches.
The legacy audit keys `mapped_teams`/`unmapped_teams` refer only to 2026 overlap;
`historical_identity_coverage` reports the complete 40-team identity coverage.

## Integration Boundary

`src/data/score_history.py` validates identities against source/clean 2026 teams,
fixtures, feature FIFA codes and targets before exposing six result-only columns.
All 128 historical fixtures and all 40 historical teams participate in Dixon-Coles
training, chronological OOF fits and cold-start scoring priors. Historical dates
use next-day availability, not invented UTC kickoffs. Each OOF fit selects only
strictly earlier results, including when multiple current fixtures share a time.

Historical rankings, Elo, squad values, lineups, xG and player IDs are unavailable.
No 2026 features are copied into old fixtures; ML still trains only on 2026 rows.
The ten archive-only teams are fitted internally but are not exposed as ML-backed
scenarios. Archive match IDs are not served by current match/detail APIs.

`python build.py` rebuilds all artifacts. Targets are 90-minute scores: historical
`team*_ft` and reconciled 2026 goal events. The 2026 source/clean tables retain
their original after-ET scores; only match-model targets and derived result-based
team features use regulation scores. Integer event minutes above 90 in AET games
are treated as extra time, with a required tied regulation result. New ambiguous
event formats must be reviewed rather than silently mixed.

DC uses a fixed 1461-day half-life measured at the prediction cutoff and ridge=1
to retain some past-tournament information while regularizing sparse teams.
Holdout models exclude all holdout results; scenario models are refitted on all
232 known results after evaluation. OOF rows persist their own rho. Evaluation
and serving use the same full-support blended probability matrix. See
`reports/worldcup_validation.md` for counts, comparisons and limitations.
