# World Cup Identity And Training Validation

Validated on 2026-09-05 in the existing Python 3.14 environment. This report
supersedes the earlier normalization-only report. Historical results are now
training inputs and all application artifacts have been rebuilt.

## Identity Audit

- Source and clean 2026 `teams.csv`: 48 identical FIFA-code/ID pairs.
- Source and clean fixtures, feature fixture/team IDs, feature FIFA codes, dates,
  kickoffs and original score targets reconcile one-to-one for 104 matches.
- Historical archive: 128 matches, 64 each in 2018/2022, 40 teams.
- All 30 shared teams retain their current ID, joined by FIFA code, not name.
- Ten historical-only IDs remain unchanged: CMR 10000, CRC 10001, DEN 10002,
  ISL 10003, NGA 10004, PER 10005, POL 10006, RUS 10007, SRB 10008, WAL 10009.
- Combined team universe: 58. All historical teams are included in DC fitting.
- Historical match IDs remain 2018001-2018064 and 2022001-2022064; current IDs
  remain 1-104. There are 232 unique IDs and zero collisions.
- Both stage/champion audits pass. All 341 historical goal events reconcile
  by fixture and credited team; all player identities remain unresolved.
- No player identity matching code was changed. Raw cached JSON and source 2026
  CSV/JSON inputs were not edited.

The collector reuses pinned raw inputs with SHA-256 hashes recorded in
`processed data/worldcup/worldcup_audit.json`. The original retrieval timestamp
is still unknown. Normalization time is not presented as retrieval time.

## Score Contract

The match model predicts 90 minutes including regulation stoppage time, excluding
extra time and shootouts. Archive targets are `team1_ft`/`team2_ft`; the 2022 final
is therefore 2-2, not 3-3 or a shootout win.

All 308 current non-shootout goal/own-goal events reconcile with original scores.
Own goals are credited to the opposing team. For the eight AET/Penalties matches,
integer goal-event minutes above 90 are excluded and a tied regulation score is
required. Five current targets change:

| Match ID | Original score | Regulation target |
| --- | --- | --- |
| 81 | 3-2 | 2-2 |
| 87 | 3-2 | 1-1 |
| 99 | 1-2 | 1-1 |
| 100 | 3-1 | 1-1 |
| 104 | 1-0 | 0-0 |

Source/clean score tables are retained unchanged. Match-model goals, derived
result label and result-based team features use regulation scores. Other existing
pre-match covariates retain their source semantics; they are not relabeled as
historical or regulation xG. The API returns `score_duration`, and match pages
explicitly label regulation forecasts/results.

## Training Protocol

History exposes only match ID, result availability date, two team IDs and two
regulation scores. No historical Elo, rankings, squads, lineups, player IDs or
observed xG are fabricated. No historical rows enter PoissonRegressor fitting.

- DC half-life is fixed at 1461 days, with ridge=1 on attack/defense coefficients
  and both venue advantages. Decay uses the prediction cutoff, including the
  gap between 2022 and the first 2026 fixture. Previous tiny ridge and extra
  OOF-only count shrinkage were replaced by consistent regularized fitting in
  OOF, holdout and serving.
- Two-level venue effect: `neutral=1` archive rows (120/128) and non-host 2026
  fixtures (91/104) estimate a neutral-venue nominal-home edge instead of
  borrowing the host advantage. Only host-involved fixtures (8 archive, 13
  current) estimate `home_advantage`. Final serving fit: host advantage
  0.5977 (x1.82, from 21 host fixtures, ridge-shrunk), neutral edge 0.2319
  (x1.26, close to the measured neutral home/away scoring ratio 1.29).
  `DixonColesModel.lambdas()` takes a `neutral` flag; team scenarios default
  to neutral because hypothetical World Cup venues are unknown.
- At final serving cutoff, 64 matches from 2018 contribute 15.8210 weighted
  matches, and 64 from 2022 contribute 34.0949. Total DC effective sample weight,
  including current matches, is 152.7361. Final DC optimization converged.
- Historical kickoff timezone remains unknown. Next-day availability is a
  conservative date-only training convention, not a claimed UTC kickoff. All
  historical records precede the evaluated 2026 fixtures by years.
- OOF fits use only results strictly earlier than the fixture time. Fixtures
  sharing a time cannot train on one another. Each row stores its own `dc_rho`
  and history counts. No final rho, warm-start parameters or validation-selected
  future ML alpha are reused for OOF; ML OOF alpha is fixed at 1.
- Before 20 current matches, the ML branch uses a time-weighted result prior,
  including available history. Thereafter ML fits only genuine current features.
- Missing OOF rows fall back to strictly earlier, time-weighted results, never
  to a future-trained scenario model.
- Validation selects ML alpha from 0.1/1/10: home=10, away=1. Holdout outcomes
  do not enter model fits or hyperparameter selection. Both components refit on
  all current data only after evaluation for undated current-team scenarios.

| Stage | Current rows | Historical rows | Notes |
| --- | ---: | ---: | --- |
| ML train | 72 | 0 | Validation selects alpha |
| ML validation | 16 | 0 | Chronological; no same-time split |
| Fixed holdout ML fit | 88 | 0 | Predicts 16 held-out fixtures |
| Fixed holdout DC fit | 88 | 128 | 216 total; predicts same 16 fixtures |
| First OOF DC fit | 0 | 128 | History supports opening cold start |
| Last OOF DC fit | 103 | 128 | Excludes match 104 itself |
| Serving ML fit | 104 | 0 | After evaluation |
| Serving DC fit | 104 | 128 | 232 rows, 58 teams |

OOF predicts all 104 current fixtures across 90 kickoff groups: 90 DC fits,
104 DC predictions, 84 ML fitted predictions and 20 ML-prior predictions.
There are no 2018/2022 evaluation rows.
The no-history DC baseline starts after eight current results, so the matched
OOF comparison uses the same 96 fixtures in both configurations.

OOF is sequential evaluation, including earlier results from the test period
when predicting later test fixtures. The separately reported fixed holdout does
not refit on test outcomes. Existing rolling pre-match covariates may incorporate
earlier completed holdout fixtures: this is a pre-match evaluation, not a forecast
of the entire remaining tournament issued at one fixed date.

## Probability Contract

Lambda is blended 0.7 DC / 0.3 ML with consistent ML clipping. DC lambdas carry
the venue level of their fixture (host edge or neutral edge); scenarios use the
neutral level. The full 0-25 matrix
is a 0.3 Poisson / 0.7 DC-corrected mixture at that blended lambda. Evaluation and
API serving call the same matrix function. WDL, RPS, BTTS, Over 2.5, log-likelihood,
modal score, top scorelines and tail coverage derive from that joint matrix.
BTTS is summed from joint cells, not calculated as independent marginals after
DC correction. The visible 0-6 grid is not independently renormalized.

## Matched Baselines

Both configurations use the same regulation targets, features, chronological
splits, fixed decay/ridge, alpha protocol and matrix correction. Only historical
results and historical cold-start priors are switched on/off. These are rebuilt
baselines, not comparisons against the previously leaked 75% accuracy.

| Ensemble metric | OOF no history (96) | OOF history (96) | Holdout no history (16) | Holdout history (16) |
| --- | ---: | ---: | ---: | ---: |
| WDL accuracy | 55.21% | 58.33% | 31.25% | 37.50% |
| Home goals MAE | 1.097546 | 1.071773 | 0.915893 | 0.962818 |
| Away goals MAE | 0.868840 | 0.871498 | 1.315763 | 1.335896 |
| Total goals MAE | 1.510203 | 1.532231 | 1.590386 | 1.686049 |
| Scoreline mean log-likelihood | -3.055273 | -3.038332 | -3.397657 | -3.325817 |
| RPS | 0.176499 | 0.164507 | 0.273575 | 0.252528 |
| BTTS Brier | 0.238288 | 0.246170 | 0.207560 | 0.216809 |
| Over 2.5 Brier | 0.251888 | 0.261314 | 0.225433 | 0.252795 |

MAE/RPS/Brier are better when lower; accuracy/log-likelihood when higher.
History improves matched OOF WDL accuracy and RPS, and holdout accuracy/RPS/
log-likelihood, but worsens several goal metrics. Sixteen holdout games are
insufficient evidence of a general predictive gain.

All-104 history OOF accuracy is 57.69%, RPS 0.167568 and mean scoreline
log-likelihood -3.005677. Ensemble goal bias is now symmetric: home -0.088,
away -0.096 goals (previously home -0.066, away -0.108 with the away
underestimation concentrated in the DC branch at -0.154). The predicted
home/away scoring ratio is 1.40 vs actual 1.36 (was 1.43). Holdout exact-score
accuracy is 12.50%; the majority outcome baseline is 18.75% WDL accuracy.
Component metrics, per-alpha validation MAE/bias tables and per-branch OOF
biases are retained in `reports/score_metrics.json`, including unchanged
holdout ML between baselines.

The dependent player-goal artifact was rebuilt using new OOF team intensities:
PR-AUC 0.222070, ROC-AUC 0.801397 (16 matches/832 player rows).
This is not a historical player-model integration or a new comprehensive audit
of player-goal evaluation. Its existing full-match player target was not changed.

## Serving Checks

| Request | HTTP | Source / lambdas home, away |
| --- | --- | --- |
| Match 1 | 200 | OOF ensemble / 1.122439, 1.019778 |
| Match 104 | 200 | OOF ensemble / 1.560387, 0.909700 |
| Teams 37/33 | 200 | Final ensemble, neutral venue / 1.449782, 1.577837 |
| Teams 9/29 | 200 | Final ensemble, neutral venue / 0.724107, 1.305804 |
| Archive match 2018001 | 400 | Explicit unknown match |
| Archive-only team 10000 vs 33 | 400 | Explicit unknown team |

Home, current match detail 1/104 and team comparison pages return 200; archive
detail 2018001 returns 404. All 104 fixture predictions and five current-team API
scenarios are covered by tests. Historical-only teams affect fitting internally,
but unsupported historical ML scenarios are not manufactured. Browser visual
layout was not exercised; existing responsive styles were unchanged.

## Commands And Tests

- `python -m src.data.collect_worldcup`: passed using retained raw cache.
- `python build.py`: all seven stages completed after final model edits.
- `python -m pytest -q`: **64 passed**.
  Warnings are joblib NumPy-array shape deprecations, not failed checks.
  New tests cover the two-level venue mechanism and neutral-by-default team
  scenarios, alongside the existing scenario/matrix/probability checks.
- Direct Python/Flask test-client diagnostics verified the requests above, final
  model counts, convergence and weighted historical participation.

Tests cover identity integrity/corruption, registry repeatability, regulation
targets, all historical team participation, history changing predictions, ML fit
isolation, simultaneous/future-result exclusion, per-OOF rho, holdout fit
boundaries, decay across the tournament gap, missing-OOF fallback and joint
matrix/market/evaluation/API coherence. An initial added test compared index
order rather than match IDs; sorting both sides fixed the test assertion.

## Exact File Inventory

Manually changed or added this turn:

```text
src/data/collect_worldcup.py
src/data/score_history.py                 new
src/features/match_features.py
src/models/dixon_coles.py
src/models/poisson.py
src/models/train_score.py
src/services/match_predictor.py
tests/test_models.py
tests/test_score_history.py               new
templates/predict_match.html
templates/match_detail.html
templates/home.html
templates/compare.html
README.md
processed data/worldcup/README.md
reports/worldcup_validation.md
```

Rebuilt feature/model/report outputs (some unaffected outputs may be byte-identical):

```text
data/features/player_match_performance.csv
data/features/overall_player_form.csv
data/features/player_form_features.csv
data/features/team_player_form_features.csv
data/features/match_model_dataset.csv
data/features/match_expected_goals_oof.csv
data/features/goal_probability_dataset.csv
models/home_goal_model.joblib
models/away_goal_model.joblib
models/score_models.joblib
models/goal_probability_model.joblib
reports/data_audit.json
reports/score_metrics.json
reports/goal_metrics.json
```

Collector regenerated these existing archive outputs without reallocating IDs:

```text
processed data/worldcup/historical_team_registry.csv
processed data/worldcup/team_mapping.csv
processed data/worldcup/worldcup_2018_matches.csv
processed data/worldcup/worldcup_2022_matches.csv
processed data/worldcup/worldcup_history_matches.csv
processed data/worldcup/worldcup_goalscorers.csv
processed data/worldcup/worldcup_audit.json
```

Full build also refreshed the existing clean outputs, preserving source semantics:

```text
data/clean/gk.csv
data/clean/match_events.csv
data/clean/match_lineups.csv
data/clean/match_prediction_features_X.csv
data/clean/match_prediction_features.csv
data/clean/match_prediction_targets_y.csv
data/clean/match_team_stats.csv
data/clean/matches_detailed.csv
data/clean/matches.csv
data/clean/Miscellaneous.csv
data/clean/player_stats.csv
data/clean/real_match_details.json
data/clean/shoot.csv
data/clean/squads_and_players.csv
data/clean/teams.csv
data/clean/tournament_stages.csv
data/clean/venues.csv
```

There is no Git repository or pre-turn artifact hash manifest, so regenerated
outputs are listed separately from manual edits rather than claiming every
refreshed byte changed. Python/pytest caches are incidental execution outputs.

## Remaining Limits

Historical dates lack verified kickoff UTC; source ordering pins archive IDs.
New/reordered raw inputs require identity review. Current integer minute data
lacks explicit periods, so future ambiguous stoppage-time/ET formats must not
be inferred silently. The current event reconciliation and tied-score checks
validate this retained dataset, not every possible provider format.

The half-life/ridge/ensemble weights are pragmatic fixed choices, not proven
optimal. The host advantage rests on 21 host-involved fixtures and stays noisy
despite ridge shrinkage; the neutral edge rests on 211 fixtures and is stable.
Sparse international results, squad turnover, missing historical covariates
and uncertain source 2026 covariate provenance limit interpretation. Scenario
features remain latest available current pre-match states, not date-specific
future lineups. No production or betting-quality accuracy claim is made.
