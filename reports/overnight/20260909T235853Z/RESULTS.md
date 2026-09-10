# Overnight evaluation — `20260909T235853Z`

**Status:** `partial`

> This experiment is incomplete or still running. Missing matches are shown as unplayed; no winner is inferred from partial coverage.

## Protocol

- Training: isolated **600 seconds per run**, 2 seed(s), native forest search: **True**.
- Primary matches: 12 paired-colour games per seed and pair, **50 ms per decision** on one CPU thread for each player. Secondary matches use **32 search simulations per decision** on GPU.
- Whole-match timeouts are 180 seconds (CPU) and 45 seconds (fixed simulations); unfinished games are reported separately.
- Training seconds are nominal allocations; the elapsed column includes checkpoint and shutdown overhead.
- Parameter counts are read from each run's initial `start` metric, before training.
- This is **not a pure architecture ablation**: the compared models use different output heads (flat, query).
- Results are automated game comparisons; this report makes no human-rating claim.

## Training

| Model | Seed | Games | Updates | Seconds | Parameters | State |
|---|---|---|---|---|---|---|
| mlp256 | 410101 | 8740 | 8640 | 609.3 | 979371 | complete |
| mlp256 | 410102 | 8122 | 8032 | 611.9 | 979371 | complete |
| squarecnn32 | 410101 | 6214 | 6144 | 603.0 | 1163523 | complete |
| squarecnn32 | 410102 | 8449 | 8320 | 602.7 | 1163523 | complete |
| hexcnn32 | 410101 | 7877 | 7744 | 602.7 | 1163523 | complete |
| hexcnn32 | 410102 | 6813 | 6688 | 602.6 | 1163523 | complete |
| querytransformer256x2 | 410101 | 524 | 480 | 601.2 | 1102124 | complete |
| querytransformer256x2 | 410102 | 717 | 704 | 601.0 | 1102124 | complete |
| querytransformer164x5 | 410101 | 582 | 550 | 600.7 | 1115664 | complete |
| querytransformer164x5 | 410102 | 476 | 448 | 601.7 | 1115664 | complete |

## Pooled equal-CPU ranking

Complete primary attempts only; retries use the last complete attempt for each pair/mode/seed. Cutoffs count as games and Wilson intervals use all recorded games.

| Model | W | L | C | Played | Unplayed | Win rate | Wilson 95% |
|---|---|---|---|---|---|---|---|
| squarecnn32 | 80 | 15 | 1 | 96 | 0 | 83.3% | 74.6%–89.5% |
| hexcnn32 | 74 | 15 | 7 | 96 | 0 | 77.1% | 67.7%–84.4% |
| mlp256 | 41 | 40 | 15 | 96 | 0 | 42.7% | 33.3%–52.7% |
| querytransformer256x2 | 9 | 74 | 13 | 96 | 0 | 9.4% | 5.0%–16.9% |
| querytransformer164x5 | 7 | 67 | 22 | 96 | 0 | 7.3% | 3.6%–14.3% |

## Equal-CPU pairwise

| Pair (left perspective) | Left W | Left L | C | Games | Wilson 95% (left) |
|---|---|---|---|---|---|
| hexcnn32 vs mlp256 | 20 | 2 | 2 | 24 | 64.1%–93.3% |
| hexcnn32 vs querytransformer164x5 | 19 | 0 | 5 | 24 | 59.5%–90.8% |
| hexcnn32 vs querytransformer256x2 | 24 | 0 | 0 | 24 | 86.2%–100.0% |
| hexcnn32 vs squarecnn32 | 11 | 13 | 0 | 24 | 27.9%–64.9% |
| mlp256 vs querytransformer164x5 | 16 | 0 | 8 | 24 | 46.7%–82.0% |
| mlp256 vs querytransformer256x2 | 19 | 0 | 5 | 24 | 59.5%–90.8% |
| mlp256 vs squarecnn32 | 4 | 20 | 0 | 24 | 6.7%–35.9% |
| querytransformer164x5 vs querytransformer256x2 | 7 | 9 | 8 | 24 | 14.9%–49.2% |
| querytransformer164x5 vs squarecnn32 | 0 | 23 | 1 | 24 | 0.0%–13.8% |
| querytransformer256x2 vs squarecnn32 | 0 | 24 | 0 | 24 | 0.0%–13.8% |

## Fixed-simulation pairwise

Three incomplete fixed-simulation matches reached the original 45-second whole-match cap. Their verified frozen-checkpoint retries used a 180-second cap and replace only those rows below; raw initial and retry reports remain linked for audit. All 20 secondary pair/seed matches are now complete. The primary equal-CPU ranking above is unchanged.

| Pair (left perspective) | Left W | Left L | C | Games | Wilson 95% (left) |
|---|---|---|---|---|---|
| hexcnn32 vs mlp256 | 18 | 1 | 5 | 24 | 55.1%–88.0% |
| hexcnn32 vs querytransformer164x5 | 17 | 0 | 7 | 24 | 50.8%–85.1% |
| hexcnn32 vs querytransformer256x2 | 20 | 2 | 2 | 24 | 64.1%–93.3% |
| hexcnn32 vs squarecnn32 | 7 | 14 | 3 | 24 | 14.9%–49.2% |
| mlp256 vs querytransformer164x5 | 14 | 2 | 8 | 24 | 38.8%–75.5% |
| mlp256 vs querytransformer256x2 | 13 | 2 | 9 | 24 | 35.1%–72.1% |
| mlp256 vs squarecnn32 | 4 | 15 | 5 | 24 | 6.7%–35.9% |
| querytransformer164x5 vs querytransformer256x2 | 2 | 14 | 8 | 24 | 2.3%–25.8% |
| querytransformer164x5 vs squarecnn32 | 1 | 21 | 2 | 24 | 0.7%–20.2% |
| querytransformer256x2 vs squarecnn32 | 0 | 21 | 3 | 24 | 0.0%–13.8% |

## Extra tie-break attempts

Attempt 3 / 80-game tie-breaks are excluded from the primary pools above.

| Report | Candidate | Opponent | W | L | C | State |
|---|---|---|---|---|---|---|
| squarecnn32-seed410101-vs-hexcnn32-seed410102-equal_cpu_time-seed989001-attempt3.json | squarecnn32 | hexcnn32 | 42 | 38 | 0 | complete |

## Continuation league evaluations

These intermediate continuation snapshots are included for audit. They do not by themselves imply promotion; promotion requires its explicit gate.

| Report | Evaluation | Candidate games | W | L | C | Unfinished |
|---|---|---|---|---|---|---|
| [league-024737-duel.json](league-024737-duel.json) | duel | 21515 | 13 | 65 | 2 | 0 |
| [league-032337-duel.json](league-032337-duel.json) | duel | 38076 | 31 | 49 | 0 | 0 |
| [league-035932-duel.json](league-035932-duel.json) | duel | 54044 | 34 | 46 | 0 | 0 |
| [league-043512-duel.json](league-043512-duel.json) | duel | 69700 | 39 | 41 | 0 | 0 |

## Promotion and publication

- Promotion gate: `False`.
- `duel_report`: [final-vs-served.json](final-vs-served.json)
- Hugging Face: [wuschelschulz/gipf-zero](https://huggingface.co/wuschelschulz/gipf-zero)

## Raw duel reports

- [hexcnn32-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json](hexcnn32-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [hexcnn32-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json](hexcnn32-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json)
- [hexcnn32-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-retry-complete.json](hexcnn32-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-retry-complete.json)
- [hexcnn32-seed410101-vs-querytransformer256x2-seed410101-equal_cpu_time-seed1380101-attempt1.json](hexcnn32-seed410101-vs-querytransformer256x2-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [hexcnn32-seed410101-vs-querytransformer256x2-seed410101-equal_simulations-seed1380101-attempt1.json](hexcnn32-seed410101-vs-querytransformer256x2-seed410101-equal_simulations-seed1380101-attempt1.json)
- [hexcnn32-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json](hexcnn32-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [hexcnn32-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json](hexcnn32-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json)
- [hexcnn32-seed410102-vs-querytransformer256x2-seed410102-equal_cpu_time-seed1380102-attempt1.json](hexcnn32-seed410102-vs-querytransformer256x2-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [hexcnn32-seed410102-vs-querytransformer256x2-seed410102-equal_simulations-seed1380102-attempt1.json](hexcnn32-seed410102-vs-querytransformer256x2-seed410102-equal_simulations-seed1380102-attempt1.json)
- [mlp256-seed410101-vs-hexcnn32-seed410101-equal_cpu_time-seed1380101-attempt1.json](mlp256-seed410101-vs-hexcnn32-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-hexcnn32-seed410101-equal_simulations-seed1380101-attempt1.json](mlp256-seed410101-vs-hexcnn32-seed410101-equal_simulations-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json](mlp256-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json](mlp256-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-querytransformer256x2-seed410101-equal_cpu_time-seed1380101-attempt1.json](mlp256-seed410101-vs-querytransformer256x2-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-querytransformer256x2-seed410101-equal_simulations-seed1380101-attempt1.json](mlp256-seed410101-vs-querytransformer256x2-seed410101-equal_simulations-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-squarecnn32-seed410101-equal_cpu_time-seed1380101-attempt1.json](mlp256-seed410101-vs-squarecnn32-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [mlp256-seed410101-vs-squarecnn32-seed410101-equal_simulations-seed1380101-attempt1.json](mlp256-seed410101-vs-squarecnn32-seed410101-equal_simulations-seed1380101-attempt1.json)
- [mlp256-seed410102-vs-hexcnn32-seed410102-equal_cpu_time-seed1380102-attempt1.json](mlp256-seed410102-vs-hexcnn32-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-hexcnn32-seed410102-equal_simulations-seed1380102-attempt1.json](mlp256-seed410102-vs-hexcnn32-seed410102-equal_simulations-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json](mlp256-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json](mlp256-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-querytransformer256x2-seed410102-equal_cpu_time-seed1380102-attempt1.json](mlp256-seed410102-vs-querytransformer256x2-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-querytransformer256x2-seed410102-equal_simulations-seed1380102-attempt1.json](mlp256-seed410102-vs-querytransformer256x2-seed410102-equal_simulations-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-squarecnn32-seed410102-equal_cpu_time-seed1380102-attempt1.json](mlp256-seed410102-vs-squarecnn32-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [mlp256-seed410102-vs-squarecnn32-seed410102-equal_simulations-seed1380102-attempt1.json](mlp256-seed410102-vs-squarecnn32-seed410102-equal_simulations-seed1380102-attempt1.json)
- [querytransformer256x2-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json](querytransformer256x2-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [querytransformer256x2-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json](querytransformer256x2-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json)
- [querytransformer256x2-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-retry-complete.json](querytransformer256x2-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-retry-complete.json)
- [querytransformer256x2-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json](querytransformer256x2-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [querytransformer256x2-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json](querytransformer256x2-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json)
- [querytransformer256x2-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-retry-complete.json](querytransformer256x2-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-retry-complete.json)
- [squarecnn32-seed410101-vs-hexcnn32-seed410101-equal_cpu_time-seed1380101-attempt1.json](squarecnn32-seed410101-vs-hexcnn32-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [squarecnn32-seed410101-vs-hexcnn32-seed410101-equal_simulations-seed1380101-attempt1.json](squarecnn32-seed410101-vs-hexcnn32-seed410101-equal_simulations-seed1380101-attempt1.json)
- [squarecnn32-seed410101-vs-hexcnn32-seed410102-equal_cpu_time-seed989001-attempt3.json](squarecnn32-seed410101-vs-hexcnn32-seed410102-equal_cpu_time-seed989001-attempt3.json)
- [squarecnn32-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json](squarecnn32-seed410101-vs-querytransformer164x5-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [squarecnn32-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json](squarecnn32-seed410101-vs-querytransformer164x5-seed410101-equal_simulations-seed1380101-attempt1.json)
- [squarecnn32-seed410101-vs-querytransformer256x2-seed410101-equal_cpu_time-seed1380101-attempt1.json](squarecnn32-seed410101-vs-querytransformer256x2-seed410101-equal_cpu_time-seed1380101-attempt1.json)
- [squarecnn32-seed410101-vs-querytransformer256x2-seed410101-equal_simulations-seed1380101-attempt1.json](squarecnn32-seed410101-vs-querytransformer256x2-seed410101-equal_simulations-seed1380101-attempt1.json)
- [squarecnn32-seed410102-vs-hexcnn32-seed410102-equal_cpu_time-seed1380102-attempt1.json](squarecnn32-seed410102-vs-hexcnn32-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [squarecnn32-seed410102-vs-hexcnn32-seed410102-equal_simulations-seed1380102-attempt1.json](squarecnn32-seed410102-vs-hexcnn32-seed410102-equal_simulations-seed1380102-attempt1.json)
- [squarecnn32-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json](squarecnn32-seed410102-vs-querytransformer164x5-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [squarecnn32-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json](squarecnn32-seed410102-vs-querytransformer164x5-seed410102-equal_simulations-seed1380102-attempt1.json)
- [squarecnn32-seed410102-vs-querytransformer256x2-seed410102-equal_cpu_time-seed1380102-attempt1.json](squarecnn32-seed410102-vs-querytransformer256x2-seed410102-equal_cpu_time-seed1380102-attempt1.json)
- [squarecnn32-seed410102-vs-querytransformer256x2-seed410102-equal_simulations-seed1380102-attempt1.json](squarecnn32-seed410102-vs-querytransformer256x2-seed410102-equal_simulations-seed1380102-attempt1.json)
