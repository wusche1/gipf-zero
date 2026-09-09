# Fixed baseline evaluation

Baseline configuration v1.2 uses seeded tie-breaking, paired seeds and balanced colors. Each player has a 50 ms per-decision budget. Games stop at 240 insertion plies or 2,000 decisions; cutoffs are not counted as wins. See `baselines/README.md` for exact weights and search settings.

| Match | A wins | B wins | Cutoffs |
|---|---:|---:|---:|
| Random vs greedy | 0 | 40 | 0 |
| Random vs minimax depth 2 | 0 | 40 | 0 |
| Greedy vs minimax depth 2 | 12 | 26 | 2 |
| MCTS128 vs minimax depth 2 | 24 | 16 | 0 |
| MCTS128 vs random | 200 | 0 | 0 |

The 40-game comparisons are preliminary, with wide uncertainty. MCTS128 vs random included 100 games in each color, all wins. Canonical JSON reports contain the detailed settings and confidence intervals. Archived earlier runs are not used for these claims.
