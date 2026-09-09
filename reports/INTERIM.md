# Interim results — 9 September 2026, 16:22 UTC

These results describe the **1,721-game, flat-policy MLP-128 checkpoint**, not
all later candidates or the final opponent. The exact training checkpoint is
`pilot-flat-final.pt` in release v0.2.0. Training started from random weights and
used only self-play outcomes and search visit targets, with no expert games,
heuristic labels, or pretrained network.

| Opponent / setting | Wins | Losses | Cutoffs | Report |
|---|---:|---:|---:|---|
| Random, neural 128 simulations | 40 | 0 | 0 | [data](pilot_mlp128-vs-random.json) |
| Greedy capture-aware heuristic, neural 128 simulations | 39 | 1 | 0 | [data](pilot_mlp128-vs-greedy.json) |
| Previous 1,065-game checkpoint, 128 simulations each | 63 | 12 | 5 | [data](duel-flat1721-vs-first.json) |
| MCTS, 128-simulation ceiling, both sides 50 ms CPU | 37 | 3 | 0 | [data](equalcpu50-flat1721-vs-mcts128.json) |
| MCTS, 10,000-simulation ceiling, both sides 50 ms CPU | 36 | 4 | 0 | [data](equalcpu50-flat1721-vs-mcts10000.json) |
| Iterative minimax, depth-4 ceiling, both sides 50 ms CPU | 38 | 2 | 0 | [data](equalcpu50-flat1721-vs-minimax4.json) |

All matches used colour-swapped pairs from seeded four-insertion random openings,
settling any resulting captures. CPU comparisons used one neural inference
thread and one game per search batch. These are elapsed-time caps checked between
search operations, with occasional overruns; full timing measurements are in the
reports. Minimax returns its last completed depth or its bounded fallback, so the
depth ceiling is not a claim that every move completed depth four. MCTS ceilings
likewise are not a claim that every move used the maximum simulations.

The 50 ms matches use separate held-out opening seeds. Their Wilson intervals are
approximately 80–97%, 77–96%, and 84–99%, respectively. Per-colour and paired-opening
breakdowns are included in the JSON. Conventional binomial intervals are useful
summaries, but paired openings can correlate outcomes. No human rating or expert
GIPF playing strength has been established.

The first intermediate checkpoint also completed larger random and greedy tests;
those are retained separately and must not be attributed to this newer model.
Architecture/search pilots and longer training are still underway. Check
[champion metadata](champion.json) for the opponent currently served.
