# Original deadline results

This document records the completed, time-bounded selection run from 9 September
2026. It is intentionally separate from later experiments. The frozen selected
checkpoint is the square 3x3 **ResNet-32 CNN**, trained for 43,869 self-play
games: [`final-aa533bb89222.pt`](https://github.com/wusche1/gipf-zero/releases/download/v1.0.0/final-aa533bb89222.pt)
(SHA-256 `aa533bb8922231a3e0dc53dd2ff947cab6899ef09a9245463c89ac99ea776ef0`).
The [orchestration summary](final/20260909T191906Z/summary.json) completed with
no recorded errors.

## Selection

The CNN won its predeclared promotion duel against the then-champion, 51--29 in
80 decisive games. The Wilson 95% lower bound was 0.528, above the required
0.5 gate. It then beat greedy 40--0 in the required gate.

| Candidate and report | Result | Conditions |
| --- | ---: | --- |
| [ResNet-32 CNN vs prior champion](final/20260909T191906Z/final-vs-champion-duel.json) | 51--29, 0 cutoffs | 80 paired-opening games; CPU, one thread, batch 1; 50 ms per decision; 10,000-simulation ceiling; max 300 insertion plies |
| [ResNet-32 CNN vs greedy](final/20260909T191906Z/final-vs-greedy.json) | 40--0, 0 cutoffs | GPU, batch 64, fixed 128 neural simulations; four-ply seeded openings; baseline move-limit setting 50 ms; max 300 insertion plies |
| [MLP-256 (103,668 games) vs CNN](final/20260909T191906Z/final_mlp-vs-champion-duel.json) | 40--40, 0 cutoffs | Same CPU 50 ms, batch-1 protocol as the promotion duel |

The later MLP-256 result had a Wilson 95% interval of 0.393--0.607 and did not
meet the predeclared promotion gate. The CNN therefore remained the selected
source.

## Held-out evaluation

All evaluations below use the frozen CNN source above, colour-balanced paired
openings, seeded opening diversity, and a 300-insertion-ply game limit. There
were no cutoffs or unfinished games in any completed report.

| Opponent | Result | Evaluation conditions |
| --- | ---: | --- |
| [Random](final/20260909T191906Z/heldout-vs-random.json) | 200--0 | GPU, batch 64, fixed 128 neural simulations; baseline move-limit setting 50 ms |
| [Greedy](final/20260909T191906Z/heldout-vs-greedy.json) | 200--0 | GPU, batch 64, fixed 128 neural simulations; baseline move-limit setting 50 ms |
| [MCTS](final/20260909T191906Z/heldout-vs-mcts.json) | 79--1 | CPU, one thread, batch 1; 50 ms per decision for both sides; each search has a 10,000-simulation ceiling |
| [Minimax depth 6](final/20260909T191906Z/heldout-vs-minimax.json) | 80--0 | CPU, one thread, batch 1; 50 ms per decision for both sides; neural ceiling 10,000 simulations |

The decisive Wilson 95% intervals are 0.981--1.000 for each 200--0 result,
0.933--0.998 against MCTS, and 0.954--1.000 against minimax. The equal-CPU-time
reports record actual search counts and elapsed search times; the
10,000-simulation value is a ceiling rather than work guaranteed on every move.

## Interpretation and follow-up

These are the original deadline results and the basis for the selected
checkpoint. Earlier pilot comparisons used differing backends, budgets, or
training allocations, so they are confounded and are not architecture evidence.

The supervised fresh-start architecture comparison is a follow-up that is still
pending. It has no promotion path and does not revise this result. Its frozen
configuration is [`ops/architecture_comparison_config.json`](../ops/architecture_comparison_config.json);
when complete, its raw reports will be under `reports/architecture-comparison/`.
