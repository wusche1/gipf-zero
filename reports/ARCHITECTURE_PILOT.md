# Superseded four-model comparison

This first controlled comparison was stopped at the user’s request after all eight ten-minute training runs and eight head-to-head matches. It is incomplete and does not rank all four architectures. The subsequent five-model overnight comparison supersedes it.

| Candidate | Opponent | Mode | Wins–losses | Cutoffs | Unfinished |
|---|---|---|---:|---:|---:|
| mlp256-seed320101-0336a24105b8.pt | resnet32-seed320101-5374e8e4b522.pt | equal_simulations | 3–5 | 0 | 0 |
| mlp256-seed320101-0336a24105b8.pt | resnet32-seed320101-5374e8e4b522.pt | equal_cpu_time | 5–3 | 0 | 0 |
| mlp256-seed320102-4e753a9bede6.pt | resnet32-seed320102-9b9f4cbc4f77.pt | equal_simulations | 2–4 | 2 | 0 |
| mlp256-seed320102-4e753a9bede6.pt | resnet32-seed320102-9b9f4cbc4f77.pt | equal_cpu_time | 4–4 | 0 | 0 |
| mlp256-seed320101-0336a24105b8.pt | hexresnet32-seed320101-a7f4960de480.pt | equal_simulations | 2–6 | 0 | 0 |
| mlp256-seed320101-0336a24105b8.pt | hexresnet32-seed320101-a7f4960de480.pt | equal_cpu_time | 2–5 | 1 | 0 |
| mlp256-seed320102-4e753a9bede6.pt | hexresnet32-seed320102-7ab72c9bd50c.pt | equal_simulations | 1–6 | 1 | 0 |
| mlp256-seed320102-4e753a9bede6.pt | hexresnet32-seed320102-7ab72c9bd50c.pt | equal_cpu_time | 1–7 | 0 | 0 |

Each architecture had two independent random initializations and the same ten-minute native-forest self-play allocation, 32 simulations per decision, batch size 256, and 32 optimizer updates per 32 completed games. Wall-clock allocations include small checkpoint/shutdown overhead differences, recorded in metrics. These are equal dedicated hardware-time trials, not equal FLOP counts.

[Raw orchestration summary](architecture-comparison/20260909T222247Z/summary.json). No conclusions about the unplayed pairs can be drawn.
