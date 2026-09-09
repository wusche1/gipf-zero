# Bounded pilot search

Each training trial had a 900-second wall-clock bound, checkpointed throughout,
and was followed by paired-opening evaluation. Models started from random
weights except the explicitly marked continuation trial. All used AdamW at
learning rate 0.001, batch size 256, a 100,000-position replay buffer, twelve
board symmetries, self-play outcomes for value targets, and MCTS visit policies
for policy targets. The training seed was 101; the continuation restored the
saved RNG states.

| Trial | Self-play simulations / parallel games | Completed games | Training cutoffs | Random W–L | Greedy W–L |
|---|---|---:|---:|---|---|
| Flat MLP, width 128, blocks 2 | 64 / 64 | 1,721 | 0 | 40–0 | 39–1 |
| Flat MLP, width 256, blocks 2 | 64 / 64 | 2,513 | 0 | 39–1 | 37–3 |
| Factorized MLP, width 128, blocks 2 | 64 / 64 | 1,947 | 0 | 40–0 | 28–12 |
| Flat residual network, width 32, two residual blocks | 32 / 64 | 4,091 | 4 | 40–0 | 36–4 |
| Deeper-search continuation of flat MLP-128 | 128 / 128 | 2,701 total; 980 additional | 0 | 40–0 | 40–0 |

All baseline matches in this table used 128 neural search simulations and 40
games; none had evaluation cutoffs. The fixed greedy opponent received a 50 ms
per-decision cap. These are practical candidate trials, not controlled causal
architecture ablations: some jobs overlapped, native feature encoding was
optimized during the search, and search depth and throughput differed. The
continuation has twice the total training wall time of the original MLP-128
when its source run is included. The larger sample count of the residual trial
partly reflects its smaller search budget.

Direct colour-paired checkpoint comparisons distinguish strengths that a
saturated random-player test misses:

- MLP-256 versus MLP-128: **119 wins, 67 losses, 14 cutoffs** in the larger
  [200-game comparison](duel-mlp256-vs-flat1721-200.json), following an initial
  80-game screen. This justified promoting MLP-256.
- Deeper MLP-128 continuation versus MLP-256: **33 wins, 42 losses, 5 cutoffs**
  in [80 games](duel-deeper-vs-mlp256.json). Its perfect small baseline scores
  did not justify replacing the incumbent.

The flat policy outputs 2,730 action logits. The factorized alternative shares
capture decisions through row and individual-double scores, but was weaker in
this trial. MLP `blocks=2` means an input hidden layer followed by two further
hidden layers; residual `blocks=2` means two two-convolution residual blocks.
All legal actions are masked by the same authoritative rules engine.

Exact configurations, training measurements, and loss curves are exported under
[training/](training/). Losses measure fit to changing self-play targets and must
not be interpreted as a direct playing-strength ranking.

The residual network scored **42–32 with 6 cutoffs** against MLP-256 at equal
128-simulation search counts ([80-game report](duel-resnet32-vs-mlp256.json)).
At 50 ms per decision on one CPU thread it scored **26–13 with 1 cutoff**
([40-game report](duel-equalcpu50-resnet32-vs-mlp256.json)). This made it the
leading long-run candidate, but the sample was too small to discard MLP-256.

For the remaining training window, both finalists continue from their saved
weights, optimizer state, RNG states, and replay. The residual network uses
128 search simulations and MLP-256 uses 64; each processes 128 parallel games
and performs 64 updates per 64 completed games. Both keep learning rate 0.001.
They are scheduled to stop at 21:15 UTC, reserving 45 minutes for final evaluation
and publication before midnight in Germany. Running both makes use of otherwise
idle GPU capacity; the final selection is based on matches, not loss curves.
The previously proven MLP champion remains available until a replacement passes
the promotion gate.
