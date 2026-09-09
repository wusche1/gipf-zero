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

## Search implementation improvement during the long run

A native PUCT argmax replaced repeated NumPy allocations while preserving the
score arithmetic and first-maximum tie rule. It matched 2,003 seeded/random/tie
cases and complete reference search policies/visit counts. A reproducible
[trained-residual-network benchmark](native-puct-benchmark.json), with both
training jobs active, measured median search time **0.875 s → 0.480 s** for
128 fresh roots and 128 simulations (**1.82× throughput**). This is a search
microbenchmark, not a claim that complete training accelerates by the same factor.
The benchmark script is `python -m ops.benchmark_search CHECKPOINT --output FILE`.

Both trainers checkpointed and restarted at approximately 16:48 UTC to adopt the
optimization; model weights, optimizer, RNG and replay were restored. In-flight
unlabelled games are not checkpointed and restart from the opening, so recovery
preserves learning state but not an identical uninterrupted self-play trajectory.
The CPU inference service also adopted the faster selector. No game rules changed.

An additional opt-in static CUDA-graph backend was tested. It matches eager
inference on the same padded batch; changing batch shape can introduce small
floating-point differences, so it is not generally bit-identical to unpadded
eager inference. Tests verify fresh optimizer weights and recapture after
parameter-storage replacement. A normal-game 45-second smoke completed 1,801
self-play games and repeated updates without new cutoffs.

The minimal prototype suggested a 29% gain, but the full implementation,
including safety checks, measured only **1.14×** after optimization
([benchmark](cuda-graph-benchmark.json)). That was below the chosen 20% keep
threshold. The residual training job briefly used it around 16:59–17:04 UTC,
then checkpointed and returned to ordinary inference. It remains available
behind `--cuda-graph-batch 128`, default off. The verified native PUCT selector
remains enabled. Reproduce the graph comparison with
`python -m ops.benchmark_cuda_graph CHECKPOINT --output FILE`.

## First long-run replacement (17:28 UTC)

The 10,084-game MLP256 checkpoint beat the 6,300-game residual champion
**61–18 with one cutoff** in 80 paired-opening games with each side receiving
50 ms of CPU search per decision ([raw report](alternative-1716-equalcpu.json)).
Mean measured search times were 50.02 and 50.30 ms; the faster MLP searched
422 versus 153 simulations on average. At equal 128 simulations its margin
was smaller: 45–32 with three cutoffs. It also won 39/40 against greedy.
The equal-time result passed the conservative promotion gate, and this exact
frozen MLP checkpoint became the served champion. This is a selection match,
not a held-out final test or a human skill rating.

Subsequent league matches use equal CPU time across different architectures
and equal simulations within the same architecture. Both training families
continue; training loss alone never triggers promotion.

The promoted 10,084-game checkpoint subsequently won **40–0** against iterative
minimax (maximum depth 6, capture quiescence), with both players capped at
250 ms per decision and neural search capped at 1,000 simulations, matching
the casual serving limit. Twenty paired openings used both colours; there
were no cutoffs ([report](mlp10084-minimax-cpu250.json)). Neural search averaged
206.6 ms and 996 simulations. Maximum depth is a search ceiling, not a claim
that minimax reached six plies within every deadline.

## Geometry-aware pilots requested during the run

Two additional randomly initialized candidates use 15-minute pilots with 64
parallel games, 32 simulations, 32 updates per 32 completed games, batch 256,
learning rate .001, and the existing twelve board symmetries.

* **HexResNet32, two blocks:** masks the top-left and bottom-right corners of
  every 3×3 convolution on the axial grid. The first pilot completed 5,666 games
  (eight training cutoffs) and scored 40–0 against random and 37–3 against greedy
  in 40-game fixed-128-simulation tests.
* **Transformer64, two blocks:** noncausal attention over 37 playable-cell
  tokens, four heads, feed-forward width 128, learned linear embeddings of
  centred axial coordinates, flattened policy/value heads. Pilot completed 5,411 games (one training cutoff), with 40–0 against both
  random and greedy in fixed-128-simulation tests.

These are practical time-limited comparisons. Different CPU contention and
search optimizations mean games/hour cannot isolate the effect of geometry.
Neither architecture has yet displaced the validated MLP champion.

## Second measured search improvement (18:59 UTC)

Profiling found repeated legal-action extraction, NumPy reductions and small
array allocations in node expansion. A combined native policy expansion
helper preserves action order and double-precision softmax. Seeded comparison
tests matched full search policies and visit counts, including noisy roots.
A three-repeat complete-search benchmark measured **1.32× residual / 1.43× MLP**
speedups ([report](native-expand-benchmark.json)). GPU-to-CPU synchronization
time in the profile includes waiting for GPU computation; it is not purely
data-transfer overhead.

Batch scaling also showed residual search throughput improving from about
32,000 root-simulations/s at 128 roots to 40,600 at 512 roots; the MLP showed
little gain above 128 ([measurements](search-batch-scaling.json)). Both trainers
checkpointed and restarted to adopt native expansion. Residual self-play now
uses 512 concurrent games; MLP remains at 128. Training targets and update
ratio per completed game are unchanged. The batch and native-expansion gains
were measured separately and should not simply be multiplied.

At equal 50 ms CPU search, the 5,666-game hex pilot lost **6–74** to the
23,016-game MLP ([match](hex-pilot-vs-mlp23016.json)). This compares practical
checkpoints with different amounts of experience, not architecture alone.

At 19:06 UTC, the 16,694-game square residual checkpoint passed the promotion
gate against the 23,016-game MLP: **56–19 with five cutoffs**, then 40–0 against
greedy. Both sides had 50 ms CPU budgets in that head-to-head match.
The 41,008-game MLP also beat its older version **60–18 with two cutoffs** at
equal simulations; a new equal-CPU-time match against the residual champion
is pending.

The transformer pilot lost **2–77 with one cutoff** against the 23,016-game
MLP at 50 ms CPU search ([match](transformer-pilot-vs-champion.json)). It learned
to beat the simple baselines, but did not approach the incumbent under this
short training and CPU-serving budget. This does not establish that transformers
are intrinsically worse: architecture size, inference speed and experience all
differ. Both new pilot checkpoints are preserved for future experiments.

The 41,008-game MLP then beat the 16,694-game residual champion **60–20, no
cutoffs**, at 50 ms CPU search and passed its greedy gate. It became the served
champion. No architecture receives promotion based solely on training loss.
