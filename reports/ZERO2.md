# GIPF Zero 2: transformer opponent trained by self-play (interim report)

Status: interim, written about 20 hours into a 48-hour run. Numbers will be updated at the end.

## What is live

The site plays a 4-layer, width-64 transformer (about 217k parameters, 0.9 MB of weights) with Monte Carlo tree search.
Search and network inference both run in one WebAssembly module with a hand-written SIMD kernel using 16-bit quantised weights.
There is no server and no ONNX Runtime; the AI download is under 1 MB and a move at the Normal level takes about 0.7 s.

## Headline results

| Comparison | Result | Conditions |
|---|---|---|
| vs GF1 1.03 (2001 Computer Olympiad winner), depths 3–6 | 181 wins in 182 valid games | earlier 96x6 net, real browser engine, 300 ms and 1000 ms per move |
| vs BoardSpace robots (Bestbot = depth 7, Smartbot, Dumbot, Weakbot) | 246–0 | deployed 64x4 net, 300 ms, 1000 ms and 400 simulations per move |
| 70M-position transformer vs previous champion (square CNN) | about +1500 Elo | equal search, 100 simulations each, checkpoint ladder |
| deployed 64x4 vs first deployed transformer (96x6) | 33–9 | real browser engine, 300 ms per move each |

All external matches cross-check the full board between both rule engines after every ply; no rule divergence occurred in any counted game.

## How it was built

1. **Pipeline.** Self-play actors (C++ tree search, batched GPU inference) write position shards; one learner trains on a sliding window and publishes weights. Other networks can train on the same stream as "followers". About 10M positions per hour on two rented RTX 4080-class boxes.
2. **Playout-cap randomisation.** Every move gets a 32-simulation search; a random quarter get 200 simulations and only those train the policy.
3. **Networks.** A hex CNN (masked 3x3 kernels) and a transformer over the 37 cells with a learned attention bias per relative hex offset. Both use a per-cell readout: 6 push directions, 3 line-capture scores and 1 "remove this double piece" score per cell, combined into the 2,730 move scores. The value head reads all cells.
4. **Browser kernel.** The transformer forward pass was written in C++ with WebAssembly SIMD and compiled into the search engine: 2.2x faster than ONNX Runtime Web for the same net, and another 1.3–1.4x with int16 weights.

## Experiments and what they showed

Unless noted, matches are 60–200 games per pairing with paired openings and colours; Elo figures come from a logistic fit of the pairings listed in `reports/zero2/ladders/`.

| Question | Result | Evidence |
|---|---|---|
| CNN or transformer? | Transformer 96x6 beat hex CNN 96x10 47–13 and hex CNN 64x6 48–12 at equal simulations after the same training time | `L1` |
| Sample reuse 8 vs 4 | Helps the CNN (38–22), little for the transformer (33–27) | `L1`, `L3a` |
| 600/100 vs 200/32 simulations in self-play | 200/32 won 114–86 after 2.5 h each | `AB` |
| First-play-urgency reduction and cpuct | FPU 0.1–0.2 with cpuct 1.0 beat the original search 83–15 at 300 simulations; best cpuct rises to 1.5–2.5 at 1000–2000 simulations | `T1`–`T5` |
| Net size vs search depth at equal browser time | Small nets win: 64x4 (2,860 sims) beat 96x6 (1,000 sims) 27–12 at a 1000 ms budget; 128x6 was last. 48x4, 64x4 and 64x6 are level at 300 ms | `L3b`, `L3c`, `L4` |
| Blended value target (outcome + search value) | Worse than outcome only (43–57) | `M2` |
| Learning-rate drop | No lasting gain while the net was still improving (99–99 after 3 h); a clear gain once progress had flattened (79–41 after 2 h) | `M3`, `M4` |
| Replay window 16M vs 6M positions | No measurable difference | `M4` |
| int16 kernel | No loss at equal simulations (97.5 of 202 points); 64–26 at equal time | WASM arena |

## Progress over training

Equal search (100 simulations), Elo relative to the ladder mean; `reports/zero2/ladders/PROG.jsonl`.

| Checkpoint | Elo |
|---|---|
| previous champion (square CNN, 44k games) | −749 |
| hex CNN, 5M positions | −718 |
| hex CNN, 14M positions | −88 |
| transformer, 16M | +315 |
| transformer, 30M | +646 |
| transformer, 50M | +731 |
| transformer, 70M | +756 |

Gains flattened after about 40M positions; the learning rate was lowered at 76M.

## Incidents

- At about 5.9M positions every learner diverged to NaN within minutes of each other. The cause was not reproduced offline. Training now uses bf16 autocast, skips steps with non-finite gradients, never publishes non-finite weights and rolls back automatically. It has not recurred in 180M positions.
- The host rewrote an SSH key file once, pausing data transfer between boxes for about 25 minutes; no data was lost.

## Known limits

- No games against humans.
- GF1 was tested to depth 6 only; depth 6 has 8 games per budget.
- One rules edge case: a capture that removes both sides' last GIPF piece. The official rules do not name a winner but say a player must keep one GIPF piece on the board; this engine lets the capturer do it and win. It occurred once in about 68,000 random games.
- Elo figures chain several pairings and carry correspondingly wide uncertainty.

## Reproduce

```
python -m training.learn --data DATA --out RUN/net --kind former --width 96 --blocks 6 --reuse 6
python -m training.selfplay --weights RUN/net/weights.pt --out DATA --cpuct 1.0 --fpu 0.1
python -m training.ladder --players a=A.pt:100:1.0:0.1 b=B.pt:100:1.0:0.1 --games 100 --out results.jsonl
python -m ops.web_model latest --checkpoint X.pt --title "GIPF Zero Transformer" --cpuct 1.5 --fpu 0.1 --int16
node benchmarks/wasm_arena.mjs --a bin=A.bin,arch=A.json,int16=1,cpuct=1.5,fpu=0.1,ms=300 --b ... --games 20
node benchmarks/gf1_match.mjs --depth 5 --ms 300 --games 20
node benchmarks/boardspace_match.mjs --level bestbot --ms 300 --games 20
```
