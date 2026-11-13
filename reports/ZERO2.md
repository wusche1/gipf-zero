# GIPF Zero 2: a transformer opponent trained by self-play, running in the browser

48-hour run, 11–13 November 2026. GPU spend: FINAL_SPEND of a $40 budget (Vast.ai, RTX 4080-class boxes).

## What is live

The site plays a 4-layer, width-64 transformer (217k parameters, 0.9 MB of weights) with Monte Carlo tree search.
Search and network inference run in one WebAssembly module, using a hand-written SIMD kernel with 16-bit quantised weights.
There is no server and no ONNX Runtime. The AI download is about 1 MB; at the Normal level a move takes roughly 0.7 s and searches a few thousand positions per second on a laptop-class CPU.

## Headline results

| Comparison | Result | Conditions |
|---|---|---|
| vs GF1 1.03 (winner of the 2001 Computer Olympiad GIPF event), search depths 3–6 | 181 wins in 182 valid games | first deployed net (96x6, 21M positions), real browser engine, 300 ms and 1000 ms per move |
| vs GF1 with our side handicapped | GF1 depth 5 ≈ our net at 8–10 simulations; depth 6 ≈ 12–16 simulations | 64x4 net, fixed simulations, 30 games per cell (12 at depth 6); see table below |
| vs BoardSpace robots (Bestbot = depth 7, Smartbot, Dumbot, Weakbot) | 246–0 | 64x4 net (second deployed weights), 300 ms, 1000 ms and 400 simulations per move |
| final training net (128x8) vs previous champion (square CNN) | about +1700 Elo | equal search, 100 simulations each, chained checkpoint ladder |
| deployed net vs previous deployments | FINAL_VS_PREV | real browser engine, 300 ms per move each |

External matches compare the full board between both rule engines after every ply. No divergence occurred in any counted game.

GF1 handicap matches (our wins–losses; our side limited to a fixed number of simulations per move, GF1 at fixed depth):

| Our simulations | vs GF1 depth 5 | vs GF1 depth 6 |
|---|---|---|
| 64 | 26–4 | 10–2 |
| 16 | 21–9 | 7–5 |
| 4 | 6–24 | – |
| 1 (raw policy) | 3–27 | – |

The site searches several hundred to a few thousand simulations per move, so the gap to GF1 is large; GF1 at depth 6 needs about 8 s per move on this machine.

## How it was built

1. **Pipeline** (`training/selfplay.py`, `training/learn.py`). Self-play actors run a C++ tree search with batched GPU inference and write position shards. One learner trains on a sliding window and publishes weights; other networks can train on the same stream as "followers". Two boxes produced about 10M positions per hour; 375M positions in total.
2. **Playout-cap randomisation.** Every move gets a 32-simulation search; a random quarter get 200 simulations with root noise, and only those train the policy.
3. **Networks** (`training/net.py`). A hex CNN (masked 3x3 kernels) and a transformer over the 37 cells with a learned attention bias per relative hex offset. Both use a per-cell readout: 6 push-direction scores, 3 line-capture scores and 1 "remove this double piece" score per cell, combined into the 2,730 move scores. The value head reads all cells.
4. **Teacher and students.** The largest net that still improves generates the self-play games (96x6, later 128x8). Smaller nets learn from the same games and are the deployment candidates.
5. **Browser kernel** (`engine/former_infer.*`). The transformer forward pass is written in C++ with WebAssembly SIMD and compiled into the search engine: 2.2x faster than ONNX Runtime Web for the same net, and a further 1.2–1.4x with int16 weights. The whole search loop runs inside WASM.

## Experiments and what they showed

Matches use paired openings and colours. "Equal search" means the same number of simulations per move on the GPU arena; "equal time" means simulations scaled by measured browser cost, or real milliseconds in the WASM engine. Raw results are in `reports/zero2/ladders/` (file names in the last column).

| Question | Result | Evidence |
|---|---|---|
| CNN or transformer? | Transformer 96x6 beat hex CNN 96x10 47–13 and hex CNN 64x6 48–12 at equal search after the same training time | `L1` |
| Sample reuse 8 vs 4 | Helps the CNN (38–22); little for the transformer (33–27) | `L1`, `L3a` |
| Search settings | First-play-urgency reduction 0.1–0.2 with cpuct 1.0 beat the original search 83–15 at 300 simulations; the best cpuct rises to 1.5–2.5 at 1000–2000 simulations | `T1`–`T5` |
| Self-play search depth | 200/32 beat 600/100 simulations 114–86 early in training; no difference at the plateau (101–99) | `AB`, `AB2`, `AB2b` |
| Blended value target (outcome + search value) | Worse than outcome only (43–57) | `M2` |
| Learning-rate drop | No lasting gain while the net was still improving (99–99 after 3 h); +100 Elo once progress had flattened (79–41); a second drop gave nothing (104–96) | `M3`, `M4`, `M5` |
| Replay window 16M vs 6M | No measurable difference | `M4` |
| Does a bigger net break the plateau? | Yes. Trained from scratch on the stream, 128x8 beat a same-age 96x6 79–41 after 2 h and the 30-hour-old 96x6 97–62 after its own learning-rate drop; 160x8 was no better than 128x8 in the time available | `CAP1`–`CAP4` |
| Does the stronger teacher help the students? | Yes. The 96x6 rose again once it learned from 128x8 games (flat from 130M to 250M positions, then about +35 Elo) | `PROG3`, `TP1`, `TP2` |
| Net size vs search depth at equal browser time | Early in training small nets won clearly (64x4 beat 96x6 27–12 at 1000 ms). After 360M positions 64x4, 64x6 and 96x6 are level at 300 ms and 1000 ms; 128x8 loses to 96x6 at 1000 ms (8–22) | `L3b`, `L3c`, `L4`, WASM arena |
| int16 kernel | No loss at equal simulations (97.5 of 202 points); 64–26 at equal time | WASM arena |

The deployed size is the smallest of the three level candidates: same strength at equal time, the smallest download, and the least sensitive to slow devices.

## Progress over training

Equal search (100 simulations), Elo relative to the previous champion, fitted over two chained checkpoint ladders (`PROG`, `PROG3`; `reports/zero2/progress-elo.json`). Chained ladders accumulate error, so treat differences of a few tens of Elo as noise.

| Checkpoint | Elo |
|---|---|
| previous champion (square CNN, 44k games) | 0 |
| hex CNN 64x6, 5M positions | +28 |
| hex CNN 64x6, 14M positions | +654 |
| transformer 96x6, 16M | +1053 |
| transformer 96x6, 30M | +1380 |
| transformer 96x6, 70M (before first learning-rate drop) | +1468 |
| transformer 96x6, 130M | +1585 |
| transformer 96x6, 250M (plateau) | +1592 |
| transformer 96x6, 345M (learning from 128x8 games) | +1624 |
| transformer 128x8 teacher, 15 h old | +1706 |

## Incidents

- **NaN divergence** at 5.9M positions: every learner diverged within minutes of each other. Not reproduced offline. Training now uses bf16 autocast, skips steps with non-finite gradients, never publishes non-finite weights and rolls back automatically; no recurrence in 370M positions.
- **Host failures:** one rented box went offline permanently (two hours of follower training lost); the provider rewrote an SSH key file once (25 minutes of paused transfer, no data lost).
- **Self-play diversity collapse**, hours 36–42. After the switch to the sharper 128x8 teacher, games converged onto the same lines: distinct positions at move 16 per data file fell from 97% to 62%. It showed up as a fast drop in policy loss with rising value loss. Self-play now starts 75% of games with 1–4 random moves, samples moves in proportion to visits until move 30 and less greedily afterwards; diversity returned to above 90% within half an hour. The teacher's progress had slowed during the collapse (84–76 against its snapshot from 4.5 h earlier, after 93–67 over the previous 3 h).

## Known limits

- No games against humans.
- GF1 was tested to depth 6 only, with 8–12 games per cell at depth 6; its timed mode was not used.
- The net-size comparison at equal time rests on 30–42 games per pairing; differences under about 60 Elo are not resolved.
- The headline GF1 and BoardSpace sweeps used earlier weights of the deployed family; FINAL_EXTERNAL
- A rules edge case: one capture that removes both sides' last GIPF piece. The official rules name no winner but say a player must keep one GIPF piece on the board; this engine lets the capturer do it and win, GF1 awards Black, BoardSpace awards the other player. It occurred once in about 68,000 random games and in no counted match game.
- Elo figures chain several pairings and carry correspondingly wide uncertainty.

## Reproduce

```
python -m training.learn --data DATA --out RUN/net --kind former --width 128 --blocks 8 --reuse 6
python -m training.selfplay --weights RUN/net/weights.pt --out DATA --cpuct 1.0 --fpu 0.1 --open-random 0.75 --temp-plies 30 --late-power 2
python -m training.learn --data DATA --out RUN/student --kind former --width 64 --blocks 4 --reuse 8 --sync-budget 1
python -m training.ladder --players a=A.pt:100:1.0:0.1 b=B.pt:100:1.0:0.1 --games 100 --out results.jsonl
python -m ops.web_model latest --checkpoint X.pt --title "GIPF Zero Transformer" --cpuct 1.5 --fpu 0.1 --int16
node benchmarks/wasm_arena.mjs --a bin=A.bin,arch=A.json,int16=1,cpuct=1.5,fpu=0.1,ms=300 --b ... --games 20
node benchmarks/gf1_match.mjs --depth 5 --ms 300 --games 20
node benchmarks/boardspace_match.mjs --level bestbot --ms 300 --games 20
```
