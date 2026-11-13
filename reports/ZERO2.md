# GIPF Zero 2: a transformer opponent trained by self-play, running in the browser

48-hour run, 11–13 November 2026.

## Summary

- The site now plays a small transformer with tree search, entirely in the browser. Play it at https://wusche1.github.io/gipf-zero/ or locally with `python -m http.server 8000 --directory web`. Weights: `web/models/latest.bin` (0.9 MB) with settings in `web/models/latest.json`.
- It beat both other GIPF programs I could find and run: GF1 (winner of the 2001 Computer Olympiad GIPF event) and BoardSpace's robots. Details and which weights were used are in the next section.
- The largest training net is roughly 1,700 Elo above the previous site model at equal search (chained estimate, wide interval).
- GPU spend was about $31.5 of the $40 budget (hourly rate times uptime; Vast.ai, RTX 4080-class boxes).

Notation: "64x4" means a transformer of width 64 with 4 layers. Three sizes matter: 64x4 (deployed), 96x6 (generated the self-play games until hour 29) and 128x8 (generated them afterwards; not deployed).

## What is live

| Level | Time budget per move | Typical simulations on a laptop CPU |
|---|---|---|
| Fast | 0.3 s | several hundred |
| Normal (default) | 1 s | about 2,000 |
| Strong | 3 s | several thousand |

The search stops early when the best move cannot be overtaken, so moves often arrive faster than the budget (about 0.2 s and 0.65 s on average at Fast and Normal in the match logs).
Search and network inference run in one WebAssembly module with a hand-written SIMD kernel and 16-bit quantised weights; there is no server and no ONNX Runtime.
The AI also thinks during your turn, shows a win-chance bar, can show its candidate moves, and varies its first six plies so games differ.
The old "Classic" opponent was removed.

## Results against other programs

Every external match compares the full board between both rule engines after each ply; games where the engines disagree are discarded (none occurred in counted games).

| Opponent | Our net and budget | Result (ours first) |
|---|---|---|
| GF1 1.03, depths 3–6 | first deployed net (96x6 after 21M positions), 300 ms and 1000 ms per move | 190 wins, 1 loss in 191 valid games; 1 further game aborted on a bridge bug, since fixed |
| GF1 depth 4 | same net, 1000 simulations | 20–0 |
| BoardSpace Bestbot (depth 7), Smartbot, Dumbot, Weakbot | 64x4, second deployed weights; 300 ms, 1000 ms, 400 simulations | 246–0 |
| GF1 depth 6 | final live weights, 300 ms | FINAL_GF1_D6 |
| BoardSpace Bestbot | final live weights, 300 ms | FINAL_BESTBOT |
| GF1 depth 5 / Bestbot | final live weights at 100 ms (proxy for a device 3–10x slower) | FINAL_SLOW |

Handicap matches place GF1 on our scale. Our side is limited to a fixed number of simulations per move; GF1 plays at fixed depth.

| Our simulations per move | vs GF1 depth 5 | vs GF1 depth 6 |
|---|---|---|
| 64 | 26–4 | 10–2 |
| 16 | 21–9 | 7–5 |
| 12 | FINAL_S12 | – |
| 8 | FINAL_S8 | – |
| 6 | FINAL_S6 | – |
| 4 | 6–24 | – |
| 1 (raw policy, no search) | 3–27 | – |

The rows for 64, 16, 4 and 1 used the 64x4 snapshot from hour 40; the rows for 12, 8 and 6 used the final live weights.
GF1 at depth 5 breaks even with our net somewhere between 4 and 16 simulations (HANDICAP_READ); depth 6 is not resolved by 12 games per cell.
The site searches hundreds to thousands of simulations per move. GF1 at depth 6 needs about 8 s per move on this machine; deeper settings and GF1's timed mode were not tested.

## How it was built

1. **Pipeline** (`training/selfplay.py`, `training/learn.py`). Self-play "actors" run a C++ tree search with batched GPU inference and write files of positions. One learner trains on a sliding window of the newest positions and publishes weights to the actors. Other networks can train on the same stream without playing ("followers"). Two boxes produced about 10M positions per hour; 390M in total.
2. **Cheap and full searches.** Every self-play move gets a 32-simulation search; a random quarter get 200 simulations with root noise, and only those train the policy ("200/32" below).
3. **Networks** (`training/net.py`). A hex CNN (masked 3x3 kernels) and a transformer over the 37 cells with a learned attention bias per relative hex offset. Both use a per-cell readout: 6 push-direction scores, 3 line-capture scores and 1 "remove this double piece" score per cell, combined into the 2,730 move scores. The value head reads all cells.
4. **Teacher and students.** The largest net that still improves generates the games (96x6, later 128x8). Smaller followers learn from those games and are the candidates for the site.
5. **Browser kernel** (`engine/former_infer.*`). The transformer forward pass is written in C++ with WebAssembly SIMD and compiled into the search engine: 2.2x faster than ONNX Runtime Web for the same net, and a further 1.2–1.4x with int16 weights.

## Experiments

Matches use paired openings and colours. "Equal search" is the same number of simulations per move in the GPU arena (`training/arena.py`). "Equal time" is either simulations scaled by measured browser cost (GPU arena) or real milliseconds in the WASM engine (`benchmarks/wasm_arena.mjs`).
Scores are written "A beat B x–y". With 60–200 games per pairing, a 60–40 score is about the smallest that is significant at the 5% level; several rows below are only suggestive and say so.
Raw results: `reports/zero2/ladders/<name>.jsonl` (one line per pairing, wins and losses from player `a`'s view), `reports/zero2/arena/` (WASM arena) and `reports/zero2/external/`.

| Question | Result | Strength of evidence | Files |
|---|---|---|---|
| CNN or transformer? | Transformer 96x6 beat hex CNN 96x10 47–13 and hex CNN 64x6 48–12 at equal search after the same training time | Clear (p < 0.0001), one training run each | `L1` |
| Training each sample 8 times instead of 4 ("reuse") | CNN: reuse 8 beat reuse 4 38–22. Transformer: 33–27 at 100 simulations and 41–19 at 300 | CNN borderline (p = 0.05); transformer significant when pooled (74–46) | `L1`, `L3a`, `L3b` |
| Search settings | First-play-urgency reduction 0.2 with cpuct 1.0 beat the original search 83–15 at 300 simulations. At 1000 simulations cpuct 1.5 beat 1.0 61–39; at 2000 it led 58–42 | First result clear; cpuct trend significant at 1000 only | `T1`–`T5` |
| Self-play search depth | 200/32 led 600/100 114–86 after 2.5 h early in training; 101–99 at the plateau | Suggestive early (p = 0.06); no difference later within about ±50 Elo | `AB`, `AB2`, `AB2b` |
| Blended value target (outcome mixed with search value) | Lost to outcome-only 43–57 | Not significant (p = 0.19); dropped for lack of benefit | `M2` |
| Learning-rate drop | No lasting gain while the net was still improving (99–99 after 3 h). At the plateau, the dropped run beat a control with the old rate 69–51; the main run (which also changed its replay window) beat the control 79–41 | Suggestive for the clean pair (p = 0.12), clear for the confounded one. A second drop gave 104–96 | `M3`, `M4`, `M5` |
| Replay window 16M vs 6M positions | 54–66 for the larger window | No difference within about ±100 Elo | `M4` |
| Does a bigger net break the plateau? | Trained from scratch on the stream, 128x8 beat a same-age 96x6 79–41 after 2 h, and beat the 30-hour-old 96x6 97–62 and again 99–61. 160x8 lost to 128x8 74–86 while three hours younger | Clear for 128x8; 160x8 unresolved | `CAP1`–`CAP4` |
| Does the stronger teacher help the students? | The 96x6 was flat from 130M to 250M positions and about 30 Elo higher at 300M–345M, after it started learning from 128x8 games | Suggestive only (interval −20 to +85), no control arm | `PROG3`, `TP1`, `TP2` |
| Net size at equal browser time, early (hour 5) | 64x4 beat 96x6 27–12 at a simulated 1000 ms budget | Clear at that stage; simulated time, not real | `L3b`, `L3c` |
| Net size at equal browser time, final | FINAL_SIZE | See text | `reports/zero2/arena`, `reports/zero2/final` |
| 128x8 in the browser | Lost to 96x6 8–22 at a real 1000 ms | Clear (p = 0.02), one budget | arena `g_t128x8_m96x6_1000_*` |
| int16 kernel | Equal simulations: 97.5 of 202 points (−12 Elo, interval −62 to +38). Equal time: int16 beat float 64–26 | Clear at equal time | arena `a_*`, `i16ms_*` |
| What does more thinking time buy? | FINAL_LEVELS | | `reports/zero2/final` |

Why 64x4 is deployed: at the end of training 64x4, 64x6 and 96x6 were indistinguishable at equal time, so I chose the smallest for download size and for slower devices.
The size question shifted during training: small nets won clearly early on, and the larger ones caught up as training matured.

## Progress over training

Equal search (100 simulations), Elo relative to the previous site model, fitted over two chained checkpoint ladders (`PROG`, `PROG3`; `reports/zero2/progress-elo.json`).
The chain passes through several lopsided pairings, so the absolute scale is loose: a bootstrap puts the 128x8 teacher at roughly +1450 to +1950. Differences of a few tens of Elo between neighbours are noise.

| Checkpoint | Elo |
|---|---|
| previous site model (square CNN, September) | 0 |
| hex CNN 64x6, 5M positions | +28 |
| hex CNN 64x6, 14M positions | +654 |
| transformer 96x6, 16M | +1053 |
| transformer 96x6, 30M | +1380 |
| transformer 96x6, 70M (before the learning-rate drop) | +1468 |
| transformer 96x6, 130M | +1585 |
| transformer 96x6, 250M (plateau) | +1592 |
| transformer 96x6, 345M (learning from 128x8 games) | +1624 |
| transformer 128x8 teacher, 15 h old | +1706 |

Most of the gain came in the first 30M positions. The learning-rate drop added roughly 100, the 96x6 then stalled, and the larger teacher added roughly another 100 at equal search.
The deployed 64x4 was not placed on this ladder; at equal time in the browser it matches the 96x6.

## Incidents

- **NaN divergence** at 5.9M positions: every learner diverged within minutes of each other. Not reproduced offline. Training now uses bf16 autocast, skips steps with non-finite gradients, never publishes non-finite weights and rolls back automatically; no recurrence in the following 380M positions.
- **Host failures:** one rented box went offline permanently (two hours of follower training lost); the provider rewrote an SSH key file once (25 minutes of paused transfer, no data lost).
- **Self-play diversity collapse**, hours 36–42. After the switch to the sharper 128x8 teacher, games converged onto the same lines: distinct positions at move 16 per data file fell from 97% to 62%. It showed up as a fast drop in policy loss with rising value loss. Self-play now starts 75% of games with 1–4 random moves, samples moves in proportion to visits until move 30 and less greedily afterwards; diversity returned to above 90% within half an hour. The teacher's progress had slowed during the collapse (84–76 against its snapshot from 4.5 h earlier, after 93–67 over the previous 3 h). The final weights include two hours of training on the repaired stream.

## Spend

| Box | Role | Hours | Cost |
|---|---|---|---|
| RTX 4080 Super, 16 vCPU | self-play and main learner | 43.8 | $9.47 |
| RTX 4080, 16 vCPU | self-play (second box) and A/B arms | 41.1 | $10.03 |
| RTX 4080 Super, 24 vCPU | followers and ladders (went offline at hour 22) | 21.1 | $5.52 |
| RTX 4080, 16 vCPU | followers and ladders (replacement) | 22.0 | $5.08 |
| RTX 4070 Ti Super | training-method tests | 7.7 | $1.25 |
| short-lived boxes (stuck or slow links) | – | – | $0.15 |
| **Total** | | | **about $31.5** |

Costs are hourly rate times uptime, not an invoice.

## Known limits

- No games against humans, and no measurement on a phone or low-end laptop; the 100 ms matches are only a proxy.
- GF1 was tested to depth 6 only and not in its timed mode. The external sweeps used at most about 20 distinct openings each.
- Most training-method rows rest on one run per arm.
- A rules edge case: one capture that removes both sides' last GIPF piece. The official rules name no winner but say a player must keep one GIPF piece on the board; this engine lets the capturer do it and win, GF1 awards Black, BoardSpace awards the other player. It occurred once in about 68,000 random games and in no counted match game.

## Next steps

1. Keep distilling the 128x8 teacher into the 64x4 (and try 64x6); the students were still rising when training stopped.
2. Human testing and a published strength-per-level table.
3. Timing on real phones and low-end laptops; multi-threaded WASM search.
4. Retrain the hex CNN with the final pipeline and teacher data for a fair architecture verdict; only one CNN run was compared.
5. GF1 in timed mode and at depth 7 and above.
6. Settle the last-GIPF-piece rule and align the engine.

## Where things are

- Final checkpoints (not in git): `/workspace/tmp/gipf-work/final/` holds `final64x4.pt` (deployed), `final64x6.pt`, `final96x6.pt`, `final128x8-teacher.pt`, and the newest 10M self-play positions in `data-tail/`. The rest of the 390M positions was on the rented boxes and is gone.
- To redeploy a model: `python -m ops.web_model latest --checkpoint X.pt --title "..." --cpuct 1.5 --fpu 0.1 --int16`, run the browser tests, push to `main` (the Pages workflow publishes `web/`). Rebuilding the WASM engine needs Emscripten (`engine/build_wasm.sh`).
- External engines: `benchmarks/gf1/build.sh` and `benchmarks/boardspace/build.sh` fetch and build GF1 and BoardSpace's robot into scratch directories.

## Reproduce

```
python -m training.learn --data DATA --out RUN/net --kind former --width 128 --blocks 8 --reuse 6
python -m training.selfplay --weights RUN/net/weights.pt --out DATA --cpuct 1.0 --fpu 0.1 --open-random 0.75 --temp-plies 30 --late-power 2
python -m training.learn --data DATA --out RUN/student --kind former --width 64 --blocks 4 --reuse 8 --sync-budget 1
python -m training.ladder --players a=A.pt:100:1.0:0.1 b=B.pt:100:1.0:0.1 --games 100 --out results.jsonl
node benchmarks/wasm_arena.mjs --a bin=A.bin,arch=A.json,int16=1,cpuct=1.5,fpu=0.1,ms=300 --b ... --games 20
node benchmarks/gf1_match.mjs --depth 5 --ms 300 --games 20
node benchmarks/boardspace_match.mjs --level bestbot --ms 300 --games 20
```
