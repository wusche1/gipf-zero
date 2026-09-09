# Gipf reference baselines

These v1.2-paired-seeds players are fixed pre-RL references. They use the authoritative
`gipf_engine.State` API and keep all rule decisions, including capture choices,
in the engine.

The frozen player settings are `random`; `greedy(capture_depth=6)`;
`minimax(depth=2, capture_depth=6)`; and
`mcts(simulations=128, exploration=1.35, rollout_depth=2)`. Every search also
accepts a monotonic per-move time budget. Minimax uses iterative deepening and
only commits an action after a full root iteration; capture quiescence is never
longer than six decisions. The searches maximize when `current_player` is the
root player and minimize otherwise, which correctly handles capture phases in
which one player owns consecutive decisions.

Greedy and minimax randomly select among exactly equal root values using their
seeded PRNG. The harness resets each player's PRNG from the colour-balanced
pair id, with distinct A/B streams. This supplies reproducible opening
diversity without repeating a prior pair after JSONL resume or changing the
evaluation weights. MCTS uses the same seeded source for expansion and rollout
choices.

The fixed evaluation from a player's perspective is:

```
100000 * terminal outcome
+ 100 * (own total material - opponent total material)
+   4 * (own reserve - opponent reserve)
+   3 * (own doubles - opponent doubles)
+  12 * (own open three-or-more line threats - opponent threats)
```

Total material is reserve plus one per single and two per double. A threat is a
same-colour run of at least three with an adjacent empty cell, evaluated on the
21 public-contract geometry lines. Doubles are a modest safety proxy; this is a
reference heuristic rather than a complete positional evaluation.

Run a balanced, resumable match with equal per-move budgets:

```
/venv/main/bin/python -m baselines.evaluate \
  --white minimax --black mcts --games 100 --move-seconds 0.05 \
  --max-insertion-plies 400 --max-decisions 2000 \
  --output artifacts/minimax-v-mcts.json
```

Even game ids play A as White and odd ids play A as Black; both games in a pair
share their state-factory seed. The harness flushes one record per game to
the same output stem with a `.jsonl` suffix and can resume from it. The JSON summary reports participant A
wins/losses, cutoffs and errors separately, plus a 95% Wilson interval over
decisive games.
