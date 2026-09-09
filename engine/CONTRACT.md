# GIPF engine contract

`gipf_engine` is the authoritative standard-GIPF rules engine.  It uses only
plain C++17 state and has no Python-specific rule logic, so the same `State`
class can be exposed through pybind11 or Emscripten/embind.

## State

`State()` creates the standard opening: White (`+1`) moves first, the six
corner spots hold alternating double pieces, and `reserves == [12, 12]`
(White, Black).  Piece values are `0` (empty), `+1`/`-1` (single), and
`+2`/`-2` (double/GIPF piece).

Python fields are `board` (37 integers), `reserves` (`[white, black]`),
`captured` (`[white, black]` basic pieces permanently lost by that colour),
`current_player`, `turn_player`, `phase` (`"push"` or `"capture"`), `winner`
(`0`, `+1`, or `-1`), and `ply` (number of inserted pieces).  `clone()` is an
independent copy; `serialize()`/`State.from_dict()` round-trip all fields.

`current_player` owns the current decision.  `turn_player` is the player who
inserted the piece that began the current turn and remains fixed while rows are
captured.  `apply(action)` mutates in place and rejects an action not returned
by `legal_actions()` with `ValueError`.

## Geometry

The board is a radius-3 axial hex.  `geometry()` returns `coordinates` in
board-index order, 21 unoriented `lines`, and 42 oriented `rays`. Coordinates
are `[q, r]`, enumerated by `r=-3..3`, then `q=max(-3,-r-3)..min(3,-r+3)`.
The 21 lines are seven each of constant `q`, constant `r`, then constant
`s=-q-r`, with constants `-3..3`; cells are ordered by increasing `r`,
increasing `q`, then increasing `q` respectively.  Ray `2*line_id` follows
the line order and ray `2*line_id+1` is its reverse.

## Actions

Push actions are `0..41`; action `ray_id` inserts one single piece at the
near end of that ray and shifts its contiguous occupied prefix one step. A
full ray is illegal.

Capture actions are `42 + line_id * 128 + mask`. `line_id` refers to the
unoriented geometry line.  A legal action removes the entire occupied segment
that contains a run of at least four pieces of `current_player`'s colour:
all singles in that segment are compulsory. `mask` has one bit per position
in the line and may select any subset of doubles in that segment to remove;
all other bits must be zero. Own removed material returns to its reserve
(a double returns two basics); opponent material is captured.

After every capture the engine re-evaluates rows.  The original mover gets
priority whenever that player has a row; otherwise the other colour captures.
The player removing rows chooses optional doubles.  When an action removes an
opponent's last double, that actor wins immediately, even when its own last
double is removed in the same action.  Reserve exhaustion is checked only
after all captures settle, for the player next due to insert.
