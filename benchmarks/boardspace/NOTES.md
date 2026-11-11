# BoardSpace.net GIPF robot, headless

`build.sh` fetches a portable Temurin 21 JDK and a sparse checkout of https://github.com/ddyer0/boardspace.net
(GPL-3, pinned commit in the script; only `boardspace-core/**/*.java`, `boardspace-games/gipf/*.java` and `jzlib`)
into scratch and compiles `gipf.GipfPlay` with everything it reaches (about 500 classes) plus `BsBridge.java`.
No BoardSpace source lives in this repo. The JDK needs a filesystem that allows exec, so it goes to
`/workspace/tmp/gipf-work/external/boardspace/jdk` (227 MB); sources and classes go to `/dev/shm/boardspace`.

    bash benchmarks/boardspace/build.sh
    node benchmarks/boardspace_match.mjs --level bestbot --ms 300 --games 20 --seed 1

## Stubs

BoardSpace's repository does not compile as published: eight classes it references are not in it.
`stubs/` supplies them. None is reached by the single-threaded alpha-beta search GIPF uses
(`Search_Driver.max_threads` is -1, so the thread pool is never built):
`lib.Task`, `lib.Meter`, `lib.CpuSpeed`, `bridge.BSClip`, `bridge.JLabel`,
`online.search.Sthread`, `online.search.ThreadPoolMonitor`, `online.search.UCTNode_Claude`.

## How the robot is run

GipfViewer starts a robot thread through `commonRobot.InitRobot`, then for every move calls
`PrepareToMove(player)` and `DoFullMove()` and executes the returned move on the game board, until the turn has passed.
`BsBridge` does exactly that synchronously on one `GipfBoard("gipf-standard", {0,1}, REVISION 102)`.
Its `Bot` subclass overrides only the three-argument `InitRobot` (which needs a live viewer and starts the thread)
and `getTreeViewer`; level selection, evaluation and search are GipfPlay's own.

Levels (`GipfPlay.InitRobot`): Dumbot = depth 4 with the material-only evaluator, Smartbot = depth 6 and
Bestbot = depth 7 with the full evaluator. The site offers Dumbot, Smartbot and Weakbot for GIPF. Weakbot does not get
what its name says: the switch tests `WEAKBOT_DEPTH` (3) instead of `WEAKBOT_LEVEL` (4), so Weakbot falls through to
depth 4 with the full evaluator. `--level depth3` passes 3 and gives the depth-3 robot that was meant.
During its first six moves the robot searches at most 5 deep without pruning and picks at random among moves
within 14 points of the best; the bridge seeds that generator.

## Rules

Our rules are BoardSpace's `gipf-standard` ("Gipf Standard - with Gipf pieces"): three GIPF pieces each on alternating
corners (white B5 E2 H5), 12 in reserve, only singles are inserted. `gipf` has no GIPF pieces, `gipf-tournament`
lets the players insert their own GIPF pieces, `gipf-matrx` adds potentials.

BoardSpace sequences a turn differently: rows of the player who did not insert are removed at the start of that
player's own turn, and the end of the game is only detected when a turn starts. The runner therefore sends `done`
when the inserting player is finished, and when our engine has declared a winner it lets BoardSpace play on
(`finish`) and requires the same winner.

Known differences in the source that the cross-check would flag (none came up in play):
- If one capture removes the last GIPF piece of both colours, we give the win to the capturing player, BoardSpace
  to the other one (`setFirstState` tests the player to move last).
- `doRowCaptures` walks the whole line and takes every piece that is marked for any row, so a piece beyond a gap
  that belongs to a crossing row would go too.
- The robot never chooses about GIPF pieces: it keeps its own, takes the opponent's, and keeps an opponent's
  GIPF piece that sits on the crossing of two rows.

The runner compares board, reserves, captured pieces, side to decide and kind of decision after every action of
ours and after every position-changing move of the robot; the robot's moves are translated by searching our legal
actions for the ones that produce BoardSpace's position.
