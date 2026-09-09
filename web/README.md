# GIPF web table

This directory is a static GitHub Pages site. Open `index.html` directly or serve
the directory with any static web server.

`game.js` is the only rules boundary. It exports `newGame()`, `legalActions(state)`,
`applyAction(state, action)`, and `geometry`. At runtime it uses `window.GIPF_ENGINE`
when present, or loads the module URL in `config.json` as `engineUrl`. A deliberately
small opening-table shell keeps the board interactive while the engine artifact is
absent; it only renders the standard opening and disables moves until the rules
engine is present.

The adapter accepts the engine contract's `State` object directly: it can construct
`new State()`, call `state.legal_actions()`/`state.apply(action)`, read numeric
`board` and `reserves` arrays, and handle numeric push/capture action IDs. A module
may instead expose `newGame`, `legal_actions`, and `apply_action` on its namespace.

To connect the machine endpoint, set `aiEndpoint` in `config.json` to the service
origin. The UI calls `POST {aiEndpoint}/api/move` with `{state, budget_ms}` and
`GET {aiEndpoint}/api/status`. The checked-in config is empty and contains no secret.
