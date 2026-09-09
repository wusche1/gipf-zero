"""Query-factorized Transformer policy without a 2,730-way projection.

The encoder mean-pools 37 axial board tokens.  Its input keeps the normal
actor-relative nine planes and appends internal row, double and query-type
highlight planes.  Capture action log probabilities are a row softmax followed
by conditional binary take/keep scores in engine line order.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import ACTIONS, COORDS, GEO, MASK, PLANES, ROWS, COLS, encode


class _EncodedState:
    """Discrete state view reconstructed from existing actor-relative planes."""
    def __init__(self, x):
        self.board = [0] * 37; self.current_player = 1; self.winner = 0
        for cell in range(37):
            row, col = ROWS[cell], COLS[cell]
            self.board[cell] = 1 if x[0,row,col] else 2 if x[1,row,col] else -1 if x[2,row,col] else -2 if x[3,row,col] else 0
        self.reserves = [int(round(float(x[5,3,3]) * 18)), int(round(float(x[6,3,3]) * 18))]
        self.phase = 'capture' if x[7,3,3] > .5 else 'push'
        self.turn_player = 1 if x[8,3,3] > .5 else -1

    def legal_actions(self):
        if self.phase == 'push':
            if self.reserves[0] <= 0: return []
            return [i for i, ray in enumerate(GEO['rays']) if any(self.board[cell] == 0 for cell in ray)]
        actions = []
        for line_id, line in enumerate(GEO['lines']):
            start = 0
            while start < len(line):
                while start < len(line) and self.board[line[start]] == 0: start += 1
                end = start
                while end < len(line) and self.board[line[end]] != 0: end += 1
                segment = line[start:end]; run = best = 0
                for cell in segment:
                    run = run + 1 if self.board[cell] > 0 else 0; best = max(best, run)
                if best >= 4:
                    doubles = [i for i, cell in enumerate(line) if cell in segment and abs(self.board[cell]) == 2]
                    for selected in range(1 << len(doubles)):
                        mask = sum(1 << doubles[i] for i in range(len(doubles)) if selected & (1 << i))
                        actions.append(42 + line_id * 128 + mask)
                start = end + 1
        return actions


class QueryTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, blocks = config.width, config.blocks
        if width % 4: raise ValueError('query transformer width must divide four heads')
        self.config = config
        self.token = nn.Linear(PLANES + 3, width)
        self.coordinate = nn.Linear(2, width, bias=False)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.0, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks)
        self.register_buffer('coordinates', torch.tensor(COORDS, dtype=torch.float32).div_(3).unsqueeze(0))
        self.push = nn.Linear(width, 42)
        self.take = nn.Linear(width, 1)
        self.value = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def _device(self): return self.coordinates.device

    def _feature(self, state, board=None, reserve=None, row_cells=(), double_cell=None, double_query=False):
        relative = np.asarray(state.board if board is None else board, np.int8)
        if board is None: relative = relative * state.current_player
        x = encode(state); x[:4] = 0
        for cell, piece in enumerate(relative):
            if piece: x[(1 if piece == 2 else 0) if piece > 0 else (3 if piece == -2 else 2), ROWS[cell], COLS[cell]] = 1
        if reserve is not None: x[5] = MASK * (reserve / 18)
        query = np.zeros((3, 7, 7), np.float32)
        if row_cells: query[0, ROWS[list(row_cells)], COLS[list(row_cells)]] = 1
        if double_cell is not None: query[1, ROWS[double_cell], COLS[double_cell]] = 1
        query[2] = MASK * float(double_query)
        return np.concatenate((x, query))

    def _embed(self, features):
        x = torch.from_numpy(np.stack(features)).to(self._device())
        tokens = x[:, :, ROWS, COLS].transpose(1, 2)
        return self.encoder(self.token(tokens) + self.coordinate(self.coordinates)).mean(1)

    @staticmethod
    def _segments(state):
        """Derive legal rows from actor-relative occupancy, retaining full segments."""
        legal = state.legal_actions(); legal_lines = {(a - 42) // 128 for a in legal if a >= 42}
        board = np.asarray(state.board, np.int8) * state.current_player
        result = []
        for line_id in legal_lines:
            line = GEO['lines'][line_id]; start = 0
            while start < len(line):
                while start < len(line) and board[line[start]] == 0: start += 1
                end = start
                while end < len(line) and board[line[end]] != 0: end += 1
                segment = line[start:end]
                run = best = 0
                for cell in segment:
                    run = run + 1 if board[cell] > 0 else 0; best = max(best, run)
                if best >= 4: result.append((line_id, tuple(segment)))
                start = end + 1
        return board, result

    def _capture_reference(self, state):
        board, rows = self._segments(state)
        legal = state.legal_actions(); masks = {}
        for action in legal:
            if action >= 42: masks.setdefault((action - 42) // 128, []).append((action - 42) % 128)
        row_features = [self._feature(state, board, row_cells=segment) for _, segment in rows]
        row_scores = self.take(self._embed(row_features)).squeeze(1)
        row_log = F.log_softmax(row_scores, 0)
        contexts = []
        own_reserve = state.reserves[0 if state.current_player == 1 else 1]
        for index, (line_id, segment) in enumerate(rows):
            allowed = masks[line_id]; line = GEO['lines'][line_id]
            doubles = [(line.index(cell), cell) for cell in segment if abs(board[cell]) == 2 and any(mask & (1 << line.index(cell)) for mask in allowed)]
            contexts.append((index, line_id, segment, allowed, doubles, own_reserve))
        take_scores = {}
        for depth in range(max((len(c[4]) for c in contexts), default=0)):
            records = []
            for context in contexts:
                row_index, _, segment, allowed, doubles, reserve = context
                if depth >= len(doubles): continue
                earlier = sum(1 << pos for pos, _ in doubles[:depth])
                prefixes = {mask & earlier for mask in allowed}
                for prefix in prefixes:
                    temp, own = self._temporary_board(board, segment, doubles, prefix, depth, reserve)
                    records.append((row_index, prefix, self._feature(state, temp, own, segment, doubles[depth][1], True)))
            if records:
                scores = self.take(self._embed([r[2] for r in records])).squeeze(1)
                take_scores.update({(row, depth, prefix): score for (row, prefix, _), score in zip(records, scores)})
        output = torch.full((ACTIONS,), -1e9, device=self._device())
        for row_index, line_id, _, allowed, doubles, _ in contexts:
            for mask in allowed:
                value = row_log[row_index]; prefix = 0
                for depth, (pos, _) in enumerate(doubles):
                    score = take_scores[row_index, depth, prefix]; chosen = bool(mask & (1 << pos))
                    value = value + F.logsigmoid(score if chosen else -score)
                    if chosen: prefix |= 1 << pos
                output[42 + line_id * 128 + mask] = value
        return output

    def _capture_batch(self, states):
        """Pack every row and hypothetical double-prefix query across states."""
        grouped, row_features = [], []
        for state_index, state in enumerate(states):
            board, rows = self._segments(state); masks = {}
            for action in state.legal_actions():
                if action >= 42: masks.setdefault((action - 42) // 128, []).append((action - 42) % 128)
            contexts = []
            reserve = state.reserves[0 if state.current_player == 1 else 1]
            for row_index, (line_id, segment) in enumerate(rows):
                line = GEO['lines'][line_id]; allowed = masks[line_id]
                doubles = [(line.index(cell), cell) for cell in segment if abs(board[cell]) == 2 and any(mask & (1 << line.index(cell)) for mask in allowed)]
                contexts.append({'row':row_index,'line':line_id,'segment':segment,'allowed':allowed,'doubles':doubles,'reserve':reserve})
                row_features.append(self._feature(state, board, row_cells=segment))
            grouped.append((state, board, contexts))
        row_scores = self.take(self._embed(row_features)).squeeze(1)
        cursor = 0; row_logs = {}
        for state_index, (_, _, contexts) in enumerate(grouped):
            row_logs.update({(state_index, context['row']): score for context, score in zip(contexts, F.log_softmax(row_scores[cursor:cursor + len(contexts)], 0))})
            cursor += len(contexts)
        queries = []
        for state_index, (state, board, contexts) in enumerate(grouped):
            for context in contexts:
                doubles = context['doubles']
                for depth in range(len(doubles)):
                    earlier = sum(1 << pos for pos, _ in doubles[:depth])
                    for prefix in {mask & earlier for mask in context['allowed']}:
                        temp, own = self._temporary_board(board, context['segment'], doubles, prefix, depth, context['reserve'])
                        queries.append((state_index, context['row'], depth, prefix, self._feature(state, temp, own, context['segment'], doubles[depth][1], True)))
        scores = self.take(self._embed([query[4] for query in queries])).squeeze(1) if queries else []
        take_scores = {(state_index, row, depth, prefix): score for (state_index, row, depth, prefix, _), score in zip(queries, scores)}
        outputs = []
        for state_index, (_, _, contexts) in enumerate(grouped):
            output = torch.full((ACTIONS,), -1e9, device=self._device())
            for context in contexts:
                for mask in context['allowed']:
                    value, prefix = row_logs[state_index, context['row']], 0
                    for depth, (pos, _) in enumerate(context['doubles']):
                        score = take_scores[state_index, context['row'], depth, prefix]
                        chosen = bool(mask & (1 << pos)); value = value + F.logsigmoid(score if chosen else -score)
                        if chosen: prefix |= 1 << pos
                    output[42 + context['line'] * 128 + mask] = value
            outputs.append(output)
        return outputs

    @staticmethod
    def _temporary_board(board, segment, doubles, prefix, depth, reserve):
        """Apply mandatory singles and earlier binary GIPF choices to a query board."""
        temp, own = board.copy(), reserve
        for pos, cell in doubles[:depth]:
            if prefix & (1 << pos):
                if temp[cell] > 0: own += 2
                temp[cell] = 0
        for cell in segment:
            if abs(temp[cell]) == 1:
                if temp[cell] > 0: own += 1
                temp[cell] = 0
        return temp, own

    @staticmethod
    def _coerce_states(states):
        if isinstance(states, torch.Tensor):
            if states.ndim != 4 or tuple(states.shape[1:]) != (9, 7, 7): raise ValueError('expected [B,9,7,7] input')
            return [_EncodedState(x) for x in states.detach().cpu().numpy()]
        elif not isinstance(states, (list, tuple)): states = [states]
        return states

    def forward_reference(self, states):
        """Unpacked capture evaluation retained to check the packed forward path."""
        states = self._coerce_states(states)
        base = self._embed([self._feature(state) for state in states])
        values = self.value(base).squeeze(1)
        output = torch.full((len(states), ACTIONS), -1e9, device=self._device())
        for index, state in enumerate(states):
            if state.phase == 'push':
                actions = torch.tensor(state.legal_actions(), device=self._device())
                output[index, actions] = F.log_softmax(self.push(base[index])[actions], 0)
            else: output[index] = self._capture_reference(state)
        return output, values

    def forward(self, states):
        """Return [B,2730] legal-action log-probs and [B] values for engine states."""
        states = self._coerce_states(states)
        base = self._embed([self._feature(state) for state in states])
        values = self.value(base).squeeze(1)
        output = torch.full((len(states), ACTIONS), -1e9, device=self._device())
        captures = []
        for index, state in enumerate(states):
            if state.phase == 'push':
                actions = torch.tensor(state.legal_actions(), device=self._device())
                output[index, actions] = F.log_softmax(self.push(base[index])[actions], 0)
            else: captures.append((index, state))
        if captures:
            for (index, _), policy in zip(captures, self._capture_batch([state for _, state in captures])): output[index] = policy
        return output, values
