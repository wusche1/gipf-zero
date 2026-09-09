import math
import numpy as np
import torch
import gipf_engine as ge

from training.model import ModelConfig, ROWS, COLS, encode
from training.query_transformer import QueryTransformer


def capture_state():
    board = [0] * 37; line = ge.geometry()['lines'][3]
    for cell in line[:4]: board[cell] = 1
    board[line[4]], board[line[5]] = 2, -2
    used = [sum(abs(x) for x in board if x > 0), sum(abs(x) for x in board if x < 0)]
    return ge.State.from_dict({'board':board,'reserves':[8,10],'captured':[18-8-used[0],18-10-used[1]],'current_player':1,'turn_player':1,'phase':'capture','winner':0,'ply':1})


def test_query_transformer_push_shape_normalization_backward_and_roundtrip(tmp_path):
    model = QueryTransformer(ModelConfig('transformer', 64, 2, 'query'))
    logits, values = model([ge.State(), ge.State()]); legal = ge.State().legal_actions()
    assert logits.shape == (2, 2730) and values.shape == (2,)
    torch.testing.assert_close(logits[:, legal].exp().sum(1), torch.ones(2))
    assert torch.all(logits[:, 42:] == -1e9)
    tensor_logits, tensor_values = model(torch.from_numpy(np.stack([encode(ge.State())])))
    torch.testing.assert_close(tensor_logits, logits[:1])
    torch.testing.assert_close(tensor_values, values[:1])
    (-logits[:, legal].mean() + values.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path/'query.pt'; torch.save(model.state_dict(), path)
    clone = QueryTransformer(ModelConfig('transformer',64,2,'query')); clone.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(clone([ge.State()])[0], model([ge.State()])[0].detach())


def test_query_capture_legal_normalization_and_temporary_double_removal():
    state = capture_state(); model = QueryTransformer(ModelConfig('transformer', 64, 2, 'query'))
    logits, _ = model([state]); legal = state.legal_actions()
    torch.testing.assert_close(logits[0, legal].exp().sum(), torch.tensor(1.0))
    assert torch.all(logits[0, np.setdiff1d(np.arange(2730), legal)] == -1e9)
    board, rows = model._segments(state); _, segment = rows[0]; line = ge.geometry()['lines'][3]
    doubles = [(4, line[4]), (5, line[5])]
    temporary_board, reserve = model._temporary_board(board, segment, doubles, 1 << 4, 1, 8)
    assert all(temporary_board[cell] == 0 for cell in line[:5]) and temporary_board[line[5]] == -2
    assert reserve == 14  # four own singles and one own double return to plane 5
    temporary = model._feature(state, temporary_board, reserve, segment, line[5], True)
    assert temporary[10].sum() == 1 and temporary[9].sum() >= 4
    for action in legal:
        mask = (action - 42) % 128
        final_board, final_reserve = model._temporary_board(board, segment, doubles, mask, len(doubles), 8)
        authoritative = state.clone(); authoritative.apply(action)
        assert final_board.tolist() == authoritative.board
        assert final_reserve == authoritative.reserves[0]


def test_query_capture_conditions_each_double_in_line_order():
    state = capture_state(); model = QueryTransformer(ModelConfig('transformer', 64, 2, 'query'))
    line = ge.geometry()['lines'][3]; scores = {ROWS[line[4]] * 7 + COLS[line[4]]: .25, ROWS[line[5]] * 7 + COLS[line[5]]: -.75}
    with torch.no_grad(): model.take.weight.zero_(); model.take.weight[0, 0] = 1; model.take.bias.zero_()
    def fake_embed(features):
        rows = []
        for feature in features:
            value = scores.get(int(np.argmax(feature[10])), 0.0) if feature[10].sum() else 0.0
            rows.append([value] + [0.0] * 63)
        return torch.tensor(rows)
    model._embed = fake_embed
    logits, _ = model([state]); both = 42 + 3 * 128 + (1 << 4) + (1 << 5)
    expected = torch.nn.functional.logsigmoid(torch.tensor(.25)) + torch.nn.functional.logsigmoid(torch.tensor(-.75))
    torch.testing.assert_close(logits[0, both], expected)
