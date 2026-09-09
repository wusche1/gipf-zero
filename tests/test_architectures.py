import torch

from training.model import ACTIONS, HexConv2d, ModelConfig, PolicyValue, load_model


def test_hex_conv_masks_non_neighbour_corners_in_forward_and_gradient():
    conv = HexConv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 0, 0] = 100
        conv.weight[0, 0, 2, 2] = 100
    x = torch.ones((1, 1, 5, 5), requires_grad=True)
    # The only nonzero raw weights are the two excluded axial offsets.
    assert torch.count_nonzero(conv(x)) == 0
    conv(x).sum().backward()
    assert conv.weight.grad[0, 0, 0, 0] == 0
    assert conv.weight.grad[0, 0, 2, 2] == 0
    # A real axial neighbour remains connected.
    with torch.no_grad():
        conv.weight[0, 0, 0, 1] = 1
    assert torch.count_nonzero(conv(x)) > 0


def test_transformer_shapes_finite_backprop_and_checkpoint_roundtrip(tmp_path):
    config = ModelConfig('transformer', 64, 2, 'flat')
    model = PolicyValue(config)
    x = torch.randn((3, 9, 7, 7))
    logits, values = model(x)
    assert logits.shape == (3, ACTIONS)
    assert values.shape == (3,)
    loss = logits.square().mean() + values.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    path = tmp_path / 'transformer.pt'
    torch.save({'config': config.__dict__, 'model': model.state_dict()}, path)
    loaded, _ = load_model(path)
    loaded_logits, loaded_values = loaded(x)
    torch.testing.assert_close(loaded_logits, logits.detach())
    torch.testing.assert_close(loaded_values, values.detach())
