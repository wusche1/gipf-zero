import pytest
import torch

from training.inference import CudaGraphInference
from training.model import ModelConfig, PolicyValue


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def padded_eager(model, features, capacity=128):
    padded = torch.zeros((capacity, 9, 7, 7), device='cuda')
    padded[:len(features)].copy_(features)
    return model(padded)


def test_cuda_graph_matches_padded_eager_at_multiple_batch_sizes():
    torch.manual_seed(7)
    model = PolicyValue(ModelConfig('resnet', 16, 1, 'flat')).cuda().eval()
    runner = CudaGraphInference(model, 'cuda', 128)
    for size in (1, 17, 64, 128):
        x = torch.randn((size, 9, 7, 7), device='cuda')
        expected_logits, expected_values = padded_eager(model, x)
        got_logits, got_values = runner.forward(x.cpu())
        # This compares like for like: the graph is a fixed padded B=128 graph.
        torch.testing.assert_close(got_logits, expected_logits[:size], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got_values, expected_values[:size], rtol=1e-5, atol=1e-6)
        # A smaller eager batch may choose different kernels, so it is only
        # expected to agree numerically, rather than bit-for-bit.
        small_logits, small_values = model(x)
        torch.testing.assert_close(got_logits, small_logits, rtol=2e-4, atol=3e-4)
        torch.testing.assert_close(got_values, small_values, rtol=2e-4, atol=3e-4)


def test_cuda_graph_observes_inplace_optimizer_updates():
    torch.manual_seed(8)
    model = PolicyValue(ModelConfig('resnet', 16, 1, 'flat')).cuda().eval()
    runner = CudaGraphInference(model, 'cuda', 128)
    x = torch.randn((17, 9, 7, 7), device='cuda')
    before_logits, before_values = runner.forward(x.cpu())
    before_logits, before_values = before_logits.clone(), before_values.clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(x)
    (logits.square().mean() + values.square().mean()).backward()
    optimizer.step()
    model.eval()

    expected_logits, expected_values = padded_eager(model, x)
    got_logits, got_values = runner.forward(x.cpu())
    torch.testing.assert_close(got_logits, expected_logits[:17], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_values, expected_values[:17], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(got_logits, before_logits)
    assert not torch.allclose(got_values, before_values)


def test_cuda_graph_recaptures_after_parameter_storage_replacement():
    torch.manual_seed(9)
    model = PolicyValue(ModelConfig('resnet', 16, 1, 'flat')).cuda().eval()
    runner = CudaGraphInference(model, 'cuda', 128)
    x = torch.randn((17, 9, 7, 7), device='cuda')
    runner.forward(x.cpu())
    parameter = next(model.parameters())
    parameter.data = parameter.detach().clone()
    expected_logits, expected_values = padded_eager(model, x)
    got_logits, got_values = runner.forward(x.cpu())
    torch.testing.assert_close(got_logits, expected_logits[:17], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_values, expected_values[:17], rtol=1e-5, atol=1e-6)
