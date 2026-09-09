"""Optional fixed-batch CUDA-graph inference for self-play.

CUDA graphs require stable memory addresses.  ``CudaGraphInference`` therefore
captures an eval-mode model with a fixed, padded input batch and returns views
of its static outputs.  The views are valid only until the next ``forward``.
Results match eager inference on the same *padded* batch; an eager smaller
batch can select different kernels and have small floating-point differences.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


class CudaGraphInference:
    """Lazily capture a static CUDA graph, with a safe eager fallback.

    In-place optimizer updates keep parameter storage unchanged, so replay uses
    the new weights.  Replacing parameter or buffer storage causes a recapture.
    A capture failure permanently disables this instance and is logged once.
    """

    def __init__(self, model: torch.nn.Module, device: str | torch.device,
                 batch_size: int = 0, logger: Callable[[str], None] = print):
        self.model = model
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.logger = logger
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input: torch.Tensor | None = None
        self.static_logits: torch.Tensor | None = None
        self.static_values: torch.Tensor | None = None
        self.signature: tuple | None = None
        self.disabled = self.batch_size <= 0 or self.device.type != 'cuda'
        self._failure_logged = False
        self._search_scope_depth = 0

    def _signature(self) -> tuple:
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        return tuple((str(t.device), str(t.dtype), tuple(t.shape), t.data_ptr()) for t in tensors)

    def _model_device(self) -> torch.device:
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            return tensor.device
        return self.device

    def _eager(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(features.to(self._model_device()))

    def _validate_storage(self) -> None:
        """Refresh graph state if a model has been moved or storage replaced."""
        model_device = self._model_device()
        if model_device != self.device:
            # ``Module.to`` or replacement of storage changed the model's
            # device.  A CUDA destination can be captured anew; CPU falls
            # back to eager without attempting a CUDA graph.
            self.device = model_device
            self.graph = self.static_input = self.static_logits = self.static_values = None
            self.signature = None
            self.disabled = self.batch_size <= 0 or self.device.type != 'cuda'
        elif self.graph is not None and self.signature != self._signature():
            self.graph = self.static_input = self.static_logits = self.static_values = None
            self.signature = None

    @contextmanager
    def search_scope(self):
        """Validate once for a synchronous MCTS search with stable model storage.

        Standalone ``forward`` calls retain validation on every call.  MCTS
        never updates or moves the model during ``search``, so this avoids
        repeatedly walking every parameter and buffer for each leaf batch.
        """
        self._validate_storage()
        self._search_scope_depth += 1
        try:
            yield self
        finally:
            self._search_scope_depth -= 1

    def _disable(self, exc: Exception) -> None:
        self.disabled = True
        self.graph = self.static_input = self.static_logits = self.static_values = None
        if not self._failure_logged:
            self.logger(f'[gipf] CUDA graph inference disabled: {type(exc).__name__}: {exc}')
            self._failure_logged = True

    def _capture(self) -> None:
        if self.model.training:
            raise RuntimeError('CUDA graph inference requires model.eval()')
        self.static_input = torch.zeros((self.batch_size, 9, 7, 7),
                                        device=self.device, dtype=torch.float32)
        # Warm up allocator and cuDNN on a non-default stream before capture.
        current = torch.cuda.current_stream(self.device)
        warmup = torch.cuda.Stream(device=self.device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                self.model(self.static_input)
        current.wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            logits, values = self.model(self.static_input)
        self.graph = graph
        self.static_logits, self.static_values = logits, values
        self.signature = self._signature()

    @torch.inference_mode()
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a CPU or CUDA float tensor, falling back outside capacity."""
        if features.ndim != 4 or tuple(features.shape[1:]) != (9, 7, 7):
            raise ValueError(f'expected [B,9,7,7] features, got {tuple(features.shape)}')
        n = int(features.shape[0])
        if n == 0:
            raise ValueError('cannot evaluate an empty batch')
        if not self._search_scope_depth:
            self._validate_storage()
        if self.disabled or self.model.training or n > self.batch_size:
            return self._eager(features)
        try:
            if self.graph is None:
                self._capture()
            assert self.graph is not None and self.static_input is not None
            assert self.static_logits is not None and self.static_values is not None
            self.static_input.zero_()
            self.static_input[:n].copy_(features)
            self.graph.replay()
            return self.static_logits[:n], self.static_values[:n]
        except Exception as exc:
            self._disable(exc)
            return self._eager(features)
