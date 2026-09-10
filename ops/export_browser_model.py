"""Export the served flat-policy checkpoint and browser parity fixtures to ONNX.

The model accepts the existing actor-relative nine planes; game legality stays
in the WASM engine and must be applied to the raw 2,730 action logits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import gipf_engine as ge

from training.model import ACTIONS, encode, load_model
from training.search import choose_action
from training.native_search import NativeBatchedMCTS, NativeNode

ROOT = Path(__file__).resolve().parents[1]
TAG = "20260909T235853Z"


class BrowserPolicyValue(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, board_features):
        logits, value = self.model(board_features)
        return logits, value


class PortableGroupNorm(torch.nn.Module):
    """Static N=1 GroupNorm expressed with broadly-supported ONNX primitives.

    PyTorch's standard ONNX lowering uses InstanceNormalization after reshaping.
    Some ORT WebGPU builds have produced incorrect values for that path.  The
    browser candidate is deliberately fixed to one board per inference, so its
    N/C/H/W shape is known and this equivalent reduction form avoids that op.
    """
    def __init__(self, source: torch.nn.GroupNorm):
        super().__init__()
        self.groups = source.num_groups
        self.channels = source.num_channels
        self.eps = source.eps
        self.register_buffer("weight", source.weight.detach().clone())
        self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, x):
        # This model candidate is intentionally static batch one / 7x7 board.
        grouped = x.reshape(1, self.groups, -1)
        mean = grouped.mean(dim=2, keepdim=True)
        centered = grouped - mean
        variance = (centered * centered).mean(dim=2, keepdim=True)
        normalized = centered / torch.sqrt(variance + self.eps)
        normalized = normalized.reshape(1, self.channels, 7, 7)
        return normalized * self.weight.reshape(1, self.channels, 1, 1) + self.bias.reshape(1, self.channels, 1, 1)


def portable_static_wrapper(model: torch.nn.Module) -> BrowserPolicyValue:
    """Copy a model and replace every GroupNorm only for static browser export."""
    portable = copy.deepcopy(model).eval()

    def replace(module: torch.nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.GroupNorm):
                setattr(module, name, PortableGroupNorm(child))
            else:
                replace(child)
    replace(portable)
    return BrowserPolicyValue(portable).eval()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_states(count: int = 16, seed: int = 20260910) -> list[ge.State]:
    """Deterministically collect legal states from actual random trajectories."""
    rng = random.Random(seed)
    pushes: list[ge.State] = [ge.State()]
    captures: list[ge.State] = []
    for _ in range(200):
        state = ge.State()
        for ply in range(240):
            if state.winner:
                break
            if state.phase == "capture" and len(captures) < count // 2:
                captures.append(state.clone())
            elif state.phase == "push" and ply and len(pushes) < count // 2:
                pushes.append(state.clone())
            actions = state.legal_actions()
            if not actions:
                break
            state.apply(rng.choice(actions))
            if len(pushes) >= count // 2 and len(captures) >= count // 2:
                return pushes[:count // 2] + captures[:count // 2]
    raise RuntimeError("did not find enough reachable capture states")


def forward_timing(model, features: np.ndarray) -> dict[str, float]:
    torch.set_num_threads(1)
    result = {}
    with torch.inference_mode():
        for size in (1, 8, 32):
            sample = np.concatenate([features] * ((size + len(features) - 1) // len(features)))[:size].copy()
            tensor = torch.from_numpy(sample)
            for _ in range(5):
                model(tensor)
            samples = []
            for _ in range(20):
                start = time.perf_counter(); model(tensor); samples.append(time.perf_counter() - start)
            result[str(size)] = float(np.median(samples) * 1000)
    return result


def ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def ort_timing(session: ort.InferenceSession, feature: np.ndarray) -> float:
    for _ in range(5):
        session.run(None, {"board_features": feature})
    samples = []
    for _ in range(50):
        start = time.perf_counter(); session.run(None, {"board_features": feature}); samples.append(time.perf_counter() - start)
    return float(np.median(samples) * 1000)


def search_timing(model, states: list[ge.State]) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    rows = []
    for index, state in enumerate(states):
        start = time.perf_counter()
        action, details = choose_action(model, state, "cpu", simulations=128, budget_ms=30_000)
        rows.append({"fixture": index, "phase": state.phase, "action": action,
                     "elapsed_ms": (time.perf_counter() - start) * 1000,
                     "simulations": details["simulations"]})
    return rows


def native_search_timing(model, states: list[ge.State]) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    rows = []
    for index, state in enumerate(states):
        root = NativeNode(state.clone()); search = NativeBatchedMCTS(model, "cpu", seed=0)
        start = time.perf_counter(); policy = search.search([root], simulations=128)[0]
        rows.append({"fixture": index, "phase": state.phase,
                     "action": int(root.actions[int(np.argmax(policy))]),
                     "elapsed_ms": (time.perf_counter() - start) * 1000,
                     "simulations": int(root.n.sum())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/champion.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "web/models")
    parser.add_argument("--skip-search-benchmark", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    model, checkpoint_data = load_model(checkpoint, "cpu")
    model.eval(); torch.set_num_threads(1)
    wrapper = BrowserPolicyValue(model).eval()
    portable_wrapper = portable_static_wrapper(model)
    features = np.stack([encode(state) for state in fixture_states()]).astype(np.float32)
    onnx_path = args.output_dir / "champion.onnx"
    torch.onnx.export(wrapper, torch.from_numpy(features[:1]), onnx_path,
                      input_names=["board_features"], output_names=["policy_logits", "value"],
                      dynamic_axes={"board_features": {0: "batch"}, "policy_logits": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, external_data=False)
    onnx.checker.check_model(str(onnx_path))
    single_path = args.output_dir / "champion-single.onnx"
    torch.onnx.export(wrapper, torch.from_numpy(features[:1]), single_path,
                      input_names=["board_features"], output_names=["policy_logits", "value"],
                      opset_version=17, do_constant_folding=True, external_data=False)
    onnx.checker.check_model(str(single_path))
    portable_path = args.output_dir / "champion-portable.onnx"
    # The legacy exporter preserves opset 17 here.  The current dynamo exporter
    # emits ReduceMean axes as inputs and cannot down-convert this graph to 17.
    torch.onnx.export(portable_wrapper, torch.from_numpy(features[:1]), portable_path,
                      input_names=["board_features"], output_names=["policy_logits", "value"],
                      opset_version=17, do_constant_folding=True, external_data=False, dynamo=False)
    onnx.checker.check_model(str(portable_path))
    portable_graph = onnx.load(portable_path, load_external_data=False)
    portable_ops = sorted({node.op_type for node in portable_graph.graph.node})
    if "InstanceNormalization" in portable_ops:
        raise AssertionError("portable candidate still contains InstanceNormalization")
    session, single_session, portable_session = ort_session(onnx_path), ort_session(single_path), ort_session(portable_path)
    validation_batches = {}
    with torch.inference_mode():
        for size in (1, 8, 32):
            batch = np.concatenate([features] * ((size + len(features) - 1) // len(features)))[:size].copy()
            native_logits, native_values = wrapper(torch.from_numpy(batch))
            ort_logits, ort_values = session.run(None, {"board_features": batch})
            np.testing.assert_allclose(ort_logits, native_logits.numpy(), rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(ort_values, native_values.numpy(), rtol=2e-5, atol=2e-6)
            validation_batches[str(size)] = {"max_abs_policy_logits": float(np.max(np.abs(ort_logits - native_logits.numpy()))),
                                             "max_abs_value": float(np.max(np.abs(ort_values - native_values.numpy())))}
        native_logits, native_values = wrapper(torch.from_numpy(features))
    validation = {"batches": validation_batches, "rtol": 2e-5, "atol": 2e-6}
    states = fixture_states()
    static_validation = {"max_abs_policy_logits": 0.0, "max_abs_value": 0.0, "legal_action_parity": True, "fixtures": []}
    with torch.inference_mode():
        for index, state in enumerate(states):
            feature = features[index:index + 1]
            torch_logits, torch_values = wrapper(torch.from_numpy(feature))
            static_logits, static_values = single_session.run(None, {"board_features": feature})
            np.testing.assert_allclose(static_logits, torch_logits.numpy(), rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(static_values, torch_values.numpy(), rtol=2e-5, atol=2e-6)
            legal = state.legal_actions()
            native_action = int(legal[int(np.argmax(torch_logits.numpy()[0, legal]))])
            static_action = int(legal[int(np.argmax(static_logits[0, legal]))])
            static_validation["max_abs_policy_logits"] = max(static_validation["max_abs_policy_logits"], float(np.max(np.abs(static_logits - torch_logits.numpy()))))
            static_validation["max_abs_value"] = max(static_validation["max_abs_value"], float(np.max(np.abs(static_values - torch_values.numpy()))))
            static_validation["legal_action_parity"] &= native_action == static_action
            static_validation["fixtures"].append({"id": index, "phase": state.phase, "native_action": native_action, "single_action": static_action})
    if not static_validation["legal_action_parity"]:
        raise AssertionError("static ONNX changed a legal fixture action")
    portable_validation = {"max_abs_policy_logits": 0.0, "max_abs_value": 0.0,
                           "legal_action_parity": True, "fixtures": [], "operators": portable_ops}
    with torch.inference_mode():
        for index, state in enumerate(states):
            feature = features[index:index + 1]
            torch_logits, torch_values = wrapper(torch.from_numpy(feature))
            portable_logits, portable_values = portable_session.run(None, {"board_features": feature})
            # The primitive form and its ONNX lowering both remain within 1e-5
            # of the trained PyTorch model, including capture decision states.
            np.testing.assert_allclose(portable_logits, torch_logits.numpy(), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(portable_values, torch_values.numpy(), rtol=1e-5, atol=1e-5)
            legal = state.legal_actions()
            native_action = int(legal[int(np.argmax(torch_logits.numpy()[0, legal]))])
            portable_action = int(legal[int(np.argmax(portable_logits[0, legal]))])
            portable_validation["max_abs_policy_logits"] = max(portable_validation["max_abs_policy_logits"], float(np.max(np.abs(portable_logits - torch_logits.numpy()))))
            portable_validation["max_abs_value"] = max(portable_validation["max_abs_value"], float(np.max(np.abs(portable_values - torch_values.numpy()))))
            portable_validation["legal_action_parity"] &= native_action == portable_action
            portable_validation["fixtures"].append({"id": index, "phase": state.phase, "native_action": native_action, "portable_action": portable_action})
    if not portable_validation["legal_action_parity"]:
        raise AssertionError("portable ONNX changed a legal fixture action")
    fixtures = [{"id": index, "phase": state.phase, "state": state.serialize(),
                 "encoded_input": features[index].tolist(), "legal_actions": state.legal_actions(),
                 "native_policy_logits": native_logits[index].numpy().tolist(), "native_value": float(native_values[index])}
                for index, state in enumerate(states)]
    champion_meta = json.loads((ROOT / "reports/champion.json").read_text())
    fixture = {"schema": "gipf-browser-onnx-fixtures-v1", "input": {"name": "board_features", "dtype": "float32", "shape": ["batch", 9, 7, 7], "encoding": "training.model.encode actor-relative planes"},
               "outputs": {"policy_logits": {"dtype": "float32", "shape": ["batch", ACTIONS], "meaning": "raw flat logits; mask legal engine action IDs before softmax"}, "value": {"dtype": "float32", "shape": ["batch"], "range": "[-1, 1]"}},
               "fixture_seed": 20260910, "reachable_state_generation": "random legal trajectories from ge.State; eight push and eight capture decision states", "fixtures": fixtures}
    fixture_path = args.output_dir / "champion-fixtures.json"; fixture_path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    benchmark = {"schema": "gipf-browser-onnx-benchmark-v1", "device": "CPU", "threads": 1,
                 "native_forward_median_ms": forward_timing(wrapper, features), "forward_warmup": 5, "forward_repeats": 20,
                 "onnxruntime_single_query_median_ms": {"dynamic": ort_timing(session, features[:1]), "static": ort_timing(single_session, features[:1]), "warmup": 5, "repeats": 50},
                 "choose_action": None if args.skip_search_benchmark else {"implementation": "training.search.BatchedMCTS (current service path)", "simulations": 128, "budget_ms": 30_000, "samples": search_timing(model, states)},
                 "native_forest_choose_action": None if args.skip_search_benchmark else {"implementation": "training.native_search.NativeBatchedMCTS", "simulations": 128, "samples": native_search_timing(model, states)}}
    benchmark_path = args.output_dir / "champion-benchmark.json"; benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    metadata = {"schema": "gipf-browser-onnx-v1", "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
                "served_artifact_sha256": champion_meta.get("served_artifact_sha256"), "source_checkpoint_sha256": champion_meta.get("source_checkpoint_sha256"),
                "architecture": checkpoint_data["config"], "games": checkpoint_data.get("games"), "iteration": checkpoint_data.get("iteration"),
                "onnx": {"path": "champion.onnx", "sha256": sha256(onnx_path), "bytes": onnx_path.stat().st_size, "opset": 17, "dynamic_batch": True,
                         "input_name": "board_features", "output_names": ["policy_logits", "value"]},
                "onnx_single": {"path": "champion-single.onnx", "sha256": sha256(single_path), "bytes": single_path.stat().st_size, "opset": 17, "dynamic_batch": False,
                                "input_name": "board_features", "input_shape": [1, 9, 7, 7], "output_names": ["policy_logits", "value"]},
                "onnx_portable": {"path": "champion-portable.onnx", "sha256": sha256(portable_path), "bytes": portable_path.stat().st_size, "opset": 17, "dynamic_batch": False,
                                  "input_name": "board_features", "input_shape": [1, 9, 7, 7], "output_names": ["policy_logits", "value"],
                                  "group_norm_lowering": "ReduceMean/Sub/Mul/Add/Sqrt/Div plus affine; no InstanceNormalization"},
                "model_sha256": sha256(onnx_path),
                "fixtures": {"path": fixture_path.name, "sha256": sha256(fixture_path), "count": len(fixtures), "onnxruntime_cpu_rtol": 2e-5, "onnxruntime_cpu_atol": 2e-6},
                "onnxruntime_cpu_validation": validation,
                "onnxruntime_cpu_single_validation": static_validation,
                "onnxruntime_cpu_portable_validation": portable_validation,
                "benchmark": benchmark_path.name}
    (args.output_dir / "champion-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
