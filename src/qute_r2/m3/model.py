from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from qute_r2.profiling.corpus import Workload

MAX_QUBITS = 24
FEATURES = 17


def graph_tensors(workloads: list[Workload], max_qubits: int = MAX_QUBITS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(workloads)
    features = np.zeros((batch, max_qubits, FEATURES), dtype=np.float32)
    adjacency = np.zeros((batch, max_qubits, max_qubits), dtype=np.float32)
    mask = np.zeros((batch, max_qubits), dtype=np.float32)
    for row, workload in enumerate(workloads):
        n = workload.n_qubits
        mask[row, :n] = 1
        for a, b in workload.edges:
            adjacency[row, a, b] = adjacency[row, b, a] = 1
        degrees = adjacency[row].sum(axis=1)
        angles: list[float] = []
        for values in (workload.gammas, workload.betas):
            padded = list(values) + [0.0] * (3 - len(values))
            angles.extend(math.sin(value) for value in padded)
            angles.extend(math.cos(value) for value in padded)
        family = float(workload.graph_family == "random_3_regular")
        for qubit in range(n):
            features[row, qubit] = np.asarray(
                [degrees[qubit] / max(1, n - 1), qubit / max(1, n - 1), n / max_qubits, workload.p / 3, family, *angles],
                dtype=np.float32,
            )
    return torch.from_numpy(features), torch.from_numpy(adjacency), torch.from_numpy(mask)


class GraphAutoregressiveSampler(nn.Module):
    def __init__(self, hidden: int = 48, message_passing_layers: int = 2, pair_mode: str = "dot") -> None:
        super().__init__()
        self.hidden = hidden
        if pair_mode not in {"dot", "mlp", "context", "shell", "gru", "hybrid"}:
            raise ValueError("pair_mode must be dot, mlp, context, shell, gru, or hybrid")
        self.message_passing_layers = message_passing_layers
        self.pair_mode = pair_mode
        self.input = nn.Linear(FEATURES, hidden)
        self.self_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(message_passing_layers))
        self.neighbor_layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(message_passing_layers))
        self.base = nn.Linear(hidden, 1)
        if pair_mode == "dot":
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
        elif pair_mode == "mlp":
            self.pair_mlp = nn.Sequential(nn.Linear(3 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        elif pair_mode in {"context", "hybrid"}:
            self.context_decoder = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            if pair_mode == "hybrid":
                self.pair_mlp = nn.Sequential(nn.Linear(3 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        elif pair_mode == "shell":
            self.context_decoder = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        else:
            self.gru = nn.GRUCell(hidden + 1, hidden)
            self.gru_output = nn.Linear(hidden, 1)
        self.adjacency_weight = nn.Parameter(torch.tensor(0.0))

    def encode_graph(self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.input(features))
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1)
        for own, neighbor in zip(self.self_layers, self.neighbor_layers):
            h = torch.nn.functional.silu(own(h) + neighbor(adjacency @ h / degree))
            h = h * mask.unsqueeze(-1)
        return h

    def parameters_for_graph(self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_graph(features, adjacency, mask)
        base = self.base(h).squeeze(-1) * mask
        if self.pair_mode == "dot":
            pair = self.query(h) @ self.key(h).transpose(-1, -2) / math.sqrt(self.hidden)
        elif self.pair_mode in {"mlp", "hybrid"}:
            n = h.shape[1]
            left = h.unsqueeze(2).expand(-1, -1, n, -1)
            right = h.unsqueeze(1).expand(-1, n, -1, -1)
            pair = self.pair_mlp(torch.cat((left, right, left * right, adjacency.unsqueeze(-1)), dim=-1)).squeeze(-1)
        else:
            pair = torch.zeros_like(adjacency)
        pair = pair + self.adjacency_weight * adjacency
        prefix_mask = torch.tril(torch.ones(pair.shape[-2:], device=pair.device), diagonal=-1)
        pair = pair * prefix_mask * mask.unsqueeze(1) * mask.unsqueeze(2)
        return base, pair

    @staticmethod
    def distance_shells(adjacency: torch.Tensor) -> list[torch.Tensor]:
        identity = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=torch.bool).expand(adjacency.shape[0], -1, -1)
        seen = identity.clone(); power = adjacency.bool(); shells = []
        for _ in range(3):
            shell = power & ~seen; shells.append(shell.to(adjacency.dtype)); seen |= power
            power = (power.to(adjacency.dtype) @ adjacency) > 0
        return shells

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        if self.pair_mode == "gru":
            h = self.encode_graph(features, adjacency, mask)
            batch, samples = bits.shape[:2]
            state = ((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).unsqueeze(1).expand(-1, samples, -1).reshape(batch * samples, -1)
            previous = torch.zeros((batch, samples, 1), device=bits.device)
            logits = []
            for position in range(h.shape[1]):
                node = h[:, position].unsqueeze(1).expand(-1, samples, -1)
                state = self.gru(torch.cat((node, previous), dim=-1).reshape(batch * samples, -1), state)
                logits.append(self.gru_output(state).reshape(batch, samples))
                previous = bits[:, :, position:position + 1]
            return torch.stack(logits, dim=-1) * mask[:, None, :]
        if self.pair_mode not in {"context", "shell", "hybrid"}:
            base, pair = self.parameters_for_graph(features, adjacency, mask)
            signed = bits * 2 - 1
            return base[:, None, :] + torch.einsum("bij,bsj->bsi", pair, signed)
        h = self.encode_graph(features, adjacency, mask)
        signed_h = (bits * 2 - 1).unsqueeze(-1) * h[:, None, :, :]
        prefix = signed_h.cumsum(dim=2) - signed_h
        positions = torch.arange(h.shape[1], device=h.device).clamp_min(1).sqrt().view(1, 1, -1, 1)
        prefix = prefix / positions
        target = h[:, None, :, :].expand(-1, bits.shape[1], -1, -1)
        shell_masks = [adjacency] if self.pair_mode in {"context", "hybrid"} else self.distance_shells(adjacency)
        contexts = []
        for shell in shell_masks:
            lower_shell = torch.tril(shell, diagonal=-1)
            context = torch.einsum("bij,bsjh->bsih", lower_shell, signed_h)
            contexts.append(context / lower_shell.sum(-1).clamp_min(1).sqrt()[:, None, :, None])
        logits = self.context_decoder(torch.cat((target, prefix, *contexts), dim=-1)).squeeze(-1)
        if self.pair_mode == "hybrid":
            _, pair = self.parameters_for_graph(features, adjacency, mask)
            logits = logits + torch.einsum("bij,bsj->bsi", pair, bits * 2 - 1)
        return logits * mask[:, None, :]


class M3Backend:
    def __init__(self, model: GraphAutoregressiveSampler, model_hash: str, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.model_hash = model_hash
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cpu") -> "M3Backend":
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=True)
        model = GraphAutoregressiveSampler(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        return cls(model, hashlib.sha256(path.read_bytes()).hexdigest(), device)

    def sample(self, workload: Workload, shots: int, seed: int = 32026) -> tuple[dict[str, int], dict[str, Any]]:
        if not 1 <= workload.n_qubits <= MAX_QUBITS:
            raise ValueError(f"supported widths are 1..{MAX_QUBITS}")
        if shots <= 0:
            raise ValueError("shots must be positive")
        start = time.perf_counter_ns()
        features, adjacency, mask = (value.to(self.device) for value in graph_tensors([workload]))
        encode_ms = (time.perf_counter_ns() - start) / 1e6
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            start = time.perf_counter_ns()
            if self.model.pair_mode in {"context", "shell", "gru", "hybrid"}:
                h = self.model.encode_graph(features, adjacency, mask)
                if self.model.pair_mode in {"context", "shell", "hybrid"}:
                    shell_masks = [adjacency] if self.model.pair_mode in {"context", "hybrid"} else self.model.distance_shells(adjacency)
                    if self.model.pair_mode == "hybrid":
                        _, pair = self.model.parameters_for_graph(features, adjacency, mask)
            else:
                base, pair = self.model.parameters_for_graph(features, adjacency, mask)
            encoder_ms = (time.perf_counter_ns() - start) / 1e6
            bits = torch.zeros((shots, workload.n_qubits), device=self.device)
            prefix_sum = torch.zeros((shots, self.model.hidden), device=self.device) if self.model.pair_mode in {"context", "shell", "hybrid"} else None
            if self.model.pair_mode == "gru":
                state = ((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).expand(shots, -1).contiguous()
                previous = torch.zeros((shots, 1), device=self.device)
            start = time.perf_counter_ns()
            for position in range(workload.n_qubits):
                if self.model.pair_mode == "gru":
                    node = h[0, position].expand(shots, -1)
                    state = self.model.gru(torch.cat((node, previous), dim=-1), state)
                    logits = self.model.gru_output(state).squeeze(-1)
                elif self.model.pair_mode in {"context", "shell", "hybrid"}:
                    target = h[0, position].expand(shots, -1)
                    prefix = prefix_sum / math.sqrt(position) if position else torch.zeros_like(target)
                    contexts = []
                    for shell in shell_masks:
                        indices = torch.nonzero(shell[0, position, :position], as_tuple=False).flatten()
                        if len(indices):
                            signs = bits[:, indices] * 2 - 1
                            contexts.append((signs.unsqueeze(-1) * h[0, indices]).sum(1) / math.sqrt(len(indices)))
                        else:
                            contexts.append(torch.zeros_like(target))
                    logits = self.model.context_decoder(torch.cat((target, prefix, *contexts), dim=-1)).squeeze(-1)
                    if self.model.pair_mode == "hybrid" and position:
                        logits = logits + (bits[:, :position] * 2 - 1) @ pair[0, position, :position]
                else:
                    logits = base[0, position]
                    if position:
                        logits = logits + (bits[:, :position] * 2 - 1) @ pair[0, position, :position]
                probability = torch.sigmoid(logits)
                bits[:, position] = torch.rand(shots, generator=generator, device=self.device) < probability
                if self.model.pair_mode in {"context", "shell", "hybrid"}:
                    prefix_sum += (bits[:, position] * 2 - 1).unsqueeze(-1) * h[0, position]
                elif self.model.pair_mode == "gru":
                    previous = bits[:, position:position + 1]
            sample_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        rows = bits.to("cpu", dtype=torch.uint8).numpy()
        counts = dict(sorted(Counter("".join(map(str, row[::-1])) for row in rows).items()))
        aggregate_ms = (time.perf_counter_ns() - start) / 1e6
        metadata = {
            "execution_route": "qute_m3",
            "model_hash": self.model_hash,
            "circuit_hash": workload.circuit_hash,
            "shots": shots,
            "per_circuit_optimizer_steps": 0,
            "exact_simulator_calls_on_neural_route": 0,
            "explicit_2n_inference_output": False,
            "timing_ms": {"encode": encode_ms, "graph_encoder": encoder_ms, "autoregressive_sample": sample_ms,
                          "counts_aggregation": aggregate_ms, "total": encode_ms + encoder_ms + sample_ms + aggregate_ms},
        }
        return counts, metadata


def save_checkpoint(model: GraphAutoregressiveSampler, path: str | Path) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"hidden": model.hidden, "message_passing_layers": model.message_passing_layers, "pair_mode": model.pair_mode}, "state_dict": model.state_dict()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
