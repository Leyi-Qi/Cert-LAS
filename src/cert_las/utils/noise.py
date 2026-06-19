import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class LayerwiseNoise:
    sigmas: List[float]
    raw_overall_scale: float
    target_overall_scale: float
    eta: float
    l2_column: str
    num_rows: int


def randn_like(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(x)
    try:
        return torch.randn_like(x, generator=generator)
    except TypeError:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def load_layerwise_noise(
    csv_path: str,
    *,
    target_overall_scale: float,
    eta: float = 1.0,
    l2_column: str = "Avg_L2_Norm_2000",
    num_params_column: str = "Num_Params",
    clip: float = 1e-3,
) -> LayerwiseNoise:
    sigmas: List[float] = []
    dims: List[int] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        if l2_column not in headers or num_params_column not in headers:
            raise ValueError(f"CSV must contain {l2_column!r} and {num_params_column!r}: {csv_path}")
        l2_idx = headers.index(l2_column)
        n_idx = headers.index(num_params_column)
        for row in reader:
            if not row or row[0] == "Total":
                break
            try:
                sigma = min(float(row[l2_idx]), float(clip))
                dim = int(row[n_idx])
            except Exception:
                continue
            sigmas.append(sigma)
            dims.append(dim)
    if not sigmas:
        raise ValueError(f"No valid layerwise rows found in {csv_path}")
    total_dim = float(sum(dims))
    weighted = sum(float(s) * float(s) * float(d) for s, d in zip(sigmas, dims))
    raw_scale = (weighted / total_dim) ** 0.5
    if raw_scale <= 0:
        raise ValueError("Layerwise noise CSV produced a non-positive overall scale")
    scaling = float(target_overall_scale) / raw_scale
    adjusted = [float(eta) * s * scaling for s in sigmas]
    return LayerwiseNoise(
        sigmas=adjusted,
        raw_overall_scale=float(raw_scale),
        target_overall_scale=float(target_overall_scale),
        eta=float(eta),
        l2_column=l2_column,
        num_rows=len(adjusted),
    )


@torch.no_grad()
def add_shared_param_noise_and_backup(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    sigmas: List[float],
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    n_a = sum(1 for _ in model_a.parameters())
    n_b = sum(1 for _ in model_b.parameters())
    if n_a != n_b:
        raise ValueError(f"Model parameter counts differ: {n_a} vs {n_b}")
    device = next(model_a.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    backup_a = [p.detach().cpu().clone() for p in model_a.parameters()]
    backup_b = [p.detach().cpu().clone() for p in model_b.parameters()]
    for idx, (pa, pb) in enumerate(zip(model_a.parameters(), model_b.parameters())):
        if idx >= len(sigmas):
            break
        if not pa.dtype.is_floating_point or not pb.dtype.is_floating_point:
            continue
        eps = randn_like(pa, generator=generator)
        pa.add_(eps * float(sigmas[idx]))
        pb.add_(eps.to(device=pb.device, dtype=pb.dtype) * float(sigmas[idx]))
    return backup_a, backup_b


@torch.no_grad()
def add_param_noise_and_backup(
    model: torch.nn.Module,
    sigmas: List[float],
    seed: int,
) -> List[torch.Tensor]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    backup = [p.detach().cpu().clone() for p in model.parameters()]
    for idx, param in enumerate(model.parameters()):
        if idx >= len(sigmas):
            break
        if param.dtype.is_floating_point:
            param.add_(randn_like(param, generator=generator) * float(sigmas[idx]))
    return backup


@torch.no_grad()
def restore_params(model: torch.nn.Module, backup_cpu: List[torch.Tensor]) -> None:
    for param, backup in zip(model.parameters(), backup_cpu):
        param.data.copy_(backup.to(device=param.device, dtype=param.dtype))
