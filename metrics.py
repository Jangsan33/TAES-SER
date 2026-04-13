from __future__ import annotations

from typing import Dict

import torch


def expert_coverage(topk_idx: torch.Tensor, num_experts: int) -> float:
    """
    Fraction of experts that are selected at least once in the current batch.
    """
    if topk_idx.numel() == 0:
        return 0.0
    used = torch.unique(topk_idx.reshape(-1)).numel()
    return float(used / max(1, num_experts))


def average_expert_entropy(gates: torch.Tensor, eps: float = 1e-8) -> float:
    """
    gates: [B, E], sparse or dense routing distribution over experts
    """
    if gates.numel() == 0:
        return 0.0
    p = gates.clamp_min(eps)
    entropy = -(p * torch.log(p)).sum(dim=-1).mean()
    return float(entropy.item())


def batch_jaccard(topk_a: torch.Tensor, topk_b: torch.Tensor) -> float:
    """
    Average Jaccard overlap between two task-specific top-k expert sets.
    """
    if topk_a.shape != topk_b.shape:
        raise ValueError("topk_a and topk_b must share the same shape.")

    match = topk_a[:, :, None] == topk_b[:, None, :]
    inter = match.any(dim=-1).float().sum(dim=-1)
    union = (2.0 * topk_a.size(1) - inter).clamp_min(1.0)
    return float((inter / union).mean().item())


def topk_unique_count(topk_idx: torch.Tensor) -> float:
    """
    Mean number of unique experts selected per sample.
    """
    if topk_idx.numel() == 0:
        return 0.0

    vals = []
    for i in range(topk_idx.size(0)):
        vals.append(float(torch.unique(topk_idx[i]).numel()))
    return float(sum(vals) / len(vals))


def summarize_task_routing(
    topk_asr: torch.Tensor,
    topk_ser: torch.Tensor,
    topk_sr: torch.Tensor,
    gates_asr: torch.Tensor,
    gates_ser: torch.Tensor,
    gates_sr: torch.Tensor,
    num_experts: int,
) -> Dict[str, float]:
    """
    Routing summary used for reviewer-facing interpretability analysis.
    """
    return {
        "coverage_asr": expert_coverage(topk_asr, num_experts),
        "coverage_ser": expert_coverage(topk_ser, num_experts),
        "coverage_sr": expert_coverage(topk_sr, num_experts),
        "entropy_asr": average_expert_entropy(gates_asr),
        "entropy_ser": average_expert_entropy(gates_ser),
        "entropy_sr": average_expert_entropy(gates_sr),
        "unique_topk_asr": topk_unique_count(topk_asr),
        "unique_topk_ser": topk_unique_count(topk_ser),
        "unique_topk_sr": topk_unique_count(topk_sr),
        "jaccard_asr_ser": batch_jaccard(topk_asr, topk_ser),
        "jaccard_asr_sr": batch_jaccard(topk_asr, topk_sr),
        "jaccard_ser_sr": batch_jaccard(topk_ser, topk_sr),
    }
