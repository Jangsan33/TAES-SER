from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn




@dataclass
class TAESSEROutput:
    loss: Optional[torch.Tensor] = None
    loss_ser: Optional[torch.Tensor] = None
    loss_asr: Optional[torch.Tensor] = None
    loss_sr: Optional[torch.Tensor] = None
    loss_mi: Optional[torch.Tensor] = None
    loss_entropy: Optional[torch.Tensor] = None

    logits_ser: Optional[torch.Tensor] = None
    logits_asr: Optional[torch.Tensor] = None
    logits_sr: Optional[torch.Tensor] = None

    routing_stats: Optional[Dict[str, Any]] = None
    routing_choices: Optional[Dict[str, Any]] = None




def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.float().unsqueeze(-1)
    num = (x * mask).sum(dim=1)
    den = mask.sum(dim=1).clamp_min(1.0)
    return num / den


def batch_jaccard(topk_a: torch.Tensor, topk_b: torch.Tensor) -> torch.Tensor:
    match = (topk_a[:, :, None] == topk_b[:, None, :])
    inter = match.any(dim=-1).float().sum(dim=-1)
    union = (2.0 * topk_a.size(1) - inter).clamp_min(1.0)
    return (inter / union).mean()


def expert_coverage(topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    used = torch.unique(topk_idx.reshape(-1)).numel()
    return torch.tensor(float(used) / float(max(1, num_experts)), device=topk_idx.device)


def mutual_information_from_task_distributions(p_e_given_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p_e_given_t: [num_tasks, num_experts]
    """
    p = p_e_given_t.float()
    p = p + eps
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    p_e = p.mean(dim=0)
    p_e = p_e + eps
    p_e = p_e / p_e.sum().clamp_min(eps)

    mi = (p * (torch.log(p) - torch.log(p_e.unsqueeze(0)))).sum(dim=-1).mean()
    return torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)




class AdapterExpert(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int, dropout: float):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.drop(self.act(self.down(x))))


class TaskRouter(nn.Module):
    def __init__(self, hidden_size: int, router_hidden: int, num_experts: int, router_dropout: float = 0.1):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_size, router_hidden),
            nn.GELU(),
            nn.Dropout(router_dropout),
            nn.Linear(router_hidden, num_experts),
        )
        self.noise = nn.Sequential(
            nn.Linear(hidden_size, router_hidden),
            nn.GELU(),
            nn.Dropout(router_dropout),
            nn.Linear(router_hidden, num_experts),
        )

    def forward(
        self,
        pooled: torch.Tensor,
        top_k: int,
        temperature: float = 1.0,
        noisy_routing: bool = True,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(pooled)

        if noisy_routing and training:
            noise_std = F.softplus(self.noise(pooled)) + 1e-6
            logits = logits + torch.randn_like(logits) * noise_std

        logits = logits / max(temperature, 1e-4)
        top_k = min(max(1, top_k), logits.size(-1))

        topk_val, topk_idx = torch.topk(logits, k=top_k, dim=-1)
        topk_gates = F.softmax(topk_val, dim=-1)

        gates = torch.zeros_like(logits)
        gates.scatter_(dim=-1, index=topk_idx, src=topk_gates)

        entropy = -(topk_gates.clamp_min(1e-8) * torch.log(topk_gates.clamp_min(1e-8))).sum(dim=-1).mean()
        return gates, topk_idx, entropy, logits




class TAESSERModel(nn.Module):


    def __init__(
        self,
        acoustic_encoder: nn.Module,
        hidden_size: int,
        vocab_size: int,
        num_emotions: int,
        num_speakers: int,
        num_experts: int = 6,
        expert_bottleneck: int = 256,
        expert_dropout: float = 0.1,
        router_hidden: int = 256,
        router_dropout: float = 0.1,
        top_k: int = 3,
        router_temperature: float = 2.0,
        noisy_routing: bool = True,
        alpha_asr: float = 0.1,
        beta_sr: float = 0.8,
        mi_coef: float = 0.002,
        ent_coef: float = 0.05,
        use_masked_pooling: bool = True,
    ):
        super().__init__()

        # Withheld in the public release: exact backbone choice / checkpoint protocol.
        self.encoder = acoustic_encoder

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.noisy_routing = noisy_routing
        self.use_masked_pooling = use_masked_pooling

        self.alpha_asr = alpha_asr
        self.beta_sr = beta_sr
        self.mi_coef = mi_coef
        self.ent_coef = ent_coef
        self.task_embed = nn.Embedding(3, hidden_size)
        self.experts = nn.ModuleList(
            [AdapterExpert(hidden_size, expert_bottleneck, expert_dropout) for _ in range(num_experts)]
        )

        self.router_asr = TaskRouter(hidden_size, router_hidden, num_experts, router_dropout)
        self.router_ser = TaskRouter(hidden_size, router_hidden, num_experts, router_dropout)
        self.router_sr = TaskRouter(hidden_size, router_hidden, num_experts, router_dropout)

        self.head_asr = nn.Linear(hidden_size, vocab_size)
        self.head_ser = nn.Linear(hidden_size, num_emotions)
        self.head_sr = nn.Linear(hidden_size, num_speakers)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_masked_pooling:
            return hidden_states.mean(dim=1)
        return masked_mean(hidden_states, attention_mask)

    def _apply_task_experts(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        router: TaskRouter,
        task_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self._pool(hidden_states, attention_mask)
        pooled = pooled + self.task_embed.weight[task_id].unsqueeze(0)

        gates, topk_idx, entropy, router_logits = router(
            pooled=pooled,
            top_k=self.top_k,
            temperature=self.router_temperature,
            noisy_routing=self.noisy_routing,
            training=self.training,
        )

        task_conditioned = hidden_states + self.task_embed.weight[task_id].view(1, 1, -1)
        expert_outputs = torch.stack([expert(task_conditioned) for expert in self.experts], dim=2)
        mixed = (expert_outputs * gates[:, None, :, None]).sum(dim=2)

        return hidden_states + mixed, gates, topk_idx, entropy, router_logits

    def _ctc_loss(
        self,
        logits_asr: torch.Tensor,
        labels_asr: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor],
        target_lengths: Optional[torch.Tensor],
        blank_id: int = 0,
    ) -> torch.Tensor:
        if labels_asr is None:
            return torch.tensor(0.0, device=logits_asr.device)

        # Public release keeps the loss form but not the full label-preparation protocol.
        if input_lengths is None or target_lengths is None:
            raise NotImplementedError(
                "ASR target formatting and length construction are intentionally omitted in the public release."
            )

        log_probs = F.log_softmax(logits_asr, dim=-1).transpose(0, 1)
        return F.ctc_loss(
            log_probs,
            labels_asr,
            input_lengths,
            target_lengths,
            blank=blank_id,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_ser: Optional[torch.Tensor] = None,
        labels_asr: Optional[torch.Tensor] = None,
        labels_sr: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        return_stats: bool = True,
    ) -> TAESSEROutput:
        # The public release assumes the external encoder returns hidden states [B, T, H].
        encoded = self.encoder(input_values, attention_mask=attention_mask)
        if isinstance(encoded, dict):
            hidden_states = encoded["last_hidden_state"]
        else:
            hidden_states = encoded

        # 1) Task-aware sparse routing over a shared expert pool
        asr_states, gates_asr, topk_asr, ent_asr, logits_router_asr = self._apply_task_experts(
            hidden_states, attention_mask, self.router_asr, task_id=0
        )
        ser_states, gates_ser, topk_ser, ent_ser, logits_router_ser = self._apply_task_experts(
            hidden_states, attention_mask, self.router_ser, task_id=1
        )
        sr_states, gates_sr, topk_sr, ent_sr, logits_router_sr = self._apply_task_experts(
            hidden_states, attention_mask, self.router_sr, task_id=2
        )

        # 2) Task heads
        logits_asr = self.head_asr(asr_states)
        logits_ser = self.head_ser(self._pool(ser_states, attention_mask))
        logits_sr = self.head_sr(self._pool(sr_states, attention_mask))

        # 3) Primary / auxiliary task losses
        loss_ser = torch.tensor(0.0, device=hidden_states.device)
        loss_asr = torch.tensor(0.0, device=hidden_states.device)
        loss_sr = torch.tensor(0.0, device=hidden_states.device)

        if labels_ser is not None:
            loss_ser = F.cross_entropy(logits_ser, labels_ser)

        if labels_sr is not None:
            loss_sr = F.cross_entropy(logits_sr, labels_sr)

        if labels_asr is not None:
            loss_asr = self._ctc_loss(
                logits_asr=logits_asr,
                labels_asr=labels_asr,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank_id=0,
            )

        task_loss = loss_ser + self.alpha_asr * loss_asr + self.beta_sr * loss_sr

        # 4) Routing regularization
        p_e_given_t = torch.stack(
            [
                F.softmax(logits_router_asr, dim=-1).mean(dim=0),
                F.softmax(logits_router_ser, dim=-1).mean(dim=0),
                F.softmax(logits_router_sr, dim=-1).mean(dim=0),
            ],
            dim=0,
        )

        loss_mi = mutual_information_from_task_distributions(p_e_given_t)

        loss_entropy = ent_asr + ent_ser + ent_sr

        loss_balance = (
            switch_balance_loss(gates_asr, topk_asr, self.num_experts)
            + switch_balance_loss(gates_ser, topk_ser, self.num_experts)
            + switch_balance_loss(gates_sr, topk_sr, self.num_experts)
        )

        # 5) Final objective
        #
        # SER is treated as the main task.
        # ASR and SR act as auxiliary tasks.
        # Routing regularizers encourage specialization while improving routing stability.
        total_loss = (
            task_loss
            - self.mi_coef * loss_mi
            - self.ent_coef * loss_entropy
        )

        routing_stats = None
        routing_choices = None

        if return_stats:
            routing_stats = {
                "coverage_asr": float(expert_coverage(topk_asr, self.num_experts).item()),
                "coverage_ser": float(expert_coverage(topk_ser, self.num_experts).item()),
                "coverage_sr": float(expert_coverage(topk_sr, self.num_experts).item()),
                "jaccard_asr_ser": float(batch_jaccard(topk_asr, topk_ser).item()),
                "jaccard_asr_sr": float(batch_jaccard(topk_asr, topk_sr).item()),
                "jaccard_ser_sr": float(batch_jaccard(topk_ser, topk_sr).item()),
                "entropy_asr": float(ent_asr.detach().item()),
                "entropy_ser": float(ent_ser.detach().item()),
                "entropy_sr": float(ent_sr.detach().item()),
            }

            routing_choices = {
                "asr": {"topk_idx": topk_asr.detach().cpu(), "gates": gates_asr.detach().cpu()},
                "ser": {"topk_idx": topk_ser.detach().cpu(), "gates": gates_ser.detach().cpu()},
                "sr": {"topk_idx": topk_sr.detach().cpu(), "gates": gates_sr.detach().cpu()},
            }

        return TAESSEROutput(
            loss=total_loss,
            loss_ser=loss_ser,
            loss_asr=loss_asr,
            loss_sr=loss_sr,
            loss_mi=loss_mi,
            loss_entropy=loss_entropy,
            logits_ser=logits_ser,
            logits_asr=logits_asr,
            logits_sr=logits_sr,
            routing_stats=routing_stats,
            routing_choices=routing_choices,
        )
