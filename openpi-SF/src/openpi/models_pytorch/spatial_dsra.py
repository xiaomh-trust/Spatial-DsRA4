"""Spatial-DsRA v4 shared-basis rank-wise full-layer spatial distillation."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialDsRAProjector(nn.Module):
    # [Spatial-DsRA v4]: d_g is the actual teacher output width, not its internal embedding width.
    def __init__(self, llm_dim, vggt_dim, num_layers, bottleneck_dim=256, rank=32,
                 temperature=1.0, allocation_floor=0.0):
        super().__init__()
        if min(llm_dim, vggt_dim, num_layers, bottleneck_dim, rank) <= 0 or rank > bottleneck_dim:
            raise ValueError("Positive dimensions and rank <= bottleneck_dim are required")
        if temperature <= 0 or not 0 <= allocation_floor < 1:
            raise ValueError("Require temperature > 0 and 0 <= allocation_floor < 1")
        self.llm_dim, self.vggt_dim, self.num_layers = llm_dim, vggt_dim, num_layers
        self.temperature, self.allocation_floor = temperature, allocation_floor
        # [Spatial-DsRA v4]: Shared LN/down/up; only rho has a layer axis.
        self.norm = nn.LayerNorm(llm_dim)
        self.down = nn.Linear(llm_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, vggt_dim)
        self.basis = nn.Parameter(torch.empty(bottleneck_dim, rank))
        self.rho = nn.Parameter(torch.zeros(num_layers, rank))
        nn.init.orthogonal_(self.basis)

    def allocation(self):
        # [Spatial-DsRA v4]: Normalize over L, independently for every rank direction.
        return ((1 - self.allocation_floor) * (self.rho.float() / self.temperature).softmax(dim=0)
                + self.allocation_floor / self.num_layers)

    def set_stage(self, stage):
        # [Spatial-DsRA v4]: Stage III freezes parameters but preserves gradients to H_l.
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2 or 3")
        for parameter in self.parameters():
            parameter.requires_grad_(stage != 3)
        self.basis.requires_grad_(stage == 2)
        self.rho.requires_grad_(stage == 2)

    def fuse(self, hidden_states):
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layer outputs, got {len(hidden_states)}")
        expected = hidden_states[0].shape
        if len(expected) != 3 or expected[-1] != self.llm_dim:
            raise ValueError(f"Expected [B,N,{self.llm_dim}], got {expected}")
        coefficients = self.allocation() - 1.0 / self.num_layers
        fused = None
        # [Spatial-DsRA v4]: mean(B_l) + sum((B_l M * (pi_l - 1/L)) M^T).
        # Stream over layers to avoid stacking L x B x N x d_v.
        for layer, hidden in enumerate(hidden_states):
            if hidden.shape != expected:
                raise ValueError(f"Layer {layer}: {hidden.shape} != {expected}")
            bottleneck = F.gelu(self.down(self.norm(hidden.to(self.down.weight.dtype))))
            basis = self.basis.to(bottleneck.dtype)
            correction = ((bottleneck @ basis) * coefficients[layer].to(bottleneck.dtype)) @ basis.T
            contribution = bottleneck / self.num_layers + correction
            fused = contribution if fused is None else fused + contribution
        return fused

    def orthogonality_loss(self):
        # [Spatial-DsRA v4]: FP32 squared Frobenius norm, without division by r or L.
        with torch.autocast(device_type=self.basis.device.type, enabled=False):
            basis = self.basis.float()
            identity = torch.eye(basis.shape[1], device=basis.device)
            return (basis.T @ basis - identity).square().sum()

    def forward(self, hidden_states, target, valid_mask=None):
        # [Spatial-DsRA v4]: Exactly one up-projection and one masked cosine loss.
        prediction = self.up(self.fuse(hidden_states))
        if prediction.shape != target.shape:
            raise ValueError(f"Fused/teacher mismatch: {prediction.shape} vs {target.shape}")
        loss = 1 - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1)
        if valid_mask is None:
            alignment = loss.mean()
        else:
            if valid_mask.shape != loss.shape or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be bool [B,N] matching visual tokens")
            alignment = loss.masked_fill(~valid_mask, 0).sum() / valid_mask.sum().clamp_min(1)
        return alignment, self.orthogonality_loss()


def spatial_stage(step, warmup_steps, absorption_start):
    # [Spatial-DsRA v4]: Boundaries are explicit experiment choices, not paper-specified defaults.
    if warmup_steps <= 0 or absorption_start <= warmup_steps:
        raise ValueError("Require 0 < spatial_warmup_steps < spatial_absorption_start")
    return 1 if step < warmup_steps else (2 if step < absorption_start else 3)
