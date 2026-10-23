#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


AUX_DIM = 27
AUX_PROFILE = slice(0, 9)
AUX_DM = slice(9, 17)
AUX_SUBINTS = slice(17, 21)
AUX_SUBBANDS = slice(21, 25)
AUX_CROSS = slice(25, 27)
PRIOR_GATE_STRENGTH = 0.2


def split_channels(out_ch: int, parts: int):
    base = int(out_ch) // int(parts)
    chans = [base] * int(parts)
    chans[-1] += int(out_ch) - sum(chans)
    return chans


class MultiScaleConv1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c3, c7, c15 = split_channels(out_ch, 3)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_ch, c3, kernel_size=3, padding=1, bias=False),
                nn.Conv1d(in_ch, c7, kernel_size=7, padding=3, bias=False),
                nn.Conv1d(in_ch, c15, kernel_size=15, padding=7, bias=False),
            ]
        )
        self.net = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor, branch_gate=None) -> torch.Tensor:
        outs = []
        for i, branch in enumerate(self.branches):
            y = branch(x)
            if branch_gate is not None:
                y = y * branch_gate[:, i].reshape(-1, 1, 1)
            outs.append(y)
        return self.net(torch.cat(outs, dim=1))


class PriorGuided1DEncoder(nn.Module):
    def __init__(self, out_dim: int = 96, prior_dim: int = 0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MultiScaleConv1DBlock(1, 32),
                MultiScaleConv1DBlock(32, 64),
                MultiScaleConv1DBlock(64, 96),
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flat = nn.Flatten()
        self.proj = nn.Linear(96, int(out_dim))
        self.prior_dim = int(prior_dim)
        self.prior_gate = nn.Linear(self.prior_dim, len(self.blocks) * 3) if self.prior_dim > 0 else None
        if self.prior_gate is not None:
            nn.init.zeros_(self.prior_gate.weight)
            nn.init.zeros_(self.prior_gate.bias)

    def gates(self, prior, batch_size: int, device, dtype):
        if self.prior_gate is None or prior is None:
            return [None] * len(self.blocks)
        raw = self.prior_gate(prior.float()).reshape(batch_size, len(self.blocks), 3)
        return [
            1.0 + PRIOR_GATE_STRENGTH * torch.tanh(raw[:, i, :]).to(device=device, dtype=dtype)
            for i in range(len(self.blocks))
        ]

    def forward(self, x: torch.Tensor, prior=None) -> torch.Tensor:
        gates = self.gates(prior, x.size(0), x.device, x.dtype)
        h = x
        for block, gate in zip(self.blocks, gates):
            h = block(h, gate)
        return self.proj(self.flat(self.pool(h)))


class MultiScaleConv2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        self.pool = bool(pool)
        c3, c5, cv, ch = split_channels(out_ch, 4)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_ch, c3, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(in_ch, c5, kernel_size=5, padding=2, bias=False),
                nn.Conv2d(in_ch, cv, kernel_size=(7, 1), padding=(3, 0), bias=False),
                nn.Conv2d(in_ch, ch, kernel_size=(1, 7), padding=(0, 3), bias=False),
            ]
        )
        self.net = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool_layer = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, branch_gate=None) -> torch.Tensor:
        outs = []
        for i, branch in enumerate(self.branches):
            y = branch(x)
            if branch_gate is not None:
                y = y * branch_gate[:, i].reshape(-1, 1, 1, 1)
            outs.append(y)
        h = self.net(torch.cat(outs, dim=1))
        return self.pool_layer(h) if self.pool else h


class PriorGuided2DEncoder(nn.Module):
    def __init__(self, out_dim: int = 128, prior_dim: int = 0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MultiScaleConv2DBlock(2, 32, pool=True),
                MultiScaleConv2DBlock(32, 64, pool=True),
                MultiScaleConv2DBlock(64, 128, pool=False),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.proj = nn.Linear(128, int(out_dim))
        self.prior_dim = int(prior_dim)
        self.prior_gate = nn.Linear(self.prior_dim, len(self.blocks) * 4) if self.prior_dim > 0 else None
        if self.prior_gate is not None:
            nn.init.zeros_(self.prior_gate.weight)
            nn.init.zeros_(self.prior_gate.bias)

    def gates(self, prior, batch_size: int, device, dtype):
        if self.prior_gate is None or prior is None:
            return [None] * len(self.blocks)
        raw = self.prior_gate(prior.float()).reshape(batch_size, len(self.blocks), 4)
        return [
            1.0 + PRIOR_GATE_STRENGTH * torch.tanh(raw[:, i, :]).to(device=device, dtype=dtype)
            for i in range(len(self.blocks))
        ]

    def forward(self, x: torch.Tensor, prior=None) -> torch.Tensor:
        gates = self.gates(prior, x.size(0), x.device, x.dtype)
        h = x
        for block, gate in zip(self.blocks, gates):
            h = block(h, gate)
        return self.proj(self.flat(self.pool(h)))


class NGC6517Encoder(nn.Module):
    """Prior-guided multimodal embedding network for pulsar/RFI episodes."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1, use_prior: bool = True) -> None:
        super().__init__()
        self.use_prior = bool(use_prior)
        self.profile_encoder = PriorGuided1DEncoder(96, prior_dim=9 if self.use_prior else 0)
        self.dm_encoder = PriorGuided1DEncoder(96, prior_dim=8 if self.use_prior else 0)
        self.image_encoder = PriorGuided2DEncoder(128, prior_dim=10 if self.use_prior else 0)
        self.meta_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.fuse = nn.Sequential(
            nn.Linear(96 + 96 + 128 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, int(embed_dim)),
        )

    def aux(self, batch):
        aux = batch.get("aux")
        if aux is None or not self.use_prior:
            return None
        aux = aux.float()
        if aux.size(1) < AUX_DIM:
            aux = torch.cat([aux, aux.new_zeros((aux.size(0), AUX_DIM - aux.size(1)))], dim=1)
        return aux[:, :AUX_DIM]

    def modality_priors(self, aux):
        if aux is None:
            return None, None, None
        profile_prior = aux[:, AUX_PROFILE]
        dm_prior = aux[:, AUX_DM]
        image_prior = torch.cat([aux[:, AUX_SUBINTS], aux[:, AUX_SUBBANDS], aux[:, AUX_CROSS]], dim=1)
        return profile_prior, dm_prior, image_prior

    def forward(self, batch) -> torch.Tensor:
        aux = self.aux(batch)
        profile_prior, dm_prior, image_prior = self.modality_priors(aux)
        profile = self.profile_encoder(batch["profile"].float(), profile_prior)
        dm = self.dm_encoder(batch["dm"].float(), dm_prior)
        image = self.image_encoder(batch["image"].float(), image_prior)
        meta = self.meta_encoder(batch["meta"].float())
        z = self.fuse(torch.cat([profile, dm, image, meta], dim=1))
        return F.normalize(z, dim=1)


def prototypical_logits(
    z_support: torch.Tensor,
    y_support: torch.Tensor,
    z_query: torch.Tensor,
    temperature: float = 10.0,
    psr_k: int = 1,
    rfi_k: int = 1,
) -> torch.Tensor:
    class_scores = []
    for cls, k in ((0, rfi_k), (1, psr_k)):
        mask = y_support == cls
        if not torch.any(mask):
            raise ValueError(f"support set does not contain class {cls}")
        z_cls = F.normalize(z_support[mask], dim=1)
        k = max(1, min(int(k), int(z_cls.size(0))))
        if k == 1:
            proto = F.normalize(z_cls.mean(dim=0, keepdim=True), dim=1)
        else:
            proto = deterministic_multi_prototypes(z_cls, k)
        sim = z_query @ proto.t()
        class_scores.append(sim.max(dim=1).values)
    return float(temperature) * torch.stack(class_scores, dim=1)


def deterministic_multi_prototypes(z: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest-first support prototypes; deterministic and training-free."""
    z = F.normalize(z, dim=1)
    if int(k) <= 1 or z.size(0) <= 1:
        return F.normalize(z.mean(dim=0, keepdim=True), dim=1)
    center = z.mean(dim=0, keepdim=True)
    first = int(torch.argmin((z @ center.t()).squeeze(1)).item())
    chosen = [first]
    for _ in range(1, int(k)):
        chosen_z = z[chosen]
        max_sim = z @ chosen_z.t()
        min_dist = 1.0 - max_sim.max(dim=1).values
        min_dist[torch.tensor(chosen, device=z.device)] = -1.0
        chosen.append(int(torch.argmax(min_dist).item()))
    proto = z[chosen]
    return F.normalize(proto, dim=1)
