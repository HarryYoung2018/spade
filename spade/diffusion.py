"""Conditional diffusion surrogate for scalar y | x."""

from __future__ import annotations

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.lin1 = nn.Linear(self.dim, self.dim)
        self.lin2 = nn.Linear(self.dim, self.dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=t.device))
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = F.silu(self.lin1(emb))
        return self.lin2(emb)


class CondEpsNet(nn.Module):
    def __init__(self, x_dim: int, hidden: int, t_dim: int):
        super().__init__()
        self.t_embed = TimeEmbedding(t_dim)
        self.x_enc = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.y_in = nn.Linear(1, hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden + hidden + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, y_t: torch.Tensor, t_cont: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t_cont.view(-1))
        xh = self.x_enc(x)
        yh = self.y_in(y_t)
        h = torch.cat([xh, yh, t_emb], dim=-1)
        return self.fc(h)


def make_beta_schedule(steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(float(beta_start), float(beta_end), int(steps))


class ScalarDiffusionSurrogate(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        steps: int,
        hidden: int,
        t_dim: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.steps = int(steps)
        self.eps_net = CondEpsNet(int(x_dim), hidden=int(hidden), t_dim=int(t_dim)).to(device)

        betas = make_beta_schedule(self.steps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumsum(torch.log(alphas), dim=0).exp()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(self, y0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t_idx].view(-1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t_idx].view(-1, 1)
        return a * y0 + b * noise

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = int(x.shape[0])
        t_idx = torch.randint(0, self.steps, (batch,), device=x.device, dtype=torch.long)
        noise = torch.randn_like(y)
        y_t = self.q_sample(y, t_idx, noise)
        t_cont = (t_idx.float() + 0.5) / float(self.steps)
        pred_eps = self.eps_net(y_t, t_cont, x)
        return F.mse_loss(pred_eps, noise)

    def ddim_sample(self, x: torch.Tensor, *, steps: int, eta: float) -> torch.Tensor:
        count = int(x.shape[0])
        device = x.device
        idxs = torch.linspace(self.steps - 1, 0, int(steps), device=device).long()
        y_t = torch.randn(count, 1, device=device)

        for i in range(len(idxs) - 1):
            t_cur = idxs[i]
            t_prev = idxs[i + 1]
            t_cont = (t_cur.float() + 0.5) / float(self.steps)
            eps = self.eps_net(y_t, t_cont.expand(count), x)

            alpha_t = self.alphas[t_cur]
            alpha_bar_t = self.alphas_cumprod[t_cur]
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_one_ab = torch.sqrt(1.0 - alpha_bar_t)
            y0_pred = (y_t - sqrt_one_ab * eps) / (sqrt_ab + 1e-8)

            alpha_bar_prev = self.alphas_cumprod[t_prev]
            if eta > 0:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_t)
                )
            else:
                sigma_t = torch.tensor(0.0, device=device)

            y_t = (
                torch.sqrt(alpha_bar_prev) * y0_pred
                + torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * eps
                + sigma_t * torch.randn_like(y_t)
            )

        t_last = idxs[-1]
        t_cont = (t_last.float() + 0.5) / float(self.steps)
        eps = self.eps_net(y_t, t_cont.expand(count), x)
        alpha_bar_t = self.alphas_cumprod[t_last]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_one_ab = torch.sqrt(1.0 - alpha_bar_t)
        y0_pred = (y_t - sqrt_one_ab * eps) / (sqrt_ab + 1e-8)
        return y0_pred


def mc_samples(
    model: ScalarDiffusionSurrogate,
    x: torch.Tensor,
    *,
    samples: int,
    steps: int,
    eta: float,
    batch_size: int,
    requires_grad: bool,
) -> torch.Tensor:
    model.eval()
    count = int(x.shape[0])
    outputs = []
    per = max(1, min(int(samples), 16))
    rounds = math.ceil(int(samples) / per)
    ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()

    with ctx:
        for _ in range(rounds):
            batch_samples = []
            for i in range(0, count, int(batch_size)):
                xb = x[i : i + int(batch_size)]
                xb_rep = xb.repeat(per, 1)
                y_s = model.ddim_sample(xb_rep, steps=steps, eta=eta)
                y_s = y_s.view(per, -1)
                batch_samples.append(y_s)
            samp = torch.cat(batch_samples, dim=1)
            outputs.append(samp)

    all_samp = torch.cat(outputs, dim=0)
    return all_samp[: int(samples)]
