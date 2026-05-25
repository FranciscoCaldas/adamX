from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class AdamX(Optimizer):
    """AdamX optimizer with gradient-similarity scaling and optional loss revert."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        alpha: float = 0.99,
        lambda_exp: float = 1.0,
        cc: float = 0.0,
        track_stats: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1).")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            alpha=alpha,
            lambda_exp=lambda_exp,
            cc=cc,
        )
        super().__init__(params, defaults)
        self.track_stats = track_stats
        self.last_step_stats: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        gamma_sum = 0.0
        gamma_min = float("inf")
        gamma_max = -float("inf")
        gamma_count = 0
        update_sum = 0.0
        update_sq_sum = 0.0
        update_numel = 0
        update_what_if_sum = 0.0
        update_what_if_numel = 0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lambda_exp = group["lambda_exp"]
            cc = group["cc"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamX does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.ones_like(p)
                    state["prev_mhat"] = torch.zeros_like(p)
                    state["prev_vhat"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]

                state["step"] += 1
                step = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)

                grad_flat = grad.view(-1)
                m_hat_flat = m_hat.view(-1)
                prev_grad_flat = state["prev_grad"].view(-1)
                prev_mhat_flat = state["prev_mhat"].view(-1)
                gamma = torch.exp(
                    lambda_exp
                    * torch.nn.functional.cosine_similarity(grad_flat, prev_grad_flat, dim=0)
                )
                state["prev_grad"].copy_(grad)
                state["prev_mhat"].copy_(m_hat)
                state["prev_vhat"].copy_(v_hat)

                v_tilde = torch.maximum(v_hat, state['prev_vhat'])
                state["prev_vhat"].copy_(v_hat)
                prev_params = p.data.clone()
                denom = v_tilde.sqrt().add(eps)
                if self.track_stats:
                    gamma_value = float(gamma.item())
                    gamma_sum += gamma_value
                    gamma_min = min(gamma_min, gamma_value)
                    gamma_max = max(gamma_max, gamma_value)
                    gamma_count += 1

                    update = m_hat.mul(gamma).mul(-lr).div(denom)
                    update_sum += update.sum().item()
                    update_sq_sum += update.square().sum().item()
                    update_numel += update.numel()
                    denom2 = v_tilde.sqrt().add(eps)
                    update_what_if = m_hat.mul(-lr).div(denom2)
                    update_what_if_sum += update_what_if.sum().item()
                    update_what_if_numel += update_what_if.numel()
                    p.data.add_(update)
                else:
                    p.data.addcdiv_(m_hat.mul(gamma).mul(-lr), denom)

        if closure is not None and loss is not None and cc != 0.0:
            with torch.enable_grad():
                new_loss = closure()
            if new_loss > loss:
                p.data.copy_(prev_params + cc * (p.data - prev_params))

        if self.track_stats:
            if gamma_count > 0 and update_numel > 0:
                update_mean = update_sum / update_numel
                update_rms = math.sqrt(update_sq_sum / update_numel)
                update_var = max(update_sq_sum / update_numel - update_mean**2, 0.0)
                update_std = math.sqrt(update_var)
                update_what_if_mean = update_what_if_sum / update_what_if_numel
                
                self.last_step_stats = {
                    "gamma": gamma_sum / gamma_count,
                    "gamma_min": gamma_min,
                    "gamma_max": gamma_max,
                    "update_norm": math.sqrt(update_sq_sum),
                    "update_mean": update_mean,
                    "update_std": update_std,
                    "update_rms": update_rms,
                }
            else:
                self.last_step_stats = {}

        return loss
