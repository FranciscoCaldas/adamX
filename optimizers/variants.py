from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


def _apply_decoupled_weight_decay(
    param: torch.nn.Parameter, lr: float, weight_decay: float
) -> None:
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)



class Yogi(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0.0,
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
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_sq = grad.mul(grad)
                direction = torch.sign(grad_sq - exp_avg_sq)
                exp_avg_sq.addcmul_(grad_sq, direction, value=1.0 - beta2)
                exp_avg_sq.clamp_(min=0.0)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)
                _apply_decoupled_weight_decay(param, lr, weight_decay)
                param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss


class Adan(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.98, 0.92),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            beta3 = 0.999
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_diff"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["prev_grad"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_diff = state["exp_avg_diff"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                state["step"] += 1
                step = state["step"]

                grad_diff = grad - prev_grad
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)

                grad_mix = grad + (1.0 - beta2) * grad_diff
                exp_avg_sq.mul_(beta3).addcmul_(grad_mix, grad_mix, value=1.0 - beta3)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                bias_correction3 = 1.0 - beta3**step

                numerator = exp_avg.div(bias_correction1)
                numerator.add_(exp_avg_diff.div(bias_correction2), alpha=1.0 - beta2)
                denom = exp_avg_sq.div(bias_correction3).sqrt().add_(eps)

                _apply_decoupled_weight_decay(param, lr, weight_decay)
                param.addcdiv_(numerator, denom, value=-lr)
                prev_grad.copy_(grad)

        return loss


class Gala(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        alpha: float = 0.99,
        lambda_exp: float = 1.0,
        cc: float = 0.0,
        weight_decay: float = 0.0,
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
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            alpha = group["alpha"]
            lambda_exp = group["lambda_exp"]
            cc = group["cc"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Gala does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["prev_grad"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                if step == 1:
                    gamma = torch.ones((), device=grad.device, dtype=grad.dtype)
                else:
                    gamma = torch.exp(
                        lambda_exp
                        * torch.nn.functional.cosine_similarity(
                            grad.view(-1), prev_grad.view(-1), dim=0
                        )
                    )
                if cc != 0.0:
                    gamma = gamma.mul(1.0 - cc).add(cc)

                direction = alpha * m_hat + (1.0 - alpha) * grad
                denom = v_hat.sqrt().add_(eps)
                _apply_decoupled_weight_decay(param, lr, weight_decay)
                param.addcdiv_(direction.mul(gamma), denom, value=-lr)
                prev_grad.copy_(grad)

        return loss