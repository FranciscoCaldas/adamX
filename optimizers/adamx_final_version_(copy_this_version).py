from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class AdamXFast(Optimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        alpha: float = 0.99,
        lambda_exp: float = 1.0,
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
            lambda_exp = group["lambda_exp"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamXFast does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.ones_like(p)
                    state["prev_vhat"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]

                state["step"] += 1
                step = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)

                gamma = torch.exp(
                    lambda_exp
                    * torch.nn.functional.cosine_similarity(
                        grad.view(-1), state["prev_grad"].view(-1), dim=0
                    )
                )
                state["prev_grad"].copy_(grad)

                v_tilde = torch.maximum(v_hat, state["prev_vhat"])
                state["prev_vhat"].copy_(v_tilde)

                denom = v_tilde.sqrt().add(eps)
                p.data.addcdiv_(m_hat.mul(gamma).mul(-lr), denom)

        return loss
