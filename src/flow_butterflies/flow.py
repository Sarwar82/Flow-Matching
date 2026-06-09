from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _expand_time(t: Tensor, x: Tensor) -> Tensor:
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


def sample_linear_path(x0: Tensor, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Return x_t and the target velocity for the straight path x0 -> x1."""

    t_view = _expand_time(t, x1)
    xt = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    return xt, target_velocity


def flow_matching_loss(model: nn.Module, x1: Tensor) -> Tensor:
    """Mean-squared flow-matching loss in pixel space."""

    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt, target_velocity = sample_linear_path(x0, x1, t)
    predicted_velocity = model(xt, t)
    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def sample_flow(
    model: nn.Module,
    num_samples: int,
    image_size: int,
    steps: int,
    device: torch.device,
    method: Literal["euler", "heun"] = "euler",
) -> Tensor:
    """Generate samples by integrating dx/dt = model(x, t) from t=0 to t=1."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")

    was_training = model.training
    model.eval()

    x = torch.randn(num_samples, 3, image_size, image_size, device=device)
    for step in range(steps):
        t_value = step / steps
        next_t_value = (step + 1) / steps
        dt = next_t_value - t_value
        t = torch.full((num_samples,), t_value, device=device)

        velocity = model(x, t)
        if method == "euler":
            x = x + dt * velocity
        else:
            proposal = x + dt * velocity
            next_t = torch.full((num_samples,), next_t_value, device=device)
            next_velocity = model(proposal, next_t)
            x = x + 0.5 * dt * (velocity + next_velocity)

    if was_training:
        model.train()
    return x.clamp(-1.0, 1.0)
