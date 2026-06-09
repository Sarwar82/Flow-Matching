from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torchvision.utils import save_image


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_image_grid(images: torch.Tensor, path: str, nrow: Optional[int] = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = (images.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
    if nrow is None:
        nrow = int(images.shape[0] ** 0.5)
    save_image(images, output_path, nrow=max(1, nrow))


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: Any,
    loss: float,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
            "loss": loss,
        },
        output_path,
    )
