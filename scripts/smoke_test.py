import torch
import torchvision
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from flow_butterflies.flow import flow_matching_loss, sample_flow
from flow_butterflies.model import FlowUNet
from flow_butterflies.utils import count_parameters, get_device


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"mps available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"cuda available: {torch.cuda.is_available()}")

    device = get_device("auto")
    model = FlowUNet(base_channels=8, time_channels=32).to(device)
    x = torch.randn(2, 3, 64, 64, device=device)
    loss = flow_matching_loss(model, x)
    loss.backward()
    samples = sample_flow(model, num_samples=2, image_size=64, steps=2, device=device)

    print(f"device: {device}")
    print(f"model parameters: {count_parameters(model):,}")
    print(f"flow loss: {loss.item():.4f}")
    print(f"generated sample tensor shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
