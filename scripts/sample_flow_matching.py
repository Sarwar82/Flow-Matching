import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from flow_butterflies.flow import sample_flow
from flow_butterflies.model import FlowUNet
from flow_butterflies.utils import get_device, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained 64x64 butterfly flow-matching model.")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--output", default="outputs/final_samples.png")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    image_size = int(train_args.get("image_size", 64))
    base_channels = int(train_args.get("base_channels", 32))
    epoch = int(checkpoint.get("epoch", 0))
    trained_images = train_args.get("dataset_images", "unknown")
    max_images = train_args.get("max_images")

    print(f"checkpoint: {args.checkpoint}")
    print(f"trained epoch: {epoch}")
    print(f"training images: {trained_images}")
    if epoch < 50 or max_images is not None:
        print(
            "warning: this checkpoint looks like a short/debug run; "
            "samples may look like noise rather than butterflies."
        )

    model = FlowUNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])

    samples = sample_flow(
        model=model,
        num_samples=args.num_samples,
        image_size=image_size,
        steps=args.steps,
        device=device,
        method=args.method,
    )
    save_image_grid(samples, args.output)
    print(f"saved {args.num_samples} samples to {args.output}")


if __name__ == "__main__":
    main()
