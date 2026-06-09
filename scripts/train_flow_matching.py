import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from flow_butterflies.data import ButterflyImageDataset
from flow_butterflies.flow import flow_matching_loss, sample_flow
from flow_butterflies.model import FlowUNet
from flow_butterflies.utils import count_parameters, get_device, save_checkpoint, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 64x64 pixel-space flow-matching butterfly model.")
    parser.add_argument("--data-dir", default="data/butterflies/raw", help="Folder containing butterfly images.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    dataset = ButterflyImageDataset(
        root=args.data_dir,
        image_size=args.image_size,
        max_images=args.max_images,
        random_flip=True,
    )
    args.dataset_images = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
    )

    model = FlowUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])

    print(f"device: {device}")
    print(f"images: {len(dataset)}")
    print(f"parameters: {count_parameters(model):,}")
    if args.epochs < 50 and not args.resume:
        print(
            "warning: pixel-space flow matching usually needs far more than "
            f"{args.epochs} epochs for recognizable samples."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs}")

        for batch in progress:
            x1 = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, x1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(1, len(dataloader))
        print(f"epoch {epoch}: mean loss {mean_loss:.4f}")

        if epoch % args.save_every == 0:
            save_checkpoint(
                str(Path(args.checkpoint_dir) / f"flow_butterflies_epoch_{epoch:04d}.pt"),
                model,
                optimizer,
                epoch,
                global_step,
                args,
                mean_loss,
            )
            save_checkpoint(
                str(Path(args.checkpoint_dir) / "latest.pt"),
                model,
                optimizer,
                epoch,
                global_step,
                args,
                mean_loss,
            )

        if epoch % args.sample_every == 0:
            samples = sample_flow(
                model=model,
                num_samples=args.sample_count,
                image_size=args.image_size,
                steps=args.sample_steps,
                device=device,
                method="heun",
            )
            save_image_grid(samples, str(Path(args.output_dir) / f"samples_epoch_{epoch:04d}.png"))


if __name__ == "__main__":
    main()
