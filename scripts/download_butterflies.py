import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from datasets import load_dataset
except ImportError as error:
    raise SystemExit(
        "Missing dependency: datasets. Install with `python -m pip install -r requirements.txt`."
    ) from error


DATASET_NAME = "huggan/smithsonian_butterflies_subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the HugGAN Smithsonian butterfly subset into a local image folder."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/butterflies/raw")
    parser.add_argument("--metadata-path", default="data/butterflies/metadata.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of images to export.")
    parser.add_argument("--image-size", type=int, default=None, help="Optional square resize while exporting.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slugify(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def image_from_row(row: dict[str, Any]) -> Image.Image:
    image = row.get("image")
    if image is None:
        raise ValueError("Dataset row does not contain an image column.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        import io

        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(image)!r}")


def metadata_from_row(row: dict[str, Any], image_path: Path) -> dict[str, Any]:
    excluded = {"image"}
    metadata = {key: value for key, value in row.items() if key not in excluded}
    metadata["local_path"] = str(image_path)
    return metadata


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    exported = 0
    skipped = 0
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, row in enumerate(dataset):
            row_id = slugify(row.get("id") or row.get("image_hash"), f"butterfly-{index:05d}")
            name = slugify(row.get("name"), "butterfly")
            image_path = output_dir / f"{index:05d}-{name}-{row_id}.png"

            if image_path.exists() and not args.overwrite:
                skipped += 1
            else:
                image = image_from_row(row)
                if args.image_size is not None:
                    image = image.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
                image.save(image_path)
                exported += 1

            metadata_file.write(json.dumps(metadata_from_row(row, image_path), ensure_ascii=True) + "\n")

    print(f"dataset: {args.dataset}/{args.split}")
    print(f"rows: {len(dataset)}")
    print(f"exported: {exported}")
    print(f"skipped existing: {skipped}")
    print(f"images: {output_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
