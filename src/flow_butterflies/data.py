from pathlib import Path
from typing import Optional

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class ButterflyImageDataset(Dataset):
    """Loads butterfly images and maps them to pixel tensors in [-1, 1]."""

    def __init__(
        self,
        root: str,
        image_size: int = 64,
        max_images: Optional[int] = None,
        random_flip: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.paths = self._find_images(self.root)
        if max_images is not None:
            self.paths = self.paths[:max_images]

        if not self.paths:
            raise FileNotFoundError(
                f"No images found in {self.root}. Add butterfly images under this folder "
                "or pass --data-dir to the folder that contains them."
            )

        transform_steps = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
        if random_flip:
            transform_steps.append(transforms.RandomHorizontalFlip())
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    @staticmethod
    def _find_images(root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)
