import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sinusoidal_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Sinusoidal embedding for continuous t in [0, 1]."""

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = 1000.0 * t[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(batch, 3, channels, height * width).unbind(dim=1)
        attention = torch.bmm(q.transpose(1, 2), k) * (channels ** -0.5)
        attention = attention.softmax(dim=-1)
        h = torch.bmm(v, attention.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class FlowUNet(nn.Module):
    """A compact UNet that predicts pixel-space velocity fields for flow matching."""

    def __init__(self, image_channels: int = 3, base_channels: int = 32, time_channels: int = 128) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.time_channels = time_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels * 4),
            nn.SiLU(),
            nn.Linear(time_channels * 4, time_channels),
        )

        self.input = nn.Conv2d(image_channels, c1, kernel_size=3, padding=1)

        self.down1a = ResBlock(c1, c1, time_channels)
        self.down1b = ResBlock(c1, c1, time_channels)
        self.downsample1 = Downsample(c1, c2)

        self.down2a = ResBlock(c2, c2, time_channels)
        self.down2b = ResBlock(c2, c2, time_channels)
        self.downsample2 = Downsample(c2, c3)

        self.down3a = ResBlock(c3, c3, time_channels)
        self.down3b = ResBlock(c3, c3, time_channels)

        self.mid1 = ResBlock(c3, c3, time_channels)
        self.mid_attn = SelfAttention2d(c3)
        self.mid2 = ResBlock(c3, c3, time_channels)

        self.up3a = ResBlock(c3 + c3, c3, time_channels)
        self.up3b = ResBlock(c3, c3, time_channels)
        self.upsample3 = Upsample(c3, c2)

        self.up2a = ResBlock(c2 + c2, c2, time_channels)
        self.up2b = ResBlock(c2, c2, time_channels)
        self.upsample2 = Upsample(c2, c1)

        self.up1a = ResBlock(c1 + c1, c1, time_channels)
        self.up1b = ResBlock(c1, c1, time_channels)

        self.output = nn.Sequential(
            group_norm(c1),
            nn.SiLU(),
            nn.Conv2d(c1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        time_embedding = self.time_mlp(sinusoidal_embedding(t, self.time_channels))

        h = self.input(x)

        h = self.down1a(h, time_embedding)
        h = self.down1b(h, time_embedding)
        skip1 = h
        h = self.downsample1(h)

        h = self.down2a(h, time_embedding)
        h = self.down2b(h, time_embedding)
        skip2 = h
        h = self.downsample2(h)

        h = self.down3a(h, time_embedding)
        h = self.down3b(h, time_embedding)
        skip3 = h

        h = self.mid1(h, time_embedding)
        h = self.mid_attn(h)
        h = self.mid2(h, time_embedding)

        h = torch.cat([h, skip3], dim=1)
        h = self.up3a(h, time_embedding)
        h = self.up3b(h, time_embedding)
        h = self.upsample3(h)

        h = torch.cat([h, skip2], dim=1)
        h = self.up2a(h, time_embedding)
        h = self.up2b(h, time_embedding)
        h = self.upsample2(h)

        h = torch.cat([h, skip1], dim=1)
        h = self.up1a(h, time_embedding)
        h = self.up1b(h, time_embedding)

        return self.output(h)
