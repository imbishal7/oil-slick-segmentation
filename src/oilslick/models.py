"""U-Net (Ronneberger et al., 2015) and Attention U-Net (Oktay et al., 2018).

Both share one implementation so the only difference between them is the
attention gates on the skip connections. The networks return raw logits;
apply ``torch.sigmoid`` to get oil probabilities.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


class DoubleConv(nn.Sequential):
    """Two 3x3 Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(*conv_bn_relu(in_ch, out_ch), *conv_bn_relu(out_ch, out_ch))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018, Fig. 2).

    The skip features ``x`` are gated by a signal ``g`` from the next-coarser
    decoder level. ``x`` is brought to ``g``'s resolution with a strided 2x2
    conv, the attention coefficients alpha in [0, 1] are computed there and
    then upsampled back to ``x``'s resolution.
    """

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=2, stride=2, bias=False)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(x_ch, x_ch, kernel_size=1, bias=False), nn.BatchNorm2d(x_ch)
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.theta_x(x)
        phi = F.interpolate(
            self.phi_g(g), size=theta.shape[2:], mode="bilinear", align_corners=False
        )
        alpha = torch.sigmoid(self.psi(F.relu(theta + phi)))
        alpha = F.interpolate(alpha, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.out(x * alpha), alpha


class UpBlock(nn.Module):
    """Bilinear upsample + 3x3 conv, (optionally gated) skip concat, DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention: bool) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            *conv_bn_relu(in_ch, out_ch),
        )
        self.gate = AttentionGate(skip_ch, in_ch, max(skip_ch // 2, 1)) if attention else None
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        alpha = None
        if self.gate is not None:
            skip, alpha = self.gate(skip, x)
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1)), alpha


class UNet(nn.Module):
    """5-level U-Net; ``attention=True`` turns it into an Attention U-Net.

    Args:
        in_channels: input channels (2 for VV, VH).
        base_channels: filters at the first level, doubled at each level.
        depth: number of resolution levels including the bottleneck.
        attention: add attention gates to every skip connection.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        depth: int = 5,
        attention: bool = False,
    ) -> None:
        super().__init__()
        widths = [base_channels * 2**i for i in range(depth)]
        self.attention = attention
        self.encoder = nn.ModuleList(
            [DoubleConv(in_channels, widths[0])]
            + [DoubleConv(widths[i - 1], widths[i]) for i in range(1, depth)]
        )
        self.decoder = nn.ModuleList(
            UpBlock(widths[i], widths[i - 1], widths[i - 1], attention)
            for i in range(depth - 1, 0, -1)
        )
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for i, block in enumerate(self.encoder):
            if i > 0:
                x = F.max_pool2d(x, kernel_size=2)
            x = block(x)
            skips.append(x)
        x = skips.pop()

        alphas = []
        for block in self.decoder:
            x, alpha = block(x, skips.pop())
            if alpha is not None:
                alphas.append(alpha)

        logits = self.head(x)
        return (logits, alphas) if return_attention else logits


def build_model(attention: bool, base_channels: int = 32, in_channels: int = 2) -> UNet:
    return UNet(in_channels=in_channels, base_channels=base_channels, attention=attention)
