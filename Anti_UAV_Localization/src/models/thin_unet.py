"""
ThinUNet: Ablation 모델 (Dynamic Convolution → Regular Convolution).

논문 Table 6, 7의 ablation study 재현용.
동일한 U-Net 구조에서 dynamic conv를 일반 conv로 교체.
- 예상 파라미터: ~15.3M (논문 기준)
"""

import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    """일반 Double Convolution 블록 (Conv-GN-LeakyReLU × 2)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ThinUNet(nn.Module):
    """
    ThinUNet: ThinDyUNet와 동일한 구조이되 Regular Convolution 사용.
    Ablation study에서 dynamic conv의 효과를 입증하기 위한 비교 모델.

    채널 수는 U-Net 기본 구조를 따름 (64→128→256→512→1024)으로
    논문의 ~15.3M 파라미터에 맞춘다.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_ch: int = 64):
        super().__init__()

        # Encoder (채널 수 증가 — 일반 U-Net 구조)
        self.enc1 = DoubleConvBlock(in_channels, base_ch)
        self.enc2 = DoubleConvBlock(base_ch, base_ch * 2)
        self.enc3 = DoubleConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(base_ch * 8, base_ch * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(base_ch * 16, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(base_ch * 8, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(base_ch * 4, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)
