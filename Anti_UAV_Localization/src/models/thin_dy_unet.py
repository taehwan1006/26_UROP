"""
ThinDyUNet: Lightweight U-Net with Dynamic Convolution for Anti-UAV Semantic Segmentation.

논문 Section 4, Figure 6 기반 구현.
- U-Net encoder-decoder 구조
- 고정 64채널 (모든 layer)
- Encoder에 SE 기반 attention dynamic convolution 적용
- ~1.3M 파라미터
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConv2d(nn.Module):
    """
    Attention-based Dynamic Convolution (논문 수식 1~5).

    N개의 커널 후보 중 입력에 따라 attention score를 계산하여
    가중합으로 dynamic kernel을 생성한 뒤 convolution을 수행한다.

    X ∈ R^{C×H×W}
    z = Conv1x1(Conv1x1(GAP(X))) ∈ R^N
    α = Softmax(z)
    K_dyn = Σ αi * Ki
    Y = LeakyReLU(GroupNorm(K_dyn * X))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        n_kernels: int = 4,
        reduction: int = 4,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_kernels = n_kernels

        # N개 커널을 하나의 텐서로 합쳐 단일 conv 호출로 처리
        # shape: (N * out_ch, in_ch, kH, kW)
        weight = torch.randn(n_kernels, out_channels, in_channels, kernel_size, kernel_size)
        weight *= (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.weight = nn.Parameter(weight.view(n_kernels * out_channels, in_channels, kernel_size, kernel_size))

        # Attention 경로: GAP → Linear → ReLU → Linear → Softmax
        mid_channels = max(in_channels // reduction, 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, n_kernels),
        )

        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        효율적 구현: conv(x, Σ αᵢ·Kᵢ) = Σ αᵢ · conv(x, Kᵢ)

        N개 커널을 한 번의 conv로 처리(출력 채널 N*out_ch)한 뒤
        attention score로 가중합한다. cuDNN이 잘 최적화하는 일반 conv 1번이라
        groups=batch_size 방식보다 훨씬 빠르다.
        """
        B = x.size(0)
        N = self.n_kernels
        C_out = self.out_channels

        # Attention: (B, N), Σαᵢ = 1
        alpha = F.softmax(self.attention(x), dim=1)

        # 단일 conv: (B, in_ch, H, W) → (B, N*out_ch, H, W)
        out = F.conv2d(x, self.weight, padding=self.padding)

        # (B, N, out_ch, H, W) → α 가중합 → (B, out_ch, H, W)
        out = out.view(B, N, C_out, out.size(2), out.size(3))
        out = (alpha.view(B, N, 1, 1, 1) * out).sum(dim=1)

        out = self.norm(out)
        out = self.act(out)
        return out


class DynamicConvBlock(nn.Module):
    """Encoder용 Dynamic Convolution 블록 (DynConv × 2)."""

    def __init__(self, in_channels: int, out_channels: int, n_kernels: int = 4):
        super().__init__()
        self.conv1 = DynamicConv2d(in_channels, out_channels, n_kernels=n_kernels)
        self.conv2 = DynamicConv2d(out_channels, out_channels, n_kernels=n_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvBlock(nn.Module):
    """Decoder용 일반 Convolution 블록 (Conv-GN-LeakyReLU × 2)."""

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


class ThinDyUNet(nn.Module):
    """
    ThinDyUNet: Lightweight Dynamic U-Net.

    - Encoder: 4단계 downsampling, 각 단계 DynamicConvBlock (64ch 고정)
    - Decoder: 4단계 upsampling, 각 단계 ConvBlock (64ch 고정)
    - Skip connections
    - 최종: 1x1 Conv → binary mask
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_ch: int = 64, n_kernels: int = 3):
        super().__init__()
        ch = base_ch

        # Encoder (Dynamic Convolution)
        self.enc1 = DynamicConvBlock(in_channels, ch, n_kernels=n_kernels)
        self.enc2 = DynamicConvBlock(ch, ch, n_kernels=n_kernels)
        self.enc3 = DynamicConvBlock(ch, ch, n_kernels=n_kernels)
        self.enc4 = DynamicConvBlock(ch, ch, n_kernels=n_kernels)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck (Regular Convolution)
        self.bottleneck = ConvBlock(ch, ch)

        # Decoder (skip concat이므로 in_channels = ch*2)
        self.up4 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch * 2, ch)

        self.up3 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch * 2, ch)

        self.up2 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch * 2, ch)

        self.up1 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch * 2, ch)

        # 최종 출력
        self.out_conv = nn.Conv2d(ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)          # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 64, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 64, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 64, H/16, W/16)

        # Decoder path with skip connections
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

        return self.out_conv(d1)  # (B, 1, H, W) — logit, sigmoid은 loss에서 처리
