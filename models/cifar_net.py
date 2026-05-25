from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero



class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * in_channels * whiten_kernel_size**2
        self.whiten = nn.Conv2d(
            in_channels,
            whiten_width,
            whiten_kernel_size,
            padding=0,
            bias=True,
        )
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], num_classes, bias=False)

    def reset(self) -> None:
        for module in self.modules():
            if type(module) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                module.reset_parameters()
        weights = self.head.weight.data
        weights *= 1 / weights.std()

    def init_whiten(self, train_images: torch.Tensor, eps: float = 5e-4) -> None:
        channels = train_images.shape[1]
        height, width = self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, height, 1)
            .unfold(3, width, 1)
            .transpose(1, 3)
            .reshape(-1, channels, height, width)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, channels, height, width)
        eigenvectors_scaled /= torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        whiten_filters = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
        self.whiten.weight.data[:] = whiten_filters.to(self.whiten.weight.dtype)

    def forward(self, x: torch.Tensor, whiten_bias_grad: bool = True) -> torch.Tensor:
        target_dtype = self.whiten.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        bias = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, bias if whiten_bias_grad else bias.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)
