"""
Marcelo Gennari do Nascimento, 2022
marcelogennari@outlook.com

This is my version of a UNet. Not 100% sure that this is correct,
or if the temporal embedding are correct as well. It looks like it
works fine, even though might be a bit of an overly large model.
"""

import torch
from torch import nn, Tensor
from .common import TemporalEmbedding, LinearAttention, LabelEmbedding


class ResConvGroupNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        batch1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.LeakyReLU()

        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        batch2 = nn.BatchNorm2d(out_channels)
        relu2 = nn.LeakyReLU()

        layers = [batch1, relu1, conv2, batch2, relu2]

        self.feat = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return x + self.feat(x)


class UNet(nn.Module):
    def __init__(self, dim_emb: int = 1024):
        super().__init__()
        ch = [64, 128, 128, 64]
        self.ch = ch
        # Positional Embedding
        self.embedding1 = TemporalEmbedding(dim_emb, 1)

        # Input is 1x28x28
        self.block1 = ResConvGroupNorm(1, ch[0])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding2 = TemporalEmbedding(dim_emb, ch[0])
        self.block2 = ResConvGroupNorm(ch[0], ch[1])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1, bias=False)

        # Now input is 64x7x7
        self.embedding3 = TemporalEmbedding(dim_emb, ch[1])
        self.block3 = ResConvGroupNorm(ch[1], ch[2])
        self.attention1 = LinearAttention(ch[2])
        self.up1 = nn.ConvTranspose2d(ch[2], ch[2], 4, stride=2, padding=1, bias=False)

        # Now input is 64x14x14
        new_ch = ch[2] + ch[1]
        self.embedding4 = TemporalEmbedding(dim_emb, new_ch)
        self.block4 = ResConvGroupNorm(new_ch, ch[3])
        self.up2 = nn.ConvTranspose2d(ch[3], ch[3], 4, stride=2, padding=1, bias=False)

        # Now input is 16x28x28
        new_ch = ch[3] + ch[0]
        self.embedding5 = TemporalEmbedding(dim_emb, new_ch)
        self.block5 = ResConvGroupNorm(new_ch, 1)
        self.out = nn.Conv2d(1, 1, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x0 = self.embedding1(x, t)
        x1 = self.block1(x0)
        x1 = self.embedding2(x1, t)
        x2 = self.block2(self.down1(x1))
        x2 = self.embedding3(x2, t)
        x3 = self.up1(self.attention1(self.block3(self.down2(x2))))
        x4 = torch.cat([x2, x3], dim=1)
        x4 = self.embedding4(x4, t)
        x5 = self.up2(self.block4(x4))
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.embedding5(x6, t)
        out = self.out(self.block5(x6))
        return out


class ConditionalUNet(UNet):
    def __init__(self, dim_emb: int = 1024):
        super().__init__(dim_emb)
        self.label_emb1 = LabelEmbedding(dim_emb, self.ch[0])
        self.label_emb2 = LabelEmbedding(dim_emb, self.ch[1])
        self.label_emb3 = LabelEmbedding(dim_emb, self.ch[2])
        self.label_emb4 = LabelEmbedding(dim_emb, self.ch[3])
        self.label_emb5 = LabelEmbedding(dim_emb, self.ch[3] + self.ch[0])

    def forward(self, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
        x0 = self.embedding1(x, t)
        x1 = self.block1(x0)
        x1 = self.label_emb1(x1, label)
        x1 = self.embedding2(x1, t)
        x2 = self.block2(self.down1(x1))
        x2 = self.label_emb2(x2, label)
        x2 = self.embedding3(x2, t)
        crossed = self.label_emb3(self.block3(self.down2(x2)), label)
        x3 = self.up1(self.attention1(crossed))
        x4 = torch.cat([x2, x3], dim=1)
        x4 = self.embedding4(x4, t)
        x5 = self.up2(self.label_emb4(self.block4(x4), label))
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.label_emb5(x6, label)
        x6 = self.embedding5(x6, t)
        out = self.out(self.block5(x6))
        return out