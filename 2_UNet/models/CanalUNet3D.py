"""UNet 3D sencilla para los parches simulados de CBCT (etapa 3).

Como `UNet3D.py` (mismo bloque conv-GroupNorm-dropout-ReLU), pero sin el embedding de posición del
parche dentro del volumen (idea de Pulpy3D: aquí los parches salen de un simulador y no hay volumen
global) y devolviendo LOGITS, no sigmoide: así se puede usar BCE con logits y etiquetas blandas.
"""
import torch
import torch.nn as nn

from models.UNet3D import UNet3d


class CanalUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16, depth=4, dropout=0.2):
        super().__init__()
        f = init_features
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)
        ch = in_channels
        for i in range(depth):
            self.encoders.append(UNet3d._block(ch, f * 2 ** i, name=f"enc{i}", prob=dropout))
            ch = f * 2 ** i
        self.bottleneck = UNet3d._block(ch, f * 2 ** depth, name="bott", prob=dropout)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = f * 2 ** depth
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose3d(ch, f * 2 ** i, kernel_size=2, stride=2))
            self.decoders.append(UNet3d._block(f * 2 ** i * 2, f * 2 ** i, name=f"dec{i}", prob=dropout))
            ch = f * 2 ** i
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat((up(x), skip), dim=1))
        return self.head(x)
