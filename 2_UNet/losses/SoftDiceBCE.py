"""Pérdida con etiqueta blanda (fracción de conducto por vóxel, 0..1) y mapa de pesos.

BCE con logits (acepta objetivos entre 0 y 1) + Dice blando por muestra. Un peso 0 en un vóxel lo quita
de las dos partes ("ignorar"). Formas: logits, objetivo y peso, todos Bx1xDxHxW.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceBCE(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight, self.dice_weight, self.eps = bce_weight, dice_weight, eps

    def forward(self, logits, target, weight=None):
        logits, target = logits.float(), target.float()
        w = torch.ones_like(target) if weight is None else weight.float()
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        inter = (p * target * w).sum(dims)
        den = (p * w).sum(dims) + (target * w).sum(dims)
        dice = 1 - (2 * inter + self.eps) / (den + self.eps)
        return self.bce_weight * bce + self.dice_weight * dice.mean()
