"""Director-specified P05-T02c models. Pure PyTorch, FP32 only.

FCN follows the Wang-style 128/256/128 8/5/3 Conv1d baseline. ResNet1D
and diagonal state-space (S4D) consume [signal, broadcast mask] = 8 channels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SameConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        if stride != 1:
            raise ValueError("SameConv1d is only used at stride=1")
        self.padding = nn.ConstantPad1d(((kernel - 1) // 2, kernel // 2), value=0.0)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, bias=False)

    def forward(self, x):
        return self.conv(self.padding(x))


class FCNWang2017(nn.Module):
    """Three-layer FCN with no mask, GAP, and a binary-logit head."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            SameConv1d(4, 128, 8), nn.BatchNorm1d(128), nn.ReLU(),
            SameConv1d(128, 256, 5), nn.BatchNorm1d(256), nn.ReLU(),
            SameConv1d(256, 128, 3), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.blocks(x).mean(dim=-1)).squeeze(-1)


class ResidualBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (nn.Identity() if in_ch == out_ch and stride == 1
                     else nn.Sequential(
                         nn.Conv1d(in_ch, out_ch, 1, stride=stride,
                                   bias=False), nn.BatchNorm1d(out_ch)))

    def forward(self, x):
        skip = self.skip(x)
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(y + skip)


class ResNet1d(nn.Module):
    """Exactly four residual stages [64,128,128,256], kernel 7."""
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            ResidualBlock1d(8, 64, 2),
            ResidualBlock1d(64, 128, 2),
            ResidualBlock1d(128, 128, 2),
            ResidualBlock1d(128, 256, 2),
        )
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.dropout(
            self.stages(x).mean(dim=-1))).squeeze(-1)


class S4DKernel(nn.Module):
    """Diagonal continuous SSM with zero-order hold, per-channel kernels."""
    def __init__(self, d_model=128, d_state=16):
        super().__init__()
        self.log_dt = nn.Parameter(torch.linspace(-6.0, -3.0, d_model))
        self.log_a_real = nn.Parameter(torch.zeros(d_model, d_state))
        imag = math.pi * torch.arange(d_state, dtype=torch.float32)
        self.a_imag = nn.Parameter(imag.unsqueeze(0).repeat(d_model, 1))
        self.c_ri = nn.Parameter(torch.randn(d_model, d_state, 2) * 0.02)

    def forward(self, length):
        # Compute complex kernels in torch; real FFT convolution follows.
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        a = torch.complex(-torch.exp(self.log_a_real), self.a_imag)
        c = torch.view_as_complex(self.c_ri.contiguous())
        dt_a = dt * a
        # Discretize B=1 exactly, avoiding division by zero (Re(A) < 0).
        # expm1(complex) is unavailable on some Windows torch 2.0.1 builds.
        b_bar = (torch.exp(dt_a) - 1.0) / a
        t = torch.arange(length, device=dt.device, dtype=dt.dtype)
        powers = torch.exp(dt_a.unsqueeze(-1) * t)
        return 2.0 * torch.sum(
            (c * b_bar).unsqueeze(-1) * powers, dim=1
        ).real


class S4DBlock(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(d_model=d_model)
        self.d_skip = nn.Parameter(torch.ones(d_model))
        self.output = nn.Sequential(
            nn.GELU(), nn.Conv1d(d_model, 2 * d_model, 1),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        length = x.shape[-1]
        u = self.norm(x.transpose(1, 2)).transpose(1, 2)
        kernel = self.kernel(length)
        freq_u = torch.fft.rfft(u, n=2 * length)
        freq_k = torch.fft.rfft(kernel, n=2 * length)
        y = torch.fft.irfft(freq_u * freq_k.unsqueeze(0),
                            n=2 * length)[..., :length]
        y = y + u * self.d_skip.view(1, -1, 1)
        return x + self.output(y)


class S4DClassifier(nn.Module):
    """Four pure-PyTorch S4D blocks, d=128, GAP, binary logit."""
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(8, 128, 1)
        self.blocks = nn.Sequential(*(S4DBlock(128) for _ in range(4)))
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.blocks(self.input_projection(x))
        pooled = self.final_norm(x.mean(dim=-1))
        return self.head(pooled).squeeze(-1)


def make_model(name):
    factory = {"fcn": FCNWang2017,
               "resnet1d": ResNet1d, "s4d": S4DClassifier}
    if name not in factory:
        raise ValueError("Unknown Director-approved model")
    model = factory[name]()
    parameters = sum(p.numel() for p in model.parameters())
    if name in ("resnet1d", "s4d") and parameters > 2_000_000:
        raise RuntimeError("Model exceeds 2,000,000 parameter limit")
    return model, parameters
