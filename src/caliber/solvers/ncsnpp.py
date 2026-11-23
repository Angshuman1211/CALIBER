# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NCSN++ in PyTorch, for Song et al.'s medical score models (D-016 / D79, T2.10).

Adapted from ``score_sde_pytorch`` (github.com/yang-song/score_sde_pytorch, Apache-2.0):
``models/ncsnpp.py``, ``models/layerspp.py``, ``models/layers.py``,
``models/up_or_down_sampling.py`` and ``op/upfirdn2d.py``. Changes from that code:

* **Only the configuration the published checkpoints use.** Song, Shen, Xing & Ermon
  (ICLR 2022) trained three priors -- BraTS 2021 brain MRI at 240^2, LIDC lung CT at 320^2,
  LDCT at 512^2 -- and all three share one architecture: Gaussian Fourier noise embedding,
  BigGAN residual blocks with FIR resampling, ``output_skip`` / ``input_skip`` pyramids
  combined by sum, skip rescaling, and output scaled by 1/sigma. They differ only in width,
  depth, attention resolution and image size, which are the constructor's arguments. The
  DDPM-block and residual-pyramid branches are not ported, so nothing here can run untested.
* **No compiled CUDA kernel.** The reference ``upfirdn2d`` JIT-compiles a C++/CUDA extension
  at import, which needs a compiler toolchain this project does not assume. Its own
  pure-PyTorch fallback (``upfirdn2d_native``), which the reference already uses on the CPU,
  is used on every device; it computes the same thing.
* **Module order is the Flax call order**, deliberately, as in the reference: the Flax
  checkpoints name their layers by class and call index (``ResnetBlockBigGANpp_9``), and
  ``scripts/convert_song_checkpoint.py`` maps them onto ``all_modules`` by that order.

The network takes images in **[0, 1]** (the checkpoints were trained with
``data.centered = False`` and rescale to [-1, 1] internally) and a noise level ``sigma`` per
image, and returns the VE-SDE score ``grad_x log p_sigma(x)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class NCSNppConfig:
    """The architecture knobs that differ between Song et al.'s three checkpoints."""

    image_size: int
    nf: int = 32
    ch_mult: tuple[int, ...] = (1, 1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (30,)
    num_channels: int = 1
    fourier_scale: float = 16.0
    fir_kernel: tuple[int, ...] = (1, 3, 3, 1)
    #: The VE-SDE noise range the model was trained on.
    sigma_min: float = 0.01
    sigma_max: float = 128.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NCSNppConfig":
        d = dict(d)
        for key in ("ch_mult", "attn_resolutions", "fir_kernel"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


#: ``configs/ve/brats_ncsnpp_continuous.py`` of score_inverse_problems, checkpoint 26.
BRATS_240 = NCSNppConfig(image_size=240)


# -- FIR resampling (StyleGAN2's upfirdn2d, pure PyTorch) ----------------------


def _upfirdn2d(x: Tensor, kernel: Tensor, up: int = 1, down: int = 1,
               pad: tuple[int, int] = (0, 0)) -> Tensor:
    """Upsample by zero insertion, pad, FIR-filter, downsample. ``upfirdn2d_native``."""
    _, channel, in_h, in_w = x.shape
    x = x.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape
    pad0, pad1 = pad

    out = x.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up - 1, 0, 0, 0, up - 1])
    out = out.view(-1, in_h * up, in_w * up, minor)
    out = F.pad(out, [0, 0, max(pad0, 0), max(pad1, 0), max(pad0, 0), max(pad1, 0)])
    out = out[:, max(-pad0, 0): out.shape[1] - max(-pad1, 0),
              max(-pad0, 0): out.shape[2] - max(-pad1, 0), :]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up + pad0 + pad1, in_w * up + pad0 + pad1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up + pad0 + pad1 - kernel_h + 1,
                      in_w * up + pad0 + pad1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down, ::down, :]
    out_h = (in_h * up + pad0 + pad1 - kernel_h) // down + 1
    out_w = (in_w * up + pad0 + pad1 - kernel_w) // down + 1
    return out.reshape(-1, channel, out_h, out_w)


def _fir(k: tuple[int, ...], gain: float, device, dtype) -> Tensor:
    k = np.asarray(k, dtype=np.float32)
    k = np.outer(k, k)
    k /= np.sum(k)
    return torch.tensor(k * gain, device=device, dtype=dtype)


def upsample_2d(x: Tensor, k: tuple[int, ...], factor: int = 2) -> Tensor:
    kernel = _fir(k, factor ** 2, x.device, x.dtype)
    p = kernel.shape[0] - factor
    return _upfirdn2d(x, kernel, up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x: Tensor, k: tuple[int, ...], factor: int = 2) -> Tensor:
    kernel = _fir(k, 1.0, x.device, x.dtype)
    p = kernel.shape[0] - factor
    return _upfirdn2d(x, kernel, down=factor, pad=((p + 1) // 2, p // 2))


# -- layers ---------------------------------------------------------------------


def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1)


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)


class GaussianFourierProjection(nn.Module):
    """Random Fourier features of log(sigma); ``W`` is fixed at training and stored."""

    def __init__(self, embedding_size: int, scale: float):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class NIN(nn.Module):
    """A 1x1 convolution written as a contraction over channels (Flax's ``NIN``)."""

    def __init__(self, in_dim: int, num_units: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(in_dim, num_units))
        self.b = nn.Parameter(torch.zeros(num_units))

    def forward(self, x: Tensor) -> Tensor:
        y = torch.einsum("bhwc,cd->bhwd", x.permute(0, 2, 3, 1), self.W) + self.b
        return y.permute(0, 3, 1, 2)


class Combine(nn.Module):
    """``progressive_combine = 'sum'``: project the input pyramid and add it."""

    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.Conv_0(x) + y


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention, with skip rescaling."""

    def __init__(self, channels: int):
        super().__init__()
        self.GroupNorm_0 = group_norm(channels)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q, k, v = self.NIN_0(h), self.NIN_1(h), self.NIN_2(h)
        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = F.softmax(w.reshape(B, H, W, H * W), dim=-1).reshape(B, H, W, H, W)
        h = self.NIN_3(torch.einsum("bhwij,bcij->bchw", w, v))
        return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    """BigGAN residual block with optional FIR up/down-sampling and a noise-level bias."""

    def __init__(self, in_ch: int, out_ch: int | None = None, temb_dim: int | None = None,
                 up: bool = False, down: bool = False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch or in_ch
        self.GroupNorm_0 = group_norm(in_ch)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        self.Dense_0 = nn.Linear(temb_dim, out_ch)
        self.GroupNorm_1 = group_norm(out_ch)
        self.Conv_1 = conv3x3(out_ch, out_ch)
        self.has_skip_conv = in_ch != out_ch or up or down
        if self.has_skip_conv:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.up, self.down, self.fir_kernel = up, down, tuple(fir_kernel)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = F.silu(self.GroupNorm_0(x))
        if self.up:
            h, x = upsample_2d(h, self.fir_kernel), upsample_2d(x, self.fir_kernel)
        elif self.down:
            h, x = downsample_2d(h, self.fir_kernel), downsample_2d(x, self.fir_kernel)
        h = self.Conv_0(h)
        h = h + self.Dense_0(F.silu(temb))[:, :, None, None]
        h = self.Conv_1(F.silu(self.GroupNorm_1(h)))  # dropout is 0 in every checkpoint
        if self.has_skip_conv:
            x = self.Conv_2(x)
        return (x + h) / np.sqrt(2.0)


# -- the network ----------------------------------------------------------------


class NCSNpp(nn.Module):
    """NCSN++ score network: ``forward(x in [0,1], sigma) -> score``."""

    def __init__(self, config: NCSNppConfig):
        super().__init__()
        self.config = config
        nf, ch_mult, n_blocks = config.nf, config.ch_mult, config.num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.all_resolutions = [config.image_size // (2 ** i) for i in range(len(ch_mult))]
        if self.all_resolutions[-1] * 2 ** (len(ch_mult) - 1) != config.image_size:
            raise ValueError(
                f"image_size {config.image_size} does not halve cleanly {len(ch_mult) - 1} "
                "times; this checkpoint's working resolution is fixed"
            )
        fir = tuple(config.fir_kernel)
        temb_dim = nf * 4
        channels = config.num_channels

        def block(**kw):
            return ResnetBlockBigGANpp(temb_dim=temb_dim, fir_kernel=fir, **kw)

        modules: list[nn.Module] = [
            GaussianFourierProjection(embedding_size=nf, scale=config.fourier_scale),
            nn.Linear(2 * nf, temb_dim),
            nn.Linear(temb_dim, temb_dim),
            conv3x3(channels, nf),
        ]
        hs_c = [nf]
        in_ch = nf
        for i_level in range(self.num_resolutions):
            for _ in range(n_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(block(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.all_resolutions[i_level] in config.attn_resolutions:
                    modules.append(AttnBlockpp(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != self.num_resolutions - 1:
                modules.append(block(down=True, in_ch=in_ch))
                modules.append(Combine(dim1=channels, dim2=in_ch))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules += [block(in_ch=in_ch), AttnBlockpp(channels=in_ch), block(in_ch=in_ch)]

        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(n_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(block(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if self.all_resolutions[i_level] in config.attn_resolutions:
                modules.append(AttnBlockpp(channels=in_ch))
            modules.append(group_norm(in_ch))
            modules.append(conv3x3(in_ch, channels))
            if i_level != 0:
                modules.append(block(in_ch=in_ch, up=True))
        assert not hs_c
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """``x`` is ``(B, C, H, W)`` in [0, 1]; ``sigma`` is ``(B,)``."""
        cfg, m = self.config, self.all_modules
        i = 0
        temb = m[i](torch.log(sigma)); i += 1
        temb = m[i](temb); i += 1
        temb = m[i](F.silu(temb)); i += 1

        x = 2 * x - 1.0  # trained with data.centered = False
        input_pyramid = x
        hs = [m[i](x)]; i += 1
        for i_level in range(self.num_resolutions):
            for _ in range(cfg.num_res_blocks):
                h = m[i](hs[-1], temb); i += 1
                if h.shape[-1] in cfg.attn_resolutions:
                    h = m[i](h); i += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = m[i](hs[-1], temb); i += 1
                input_pyramid = downsample_2d(input_pyramid, cfg.fir_kernel)
                h = m[i](input_pyramid, h); i += 1
                hs.append(h)

        h = hs[-1]
        h = m[i](h, temb); i += 1
        h = m[i](h); i += 1
        h = m[i](h, temb); i += 1

        pyramid = None
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(cfg.num_res_blocks + 1):
                h = m[i](torch.cat([h, hs.pop()], dim=1), temb); i += 1
            if h.shape[-1] in cfg.attn_resolutions:
                h = m[i](h); i += 1
            pyramid_h = m[i + 1](F.silu(m[i](h))); i += 2
            pyramid = pyramid_h if pyramid is None else (
                upsample_2d(pyramid, cfg.fir_kernel) + pyramid_h)
            if i_level != 0:
                h = m[i](h, temb); i += 1
        assert not hs and i == len(m)
        return pyramid / sigma.reshape(-1, *([1] * (x.dim() - 1)))


def sigma_grid(config: NCSNppConfig, n: int) -> Tensor:
    """The VE-SDE's geometric noise levels, ``sigma_max`` down to ``sigma_min``."""
    return torch.exp(torch.linspace(math.log(config.sigma_max), math.log(config.sigma_min), n))
