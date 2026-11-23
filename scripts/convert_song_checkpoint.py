"""Convert one of Song et al.'s Flax NCSN++ checkpoints to a PyTorch state dict (D79, T2.10).

    pip install msgpack            # this script only; loading the output needs torch alone
    python scripts/convert_song_checkpoint.py \
        checkpoints/song_inverse_problems/brats_ncsnpp_ve/checkpoint_26 \
        --out checkpoints/song_inverse_problems/brats_ncsnpp_ve/ncsnpp_brats240.pt

The checkpoints (github.com/yang-song/score_inverse_problems, Google Drive) are Flax
``msgpack`` files. They are read here with ``msgpack`` and numpy only: no JAX, no flax, and
-- unlike ``torch.load`` on a pickle -- nothing in the file can execute code. The EMA
weights (``params_ema``) are the ones the paper samples with, so those are converted.

**The mapping is by call order, per layer class.** Flax names a layer ``<Class>_<n>``, with
``n`` counting that class's calls in the forward pass. ``NCSNpp.all_modules`` is built in the
same forward order, so the k-th torch module of a type is the Flax layer ``<Class>_k``.
Every Flax tensor must be consumed exactly once and every torch tensor filled with the same
number of values, or the conversion is refused. That rules out the silent failure here: a
plausible-looking network with two layers swapped.

Layout changes: Flax convolutions are ``(kh, kw, in, out)`` and torch ``(out, in, kh, kw)``;
Flax dense kernels are ``(in, out)`` and torch ``(out, in)``; Flax stores GroupNorm's scale
and bias as ``(1, 1, 1, C)``. ``NIN`` and the Fourier ``W`` have the same layout in both.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from caliber.solvers.ncsnpp import (
    NIN,
    AttnBlockpp,
    Combine,
    GaussianFourierProjection,
    NCSNpp,
    NCSNppConfig,
    ResnetBlockBigGANpp,
)

#: torch module type -> the Flax class name it was saved under.
FLAX_CLASS = {
    GaussianFourierProjection: "GaussianFourierProjection",
    nn.Linear: "Dense",
    nn.Conv2d: "Conv",
    nn.GroupNorm: "GroupNorm",
    ResnetBlockBigGANpp: "ResnetBlockBigGANpp",
    AttnBlockpp: "AttnBlockpp",
    Combine: "Combine",
}

#: Song et al.'s three configurations (configs/ve/*_ncsnpp_continuous.py).
CONFIGS = {
    "brats240": NCSNppConfig(image_size=240),
    "lidc320": NCSNppConfig(image_size=320, ch_mult=(1, 1, 2, 2, 2, 2, 2),
                            attn_resolutions=(20,)),
    "ldct512": NCSNppConfig(image_size=512, nf=16, ch_mult=(1, 2, 4, 8, 16, 32, 32),
                            num_res_blocks=1, attn_resolutions=(16,), sigma_max=220.0),
}


def read_flax_msgpack(path: str | Path) -> dict:
    """A Flax checkpoint as nested dicts of numpy arrays, without JAX or flax."""
    import msgpack

    def ext_hook(code, data):
        if code == 1:  # flax's ndarray extension: (shape, dtype name, raw bytes)
            shape, dtype, buf = msgpack.unpackb(data, raw=False)
            return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()
        if code == 3:  # numpy scalar
            dtype, buf = msgpack.unpackb(data, raw=False)
            return np.frombuffer(buf, dtype=np.dtype(dtype))[0]
        return msgpack.ExtType(code, data)

    with open(path, "rb") as fh:
        return msgpack.unpackb(fh.read(), ext_hook=ext_hook, raw=False, strict_map_key=False)


def _leaf(module: nn.Module, name: str, flax: dict) -> np.ndarray:
    """The Flax tensor for ``module.<name>``, in torch's layout."""
    if isinstance(module, nn.Conv2d):
        return flax["kernel"].transpose(3, 2, 0, 1) if name == "weight" else flax["bias"]
    if isinstance(module, nn.Linear):
        return flax["kernel"].T if name == "weight" else flax["bias"]
    if isinstance(module, nn.GroupNorm):
        return (flax["scale"] if name == "weight" else flax["bias"]).reshape(-1)
    if isinstance(module, (NIN, GaussianFourierProjection)):
        return flax[name]
    raise TypeError(f"no layout rule for {type(module).__name__}.{name}")


def _fill(module: nn.Module, flax: dict, prefix: str, used: set, out: dict) -> None:
    """Fill ``module``'s own tensors from ``flax``, then recurse into named children."""
    for name, _ in module.named_parameters(recurse=False):
        out[f"{prefix}{name}"] = _leaf(module, name, flax)
        used.add(f"{prefix}{name}")
    for child_name, child in module.named_children():
        if child_name not in flax:
            raise KeyError(f"{prefix}{child_name}: no Flax layer of that name")
        _fill(child, flax[child_name], f"{prefix}{child_name}.", used, out)


def _has_arrays(tree) -> bool:
    if isinstance(tree, dict):
        return any(_has_arrays(v) for v in tree.values())
    return True


def convert(params: dict, config: NCSNppConfig) -> dict[str, torch.Tensor]:
    """A torch state dict for ``NCSNpp(config)`` from a Flax ``params_ema`` tree."""
    model = NCSNpp(config)
    counters: dict[str, int] = {}
    arrays: dict[str, np.ndarray] = {}
    used: set[str] = set()
    consumed: set[str] = set()
    for idx, module in enumerate(model.all_modules):
        cls = FLAX_CLASS.get(type(module))
        if cls is None:
            raise TypeError(f"all_modules[{idx}] is a {type(module).__name__}, with no Flax class")
        k = counters.get(cls, 0)
        counters[cls] = k + 1
        key = f"{cls}_{k}"
        if key not in params:
            raise KeyError(f"all_modules[{idx}] wants {key}, which the checkpoint does not have")
        consumed.add(key)
        _fill(module, params[key], f"all_modules.{idx}.", used, arrays)

    # Parameter-free layers (the pyramids' FIR Downsample/Upsample) are serialised as empty
    # dicts by some Flax versions -- the LIDC checkpoint has them, the BraTS one does not.
    # They carry nothing to map; any layer that does carry a tensor must be mapped.
    unused = sorted(k for k in set(params) - consumed if _has_arrays(params[k]))
    if unused:
        raise ValueError(f"checkpoint layers never mapped: {unused[:5]}{'...' if len(unused) > 5 else ''}")
    state = model.state_dict()
    missing = sorted(k for k in state if k not in arrays)
    if missing:
        raise ValueError(f"torch tensors never filled: {missing[:5]}")
    out = {}
    for name, ref in state.items():
        arr = np.ascontiguousarray(arrays[name], dtype=np.float32)
        if tuple(arr.shape) != tuple(ref.shape):
            raise ValueError(f"{name}: checkpoint {arr.shape} against model {tuple(ref.shape)}")
        out[name] = torch.from_numpy(arr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="a Flax checkpoint file, e.g. .../checkpoint_26")
    ap.add_argument("--config", default="brats240", choices=sorted(CONFIGS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    raw = Path(args.checkpoint).read_bytes()
    state = read_flax_msgpack(args.checkpoint)
    if "params_ema" not in state:
        print(f"[convert] {args.checkpoint} has no params_ema; keys {sorted(state)}",
              file=sys.stderr)
        return 1
    config = CONFIGS[args.config]
    weights = convert(state["params_ema"], config)
    n = sum(t.numel() for t in weights.values())
    torch.save({
        "config": config.to_dict(),
        "state_dict": weights,
        "source": {"file": Path(args.checkpoint).name,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "step": int(np.asarray(state.get("step", -1))),
                   "weights": "params_ema"},
    }, args.out)
    print(f"[convert] {len(weights)} tensors, {n:,} values -> {args.out}")
    print(f"[convert] source sha256 {hashlib.sha256(raw).hexdigest()}, "
          f"step {int(np.asarray(state.get('step', -1)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
