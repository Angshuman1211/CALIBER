"""Will this run fit in memory? — the preflight check (P-24, D-041).

P-24: on this project's CPU-only machine, a solver pass whose allocation does not fit
does not always raise. Sometimes torch raises ``RuntimeError: DefaultCPUAllocator: not
enough memory``; sometimes the process simply segfaults, with no traceback and no output.
A run that dies that way at hour seven looks, in the log, like a job that stopped.

Checkpointing (D-038) and resume (D-040) limit what such a death costs. This module is
the cheaper half: refuse to start a run that cannot finish, before any time is spent.

**The estimate is measured, not derived.** Peak commit charge of one UNet pass, on
``ddpm-celebahq-256`` at ``torch 2.13.0+cpu``:

===========  =======  =============  =============
resolution     draws   peak commit    model says
===========  =======  =============  =============
64²                1        2.75 GB        2.80 GB
96²                1        2.79 GB        2.80 GB
128²               1        3.00 GB        3.00 GB
160²               1        3.22 GB        3.56 GB
192²               1        4.20 GB        4.24 GB
256²               1        5.99 GB        5.98 GB
96²                4        4.14 GB        4.24 GB
128²               2        4.09 GB        4.00 GB
128²               4        5.98 GB        5.98 GB
128²               8      segfault        10.0 GB   (7.6 GB was free)
256²               4      out of memory   18.0 GB   (~5 GB was free)
===========  =======  =============  =============

Linear in pixels × draws above ~128², with a floor below it, and it called both failures.
It is a **fit to one machine, one checkpoint and one torch build**, so it is a guard rail,
not a law: the escape hatch below exists for when it is wrong.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
from typing import Any

# Fitted on the table above. BASE covers the interpreter, torch, and the loaded weights;
# PER_PIXEL_PER_DRAW is the activation cost that scales with the work.
BASE_GB = 2.0
PER_PIXEL_PER_DRAW_GB = 6.1e-5
#: Nothing observed below this, whatever the arithmetic says.
FLOOR_GB = 2.8
#: A backward pass (DPS) added ~0.4 GB at 128²; carried as a flat term rather than a
#: second slope, because it was measured at one size only.
BACKWARD_EXTRA_GB = 0.4


def estimate_peak_commit_gb(
    image_shape: tuple[int, int], n_draws: int, backward: bool = False
) -> float:
    """Peak committed memory one solver pass is expected to need, in GB."""
    pixels = int(image_shape[0]) * int(image_shape[1])
    estimate = BASE_GB + PER_PIXEL_PER_DRAW_GB * pixels * max(1, int(n_draws))
    if backward:
        estimate += BACKWARD_EXTRA_GB
    return max(FLOOR_GB, estimate)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wt.DWORD),
        ("dwMemoryLoad", wt.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_commit_gb() -> float | None:
    """Commit charge still available, in GB, or ``None`` where it cannot be determined.

    Commit rather than free RAM, because that is the limit an allocation actually hits:
    the failures in P-24 happened with several GB of physical memory free.

    Returns ``None`` off Windows and whenever the query fails, and every caller treats
    that as "unknown" and proceeds. A memory check that cannot read the memory must not
    block a run.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        kernel32.GlobalMemoryStatusEx.restype = wt.BOOL
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPageFile / 1e9
    except (OSError, AttributeError, ValueError):
        return None


def memory_preflight(
    image_shape: tuple[int, int], n_draws: int, backward: bool = False
) -> dict[str, Any]:
    """Report whether a run of this shape is expected to fit.

    ``verdict`` is one of:

    * ``"unknown"``  — available memory could not be read; proceed.
    * ``"ok"``       — the estimate fits with room to spare.
    * ``"tight"``    — it fits, but within 20%; another application growing could take it.
    * ``"too_big"``  — the estimate exceeds what is available; the run is expected to die.
    """
    needed = estimate_peak_commit_gb(image_shape, n_draws, backward)
    available = available_commit_gb()
    if available is None:
        verdict = "unknown"
    elif needed > available:
        verdict = "too_big"
    elif needed > 0.8 * available:
        verdict = "tight"
    else:
        verdict = "ok"
    return {
        "needed_gb": round(needed, 2),
        "available_gb": None if available is None else round(available, 2),
        "verdict": verdict,
    }
