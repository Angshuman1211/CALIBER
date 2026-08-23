"""Reproducibly fetch every openly-available resource this project needs (T2.1).

    python scripts/fetch_data.py --list
    python scripts/fetch_data.py fastmri_plus
    python scripts/fetch_data.py --all
    python scripts/fetch_data.py m4raw_val --extract

Downloads resume on interruption, so a dropped connection costs only the
remainder. Nothing here is destructive: an existing complete file is skipped.

**What this script deliberately cannot do.** fastMRI requires a signed Data
Sharing Agreement and an approved application naming your institution and
intended use. That is a legal agreement between you and NYU, it cannot be
automated, and third-party mirrors would breach it. ``--list`` prints the
application URL and marks the resource MANUAL.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Resource:
    key: str
    description: str
    dest: str
    url: str | None = None
    extract_to: str | None = None
    approx_mb: int = 0
    licence: str = ""
    manual_url: str | None = None
    needed_for: str = ""
    files: list[tuple[str, str]] = field(default_factory=list)  # (url, relative dest)

    @property
    def is_manual(self) -> bool:
        return self.url is None and not self.files


RESOURCES: list[Resource] = [
    Resource(
        key="fastmri",
        description="fastMRI raw k-space [28] - primary empirical substrate",
        dest="data/raw/fastmri",
        manual_url="https://fastmri.med.nyu.edu/",
        licence="NYU Data Sharing Agreement - application required, no redistribution",
        needed_for="Week 3 (T2.1). The critical path.",
        approx_mb=7000,
    ),
    Resource(
        key="fastmri_plus",
        description="fastMRI+ pathology annotations [29]",
        dest="data/annotations/fastmri_plus",
        licence="MIT",
        needed_for="Week 3 - region-level labels for MRI flag evaluation",
        approx_mb=2,
        files=[
            (
                "https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations/knee.csv",
                "knee.csv",
            ),
            (
                "https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations/brain.csv",
                "brain.csv",
            ),
            (
                "https://raw.githubusercontent.com/microsoft/fastmri-plus/main/"
                "Annotations/knee_file_list.csv",
                "knee_file_list.csv",
            ),
            (
                "https://raw.githubusercontent.com/microsoft/fastmri-plus/main/"
                "Annotations/brain_file_list.csv",
                "brain_file_list.csv",
            ),
            (
                "https://raw.githubusercontent.com/microsoft/fastmri-plus/main/LICENSE",
                "LICENSE",
            ),
        ],
    ),
    Resource(
        key="m4raw_val",
        description="M4Raw validation - open raw k-space MRI, fastMRI-compatible .h5",
        dest="data/archives/m4raw",
        url=(
            "https://zenodo.org/api/records/7523691/files/"
            "M4RawV1.1_multicoil_val.zip/content"
        ),
        extract_to="data/raw/m4raw",
        licence="CC-BY-4.0 - open, no application",
        needed_for="Weeks 2-3 stopgap while fastMRI access is pending",
        approx_mb=2806,
    ),
    Resource(
        key="m4raw_train",
        description="M4Raw train (1024 volumes) - only if more calibration data is needed",
        dest="data/archives/m4raw",
        url=(
            "https://zenodo.org/api/records/7523691/files/"
            "M4RawV1.1_multicoil_train.zip/content"
        ),
        extract_to="data/raw/m4raw",
        licence="CC-BY-4.0",
        needed_for="optional",
        approx_mb=11948,
    ),
    Resource(
        key="ddpm_celebahq",
        description="google/ddpm-celebahq-256 - unconditional diffusion prior",
        dest="checkpoints/ddpm-celebahq-256",
        licence="Apache-2.0",
        needed_for="Week 2 DPS/DDNM validation; Week 10 natural-image appendix",
        approx_mb=435,
        files=[
            (
                "https://huggingface.co/google/ddpm-celebahq-256/resolve/main/config.json",
                "config.json",
            ),
            (
                "https://huggingface.co/google/ddpm-celebahq-256/resolve/main/"
                "scheduler/scheduler_config.json",
                "scheduler/scheduler_config.json",
            ),
            (
                "https://huggingface.co/google/ddpm-celebahq-256/resolve/main/"
                "diffusion_pytorch_model.bin",
                "diffusion_pytorch_model.bin",
            ),
        ],
    ),
    Resource(
        key="score_medical",
        description="Song et al. [27] score priors for knee MRI and CT",
        dest="checkpoints/score_inverse_problems",
        manual_url="https://github.com/yang-song/score_inverse_problems",
        licence=(
            "Checkpoints on Google Drive. The MRI model is fastMRI-derived, so the "
            "NYU agreement plausibly applies to it too - confirm before use or release."
        ),
        needed_for="Week 2 (T2.2) - the medical generative priors",
        approx_mb=2000,
    ),
    Resource(
        key="lodopab",
        description="LoDoPaB-CT ground truth [30]",
        dest="data/archives/lodopab",
        manual_url="https://zenodo.org/records/3384092",
        licence="CC-BY",
        needed_for="Week 7 - ALREADY DOWNLOADED, see data/README.md",
        approx_mb=19100,
    ),
    Resource(
        key="totalsegmentator",
        description="TotalSegmentator [31] CT volumes + 117 organ masks",
        dest="data/archives/totalsegmentator",
        manual_url="https://zenodo.org/records/10047292",
        licence="CC-BY-4.0",
        needed_for="Week 7 - ALREADY DOWNLOADED, see data/README.md and D-013",
        approx_mb=23600,
    ),
]

BY_KEY = {r.key: r for r in RESOURCES}


def curl(url: str, out: Path) -> bool:
    """Download with resume and retries. Returns True on success."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-L", "-C", "-",
        "--retry", "10", "--retry-delay", "5", "--retry-all-errors",
        "--speed-limit", "1024", "--speed-time", "60",
        "--progress-bar",
        "-o", str(out), url,
    ]
    print(f"  -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return subprocess.run(cmd, check=False).returncode == 0


def verify_zip(path: Path, deep: bool = False) -> str | None:
    """Return a human-readable problem with ``path``, or None if it is sound.

    A truncated download is the failure mode that costs the most time, because
    it does not look like a failure: the file is present, it is large, and its
    first bytes are a valid zip header. Only the central directory at the *end*
    is missing, so nothing notices until an extraction is attempted - possibly
    weeks later. This is exactly what happened to the M4Raw archive, which sat
    recorded as fetched at 0.8 GB of 2.8 GB.

    Opening the archive is enough to catch that, and costs one seek: the
    central directory is the part a truncated file has lost. ``deep`` adds a
    CRC check of every member, which catches corruption rather than truncation
    but has to decompress the whole archive - minutes per archive here, since
    these run to 23 GB. Not the default for that reason.
    """
    if not path.exists():
        return "missing"
    try:
        with zipfile.ZipFile(path) as zf:
            if deep:
                broken = zf.testzip()
                if broken:
                    return f"corrupt member: {broken}"
        return None
    except zipfile.BadZipFile:
        return (
            f"not a readable zip ({path.stat().st_size / 1e9:.2f} GB present) - "
            "almost certainly a truncated download; re-run to resume"
        )


def fetch(res: Resource, extract: bool = False) -> bool:
    print(f"\n=== {res.key} - {res.description}")
    print(f"    licence: {res.licence}")

    if res.is_manual:
        print(f"    MANUAL: this resource cannot be scripted.\n    {res.manual_url}")
        print(f"    needed for: {res.needed_for}")
        return False

    ok = True
    if res.files:
        for url, rel in res.files:
            target = REPO / res.dest / rel
            if target.exists() and target.stat().st_size > 0:
                print(f"  skip (present): {rel}")
                continue
            ok &= curl(url, target)
    elif res.url:
        parts = res.url.rstrip("/").split("/")
        name = parts[-2] if res.url.endswith("/content") else "download.zip"
        target = REPO / res.dest / name
        ok &= curl(res.url, target)

        # Verify before extracting, and report a bad archive as a failure of
        # the whole resource. Otherwise a partial download is silently treated
        # as success and the status board records something that is not there.
        problem = verify_zip(target)
        if problem:
            print(f"  ARCHIVE PROBLEM: {problem}", file=sys.stderr)
            return False
        print(f"  verified: {target.name} ({target.stat().st_size / 1e9:.2f} GB)")

        if ok and extract and res.extract_to:
            dest = REPO / res.extract_to
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  extracting to {res.extract_to} ...")
            with zipfile.ZipFile(target) as zf:
                zf.extractall(dest)
            print("  extracted.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch project data (T2.1)")
    ap.add_argument("keys", nargs="*", help="resource keys to fetch")
    ap.add_argument("--all", action="store_true", help="fetch everything scriptable")
    ap.add_argument("--list", action="store_true", help="show all resources and exit")
    ap.add_argument("--extract", action="store_true", help="unzip archives after download")
    ap.add_argument(
        "--verify", action="store_true",
        help="check every archive already on disk for truncation, then exit",
    )
    ap.add_argument(
        "--deep", action="store_true",
        help="with --verify, also CRC-check every member (slow: decompresses everything)",
    )
    args = ap.parse_args()

    if args.verify:
        problems = 0
        for archive in sorted((REPO / "data" / "archives").rglob("*.zip")):
            problem = verify_zip(archive, deep=args.deep)
            rel = archive.relative_to(REPO)
            size = archive.stat().st_size / 1e9
            if problem:
                problems += 1
                print(f"BAD   {rel}  ({size:.2f} GB)  {problem}")
            else:
                print(f"ok    {rel}  ({size:.2f} GB)")
        print(f"\n{problems} problem(s).")
        return 1 if problems else 0

    if args.list or (not args.keys and not args.all):
        print(f"{'key':<18} {'size':>9}  {'auto':<5} needed for")
        print("-" * 88)
        for r in RESOURCES:
            size = f"{r.approx_mb / 1024:.1f} GB" if r.approx_mb >= 1024 else f"{r.approx_mb} MB"
            print(f"{r.key:<18} {size:>9}  {'no' if r.is_manual else 'yes':<5} {r.needed_for}")
        print("\nfastMRI is the critical path and must be applied for by hand:")
        print("  https://fastmri.med.nyu.edu/")
        return 0

    keys = [r.key for r in RESOURCES if not r.is_manual] if args.all else args.keys
    unknown = [k for k in keys if k not in BY_KEY]
    if unknown:
        print(f"unknown resource(s): {unknown}\navailable: {sorted(BY_KEY)}", file=sys.stderr)
        return 2

    failed = [k for k in keys if not fetch(BY_KEY[k], extract=args.extract)]
    if failed:
        print(f"\nnot completed automatically: {failed}", file=sys.stderr)
        return 1
    print("\nall requested resources present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
