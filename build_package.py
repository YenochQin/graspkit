"""Build and check distributions locally; never publish or change the environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=root / "dist")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous grasp-kit archives in outdir",
    )
    args = parser.parse_args()
    outdir: Path = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for pattern in ("grasp_kit-*.whl", "grasp_kit-*.tar.gz"):
            for archive in outdir.glob(pattern):
                archive.unlink()
    if any(outdir.iterdir()):
        parser.error("Output directory must be empty; use --clean or a new --outdir.")

    # Build's default creates the wheel from the sdist, checking source completeness.
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(outdir), str(root)],
        check=True,
    )
    archives = sorted(outdir.glob("*.whl")) + sorted(outdir.glob("*.tar.gz"))
    subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", *map(str, archives)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_release.py"),
            str(outdir),
            "--smoke",
        ],
        check=True,
    )
    print(f"Validated local distributions: {outdir}")


if __name__ == "__main__":
    main()
