"""Check release archives and exercise the compatible wheel without installing it.

Dependencies: packaging plus the project's runtime dependencies for --smoke.
Run with the shared Tools interpreter in the workspace.
"""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import fnmatch
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

NAME = "grasp-kit"
LICENSE = "GPL-3.0-only"
WHEEL_ROOTS = {"graspkit", "graspkit_plot"}
WHEEL_REQUIRED = {
    "graspkit/__init__.py",
    "graspkit/version.py",
    "graspkit/py.typed",
    "graspkit_plot/__init__.py",
}
SDIST_PATTERNS = {
    ".gitignore",  # Hatchling includes its VCS exclusion rules in source archives.
    "src/graspkit/*.py",
    "src/graspkit/py.typed",
    "src/graspkit_plot/*.py",
    "pyproject.toml",
    "PKG-INFO",
    "LICENSE",
    "README.md",
    "README_zh.md",
    "INSTALL.md",
    "docs/releasing.md",
    "build_package.py",
    "scripts/check_release.py",
}
SDIST_REQUIRED = {
    "pyproject.toml",
    "LICENSE",
    "README.md",
    "src/graspkit/version.py",
    "src/graspkit/py.typed",
    "scripts/check_release.py",
}


def check_metadata(raw: bytes) -> str:
    metadata = BytesParser(policy=policy.default).parsebytes(raw)
    assert metadata["Name"] == NAME, metadata["Name"]
    version = Version(str(metadata["Version"]))
    assert not version.is_prerelease and not version.is_devrelease, version
    assert str(metadata["Requires-Python"]).replace(" ", "") == ">=3.14"
    assert metadata["License-Expression"] == LICENSE
    assert metadata["Author-email"] and metadata["Description-Content-Type"]
    assert len(str(metadata.get_payload()).strip()) > 100, "Missing project description"
    assert "Development Status :: 5 - Production/Stable" in metadata.get_all(
        "Classifier", []
    )
    assert metadata.get_all("License-File"), "Missing license files"
    for requirement in metadata.get_all("Requires-Dist", []):
        assert " @ " not in requirement, f"Non-index dependency: {requirement}"
    return str(version)


def check_archive(path: Path) -> tuple[str, bool]:
    if path.suffix == ".whl":
        _, filename_version, _, tags = parse_wheel_filename(path.name)
        assert all(not tag.platform.startswith("linux_") for tag in tags), (
            "Plain linux wheels are not PyPI release artifacts; build manylinux/musllinux."
        )
        with zipfile.ZipFile(path) as archive:
            names = {name for name in archive.namelist() if not name.endswith("/")}
            assert WHEEL_REQUIRED <= names, WHEEL_REQUIRED - names
            assert all(
                PurePosixPath(name).parts[0] in WHEEL_ROOTS
                or PurePosixPath(name).parts[0].endswith(".dist-info")
                for name in names
            ), f"Unexpected wheel contents: {names}"
            assert not any(
                "__pycache__" in name or name.endswith((".pyc", ".pdb"))
                for name in names
            )
            (metadata_name,) = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            version = check_metadata(archive.read(metadata_name))
            assert Version(version) == filename_version, (
                "Filename/metadata version mismatch"
            )
            metadata = BytesParser(policy=policy.default).parsebytes(
                archive.read(metadata_name)
            )
            dist_info = metadata_name.rsplit("/", 1)[0]
            for license_file in metadata.get_all("License-File", []):
                assert f"{dist_info}/licenses/{license_file}" in names, license_file
            if NAME == "rcsfs":
                binaries = [
                    name
                    for name in names
                    if name.startswith("rcsfs/_rcsfs")
                    and name.endswith((".so", ".pyd"))
                ]
                assert len(binaries) == 1, (
                    f"Expected only the built native extension: {binaries}"
                )
                expected_suffix = (
                    ".pyd"
                    if all(tag.platform.startswith("win") for tag in tags)
                    else ".so"
                )
                assert binaries[0].endswith(expected_suffix), binaries
                assert f"{dist_info}/entry_points.txt" in names
        return version, bool(tags & set(sys_tags()))

    with tarfile.open(path, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in files}
        assert len(roots) == 1, roots
        names = {
            str(PurePosixPath(member.name).relative_to(next(iter(roots))))
            for member in files
        }
        unexpected = {
            name
            for name in names
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in SDIST_PATTERNS)
        }
        assert not unexpected, (
            f"Unexpected source archive contents: {sorted(unexpected)}"
        )
        assert SDIST_REQUIRED <= names, SDIST_REQUIRED - names
        assert not any(
            "__pycache__" in name or name.endswith((".pyc", ".so", ".pyd", ".pdb"))
            for name in names
        )
        (metadata_member,) = [
            member for member in files if member.name == f"{next(iter(roots))}/PKG-INFO"
        ]
        stream = archive.extractfile(metadata_member)
        assert stream is not None
        version = check_metadata(stream.read())
    return version, False


def smoke(unpacked: Path) -> None:
    sys.path.insert(0, str(unpacked))
    from importlib.metadata import version
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import graspkit
    import graspkit_plot
    from graspkit.data_IO import EnergyFileLoader, MixCoefLoader
    from graspkit.CSFs_processor import select_csf_indices_above_ci_squared_cutoff
    from graspkit_plot import configure_matplotlib_for_publication
    import matplotlib.pyplot as plt

    assert Path(graspkit.__file__).is_relative_to(unpacked)
    assert Path(graspkit_plot.__file__).is_relative_to(unpacked)
    assert graspkit.__version__ == version(NAME)
    assert EnergyFileLoader and MixCoefLoader
    selected = select_csf_indices_above_ci_squared_cutoff(
        np.array([0.8, 0.1, 0.5]), ci_squared_cutoff=0.1
    )
    assert selected.tolist() == [0, 2]
    configure_matplotlib_for_publication()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(unpacked / "smoke.png")
    plt.close(fig)
    assert (unpacked / "smoke.png").stat().st_size > 0
    print(f"Wheel imports, CSF selection and plotting passed: {graspkit.__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--unpacked", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.unpacked:
        smoke(args.directory.resolve())
        return
    archives = sorted(args.directory.glob("*.whl")) + sorted(
        args.directory.glob("*.tar.gz")
    )
    assert archives, f"No release archives in {args.directory}"
    versions: set[str] = set()
    compatible: list[Path] = []
    for path in archives:
        version, can_import = check_archive(path)
        versions.add(version)
        if can_import:
            compatible.append(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"{digest}  {path.name}")
    assert len(versions) == 1, f"Mixed release versions: {versions}"
    if args.smoke:
        assert compatible, "No wheel compatible with this interpreter/platform"
        for wheel in compatible:
            with tempfile.TemporaryDirectory(prefix="release-smoke-") as temporary:
                unpacked = Path(temporary)
                with zipfile.ZipFile(wheel) as archive:
                    archive.extractall(unpacked)
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(Path(__file__).resolve()),
                        str(unpacked),
                        "--unpacked",
                    ],
                    cwd=unpacked,
                    check=True,
                )
    print(f"Validated {len(archives)} archive(s) for {NAME} {next(iter(versions))}.")


if __name__ == "__main__":
    main()
