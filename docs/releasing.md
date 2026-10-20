# Preparing a grasp-kit release

The prepared stable version is **3.4.0**. Its single source is
`src/graspkit/version.py`; Hatchling reads it through `dynamic = ["version"]`.
These instructions build and validate files locally. They do not upload to
TestPyPI or PyPI, create tags, commit, or push.

## Local build in the paired workspace

Set up or refresh the shared environment from `graspkit-tools/` with
`uv sync --extra cpu` (`--extra gpu` on a CUDA host). Then, from `graspkit/`:

```bash
source ../graspkit-tools/.venv/bin/activate
python build_package.py --clean
```

The helper uses the current interpreter, builds a source archive and a wheel
from that archive, runs `twine check --strict`, checks the file manifest and
metadata, and runs a wheel smoke test. Build isolation supplies Hatchling.
It never installs tools into the runtime or copies files to the Tools repo.
The obsolete `--dev` option is removed; edit the version source deliberately.
`--clean` removes only matching grasp-kit archives from the selected output
folder. Unknown files cause an error instead of being deleted.

Output: `dist/grasp_kit-3.4.0.tar.gz` and
`dist/grasp_kit-3.4.0-py3-none-any.whl`. An optional `--outdir PATH` selects an
empty output folder. The shell/batch wrappers select the shared Tools Python.

Equivalent individual checks:

```bash
python -m build
python -m twine check --strict dist/*
python scripts/check_release.py dist --smoke
python -m pytest tests/
```

Do not reuse a directory containing artifacts from other versions. The checker
prints SHA-256 hashes and rejects mixed versions, missing metadata/licenses,
unexpected archive contents and absent typing markers. The smoke test starts
a separate interpreter from a temporary directory, loads the extracted wheel,
checks its import origin and version, selects CSFs and renders a plot. Locally,
runtime dependencies come from Tools; this is not a clean dependency-resolution
test. The CI workflow installs the wheel and dependencies on a fresh runner.

## Files and license

The wheel contains `graspkit` and `graspkit_plot`. The source archive includes
those sources, metadata, GPLv3 text and selected installation/build docs.
Research notes, agent instructions, tests, local data and historical reports
are excluded. Full regression tests remain in the development repository.
The existing GPLv3 license is recorded as `GPL-3.0-only`.

## Build-only CI

`.github/workflows/build-release.yml` runs by manual dispatch or by an explicit
push to a `codex/release-build-check-*` branch. It builds
and checks on Linux, Windows and macOS using Python 3.14, and stores GitHub
Actions artifacts. It has no PyPI credentials, OIDC publishing permission,
publish job or tag-triggered upload. CI uses an ephemeral standalone runner;
it does not create a second runtime in this paired local workspace.
A universal wheel still relies on its dependencies supporting the target OS.
Python versions above 3.14 are permitted by metadata but not yet in this matrix.

Until the workflow reaches the default branch, start a check from the release
commit by pushing it to a dedicated branch, for example:

```bash
git push origin HEAD:refs/heads/codex/release-build-check-20261020
```

Choose a fresh suffix for a separate run of an unchanged commit. Normal
development-branch pushes and tags do not trigger this workflow. Once the
workflow is on the default branch, the Actions page also offers **Run workflow**
with a branch selector. Review all three platform jobs before using the artifacts.

## Before a later publication

1. Review the exact source commit and release changes; rerun the workflow on it.
2. Confirm PyPI ownership/availability of `grasp-kit`, and that version 3.4.0
   has not already been published. A missing public project page alone does
   not prove that a name is available.
3. Inspect the actual archives and hashes. Publishing makes their contents
   public even if the Git repository remains private. Repository URLs in the
   metadata require access to that repository; provide accessible public docs
   or change visibility deliberately if external users need them.
4. Test installation from TestPyPI only when an upload is explicitly authorized.
5. Configure a separate publishing workflow with PyPI Trusted Publishing when
   ready. Keep the build job separate and publish only reviewed artifacts.
6. Tag the approved commit and record the version/release notes. A later commit
   request also covers the parent gitlink update; this preparation creates no tag.

References: [PyPA packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
