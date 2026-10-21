# Preparing a graspkit-atomic release

The prepared stable version is **3.4.1**. Its single source is
`src/graspkit/version.py`; Hatchling reads it through `dynamic = ["version"]`.
The local build commands below do not upload files or create Git tags.
The separate manual publishing workflow is described near the end of this guide.

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
`--clean` removes only matching graspkit-atomic archives from the selected output
folder. Unknown files cause an error instead of being deleted.

Output: `dist/graspkit_atomic-3.4.1.tar.gz` and
`dist/graspkit_atomic-3.4.1-py3-none-any.whl`. An optional `--outdir PATH` selects an
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
push to a `codex/release-build-check-*` branch. It is also reusable through
`workflow_call`, so the publishing workflow builds the same commit. It builds
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
2. Confirm PyPI ownership/availability of `graspkit-atomic`, and that version 3.4.1
   has not already been published. A missing public project page alone does
   not prove that a name is available.
3. Inspect the actual archives and hashes. Publishing makes their contents
   public even if the Git repository remains private. Repository URLs in the
   metadata require access to that repository; provide accessible public docs
   or change visibility deliberately if external users need them.
4. Test installation from TestPyPI only when an upload is explicitly authorized.
5. Configure the Trusted Publisher described below and run the manual
   publishing workflow only when the upload is authorized.
6. Tag the approved commit and record the version/release notes. A later commit
   request also covers the parent gitlink update; this preparation creates no tag.

References: [PyPA packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

## Manual PyPI publishing

`.github/workflows/publish.yml` is named **Publish to PyPI** in GitHub Actions.
It has only a `workflow_dispatch` trigger. Pushing commits, tags, or completing
an ordinary build does not start it. It accepts two inputs:

- `version`: the exact version in the source, currently `3.4.1`.
- `publish`: a checkbox, **false by default**. Leave it unchecked for a dry run.

Only `main` is accepted. Preflight checks the requested version before the
existing three-platform build matrix runs. The assembler validates every
platform's package name, filename/metadata version and Python/platform tags,
then retains one canonical Linux source archive. GraspKit keeps only the Linux
copy of its universal wheel; rCSFs keeps all three platform wheels. This avoids
silently overwriting duplicate filenames from different runners.

The prepared archives are stored in the **pypi-distributions** Actions artifact;
the job summary lists their SHA-256 hashes and source commit. Preparation and
build jobs have no OIDC permission. Only the final, separately gated `publish`
job uses the `pypi` environment and `id-token: write`. That job downloads the
prepared artifact and invokes `pypa/gh-action-pypi-publish`; it does not rebuild.
When `publish` is false, the upload job is skipped entirely. The workflow group
serializes publication attempts. It neither bumps versions nor creates tags.

### Configure authorization in the browser

This rename creates a new PyPI project, `graspkit-atomic`. Authorization for the
previous `grasp-atomic-kit` project does not carry over; configure the publisher
for the new name below. The repository and Python import names stay unchanged.

1. In this GitHub repository, open **Settings → Environments → New environment**
   and create an environment named exactly **pypi**. No PyPI token or environment
   secret is needed. You can restrict its deployment branches to `main`.
2. Log in to [PyPI Publishing](https://pypi.org/manage/account/publishing/).
   Under **Add a new pending publisher**, choose **GitHub** and enter:

   | PyPI field | Value |
   | --- | --- |
   | PyPI Project Name | `graspkit-atomic` |
   | Owner | `YenochQin` |
   | Repository name | `graspkit` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   Enter only `publish.yml`, without `.github/workflows/`. The owner is the
   GitHub account name, not the PyPI username. Add the pending publisher. This
   does not upload files, create the project or reserve its name; the project
   is created on the first successful publication. If you already own an
   existing project with this name, add the publisher in that project's
   Publishing settings instead.
3. For a rehearsal, choose **Actions → Publish to PyPI → Run workflow**, select
   `main`, enter `3.4.1` and leave **publish unchecked**. All builds and
   package checks run, and the upload job is skipped. A dry run does not test
   the PyPI OIDC exchange because it never enters the publishing job.
4. Only when publication is authorized, run the same workflow from `main` with
   the exact version and **publish checked**. Then install that version from
   PyPI to verify it. TestPyPI is not a destination in this production workflow.

The pending-publisher setup follows the [PyPI guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
