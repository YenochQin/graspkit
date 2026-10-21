# Preparing a graspkit-atomic release

The prepared stable version is **3.4.3**. Its single source is
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

Output: `dist/graspkit_atomic-3.4.3.tar.gz` and
`dist/graspkit_atomic-3.4.3-py3-none-any.whl`. An optional `--outdir PATH` selects an
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
2. Confirm PyPI ownership/availability of `graspkit-atomic`, and that version 3.4.3
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

## Manual publishing to PyPI and GitHub

`.github/workflows/publish.yml` is named **Publish packages** in GitHub Actions.
It runs only through **Run workflow**, from `main`. Pushing commits, tags, or
completing a build does not start a publication. Its inputs are:

| Input | Default | Purpose |
| --- | --- | --- |
| `version` | Required | Exact source version, currently `3.4.3`. |
| `publish_pypi` | Off | Upload the validated packages to PyPI. |
| `github_release` | Off | Create a GitHub Release using the same packages. |
| `draft` | On | Keep the GitHub Release as a draft; turn off only for formal publication. |

Open **Actions → Publish packages → Run workflow**, select `main`, enter the
source version, and select the destinations you intend to use:

| PyPI | GitHub Release | Result |
| --- | --- | --- |
| Off | Off | Build and validate only; no publication. |
| On | Off | Build, validate and upload to PyPI. |
| Off | On | Build, validate and create a GitHub Release (draft by default). |
| On | On | Build and validate once, then upload the same files to both destinations. |

A prior dry run is optional. No separate workflow, run ID or tag input is
needed. The GitHub tag is derived as `v{version}`, for example `v3.4.3`.
The `draft` option has no effect when `github_release` is off.

### One build, two independent destinations

Preflight checks the requested version against the source before the existing
Linux/Windows/macOS build matrix runs. The assembler checks package names,
filename/metadata versions and Python/platform tags across the full matrix.
It retains one Linux source archive, plus one universal wheel for GraspKit or
three native wheels for rCSFs. The canonical files are stored in the
**pypi-distributions** Actions artifact with SHA-256 hashes in the job summary.

Both upload jobs depend directly on the successful `prepare` job. They each
consume that same artifact and do not rebuild. Neither upload job depends on
the other: a PyPI failure does not block GitHub Release creation, and vice versa.
If an upload fails before creating a Release or accepting files, retrying that
failed job reuses the run's existing validated artifact. Resolve any existing
file/Release conflict before retrying; successful uploads are not overwritten.
The workflow has no rollback across services; inspect each job's result.
The concurrency group serializes runs of this publishing workflow.

A dry run (both destination options off) does not exercise PyPI authorization.
PyPI uploads only work for filenames that have not already been used. For a
new release of modified package contents, update the source version and enter
that version in the workflow; do not try to overwrite an existing release.

### GitHub Release behavior

The Release job automatically reads this run's ID, attempt and source commit
from GitHub. It confirms that preflight, all three builds and preparation have
succeeded; it does not require the whole workflow or the PyPI job to finish.
It checks the downloaded artifact's GitHub SHA-256 digest, package metadata and
wheel platforms before creating a Release. These checks run with Python's
standard library and the GitHub CLI supplied by the hosted runner.

The Release contains the wheel(s), source archive and `SHA256SUMS`. Its notes
record the exact source commit, build-run URL, artifact ID and per-file hashes.
An existing tag must resolve to that commit; an existing Release for the same
tag, including a draft, causes a failure instead of being overwritten. Partial
drafts are retained for inspection. Review drafts in the repository's
**Releases** page. When a tag does not exist yet, GitHub may defer creating it
until the draft is published; the draft retains the exact target commit.

The default is a **draft**. Formal GitHub publication happens only when
`github_release` is on and `draft` is explicitly off. The Release job uses the
built-in `GITHUB_TOKEN` with `actions: read` and `contents: write`; it needs no
personal access token, PyPI credential or OIDC permission.

### PyPI authorization

Only the `publish_pypi` job uses the `pypi` environment and `id-token: write`.
It downloads the prepared artifact and invokes `pypa/gh-action-pypi-publish`.
The workflow filename is still `publish.yml`; changing its display name to
**Publish packages** does not change the Trusted Publisher identity.

1. In this GitHub repository, open **Settings → Environments** and ensure an
   environment named exactly `pypi` exists. No PyPI API token is needed.
2. For an existing PyPI project you own, open its **Publishing** settings and
   add or verify the GitHub Trusted Publisher. For a new project, use
   [account Publishing](https://pypi.org/manage/account/publishing/) to add a
   pending publisher instead:

   | PyPI field | Value |
   | --- | --- |
   | PyPI Project Name (pending publisher only) | `graspkit-atomic` |
   | Owner | `YenochQin` |
   | Repository name | `graspkit` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   Enter only the workflow filename, without `.github/workflows/`. A pending
   publisher does not reserve the project name; the first successful use
   creates the project. Authorization for another PyPI project does not
   automatically authorize this one.
3. To rehearse, leave **publish_pypi** and **github_release** unchecked. To upload
   to PyPI when authorized, enable **publish_pypi**. Enable **github_release**
   too if you also want the same files on GitHub, leaving **draft** checked
   unless a formal Release is intended. This workflow has no TestPyPI target.

See the [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/).
