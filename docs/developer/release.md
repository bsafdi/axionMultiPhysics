# Releasing

A release is a snapshot of the source tree at a tagged commit, with wheels for PyPI and a matching documentation
site. This page lists the steps.

## Before the tag

1. **Version.** Set the new version in three places, which `tests/test_packaging.py` checks agree:
   `pyproject.toml` (`project.version`), `CMakeLists.txt` (`project(axionMultiPhysics VERSION ...)`) and
   `cpp/include/axionmultiphysics/core.hpp` (`axmp::kVersion`).
2. **Changelog and citation.** Add the release section to `CHANGELOG.md`, and update `version` and `date-released` in
   `CITATION.cff`.
3. **Tests.** Run the full suite on an MPI build (`AXMP_SLOW=1 AXMP_MPI=1`, [Running the tests](testing.md)) and check
   that CI is green on the commit to be tagged.
4. **Generated material.** Regenerate what depends on the code and has changed: validation records (their scripts in
   `benchmarks/`), the walkthrough notebooks' outputs (`python scripts/docs/execute_notebooks.py` on an MPI build with
   STRUMPACK), and the homepage figures
   (`python docs/images/make_homepage_figures.py`, then `--check`; [Documentation and figures](docs.md)). Never edit
   generated numbers by hand. Commit a regenerated figure together with the generator change that caused it, and look
   at the images themselves (titles and labels): a text search does not see them.
5. **Documentation.** `mkdocs build --strict` passes, the user guide and `llms.txt` describe the released API, and
   beta labels match the code.
6. **Clean tree.** The snapshot contains source, documentation, tests, examples and the data they read; no build
   products, run directories, logs or machine-specific files.

## Tag and publish

1. **Deploy the documentation** by starting the documentation workflow by hand on `main` (`workflow_dispatch`).
   GitHub Pages must be enabled with "GitHub Actions" as the source. This step comes first: the wheels, the sdist and
   the GitHub release all carry the documentation URL (`pyproject.toml`, `CITATION.cff`, `README.md`, `mkdocs.yml`),
   so the site has to serve this version before they are published. Pages deploys only from `main` and only when the
   workflow is started by hand, so pushing the release commit does not refresh the site by itself; open the deployed
   page and check that it is the version being released before going on.
2. **Tag** the release commit `vX.Y.Z` and push the tag. The wheels workflow (`.github/workflows/wheels.yml`) builds
   the wheels for every platform and Python version and the sdist, and installs and tests the wheels on clean runners.
3. **Publish to PyPI** by starting the wheels workflow by hand on the tag (`workflow_dispatch`) with `publish: pypi`,
   and approving the `pypi` environment when the run asks for a reviewer. The upload uses trusted publishing, so no
   API token is stored in the repository: PyPI trusts this repository's `wheels.yml` in the GitHub environment
   `pypi`. An index accepts each version only once, so a broken upload is fixed by a new patch version. The publish
   job runs only on a `v*` tag, so the uploaded files are the ones built from the tagged commit. `README.md` uses
   absolute links and image URLs so that they resolve on PyPI as on GitHub; PyPI does not render GitHub's inline
   math, which it shows as plain text.
4. **Publish the GitHub release** with the changelog section, and upload `docs/images/home/social_preview.png` as the
   repository's social preview (Settings, General). Publishing the release archives the tagged source on Zenodo,
   which mints the DOI. The repository must be switched on in Zenodo's GitHub settings before the release is
   published; Zenodo takes the title, authors and licence from `CITATION.cff`.

A TestPyPI rehearsal is optional and is not part of a normal release. To rehearse, start the same workflow on the tag
with `publish: testpypi` before step 3, and then, in a clean environment, install from TestPyPI and solve the
quick-start toroid:
`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "axionmultiphysics[mesh]"`.
TestPyPI uses trusted publishing through the `testpypi` environment in the same way, and it too accepts each version
only once, so a rehearsed version number cannot be reused there.

## After the release

1. **DOI.** Zenodo assigns two DOIs: one for this version and a concept DOI that always resolves to the newest
   version. Add the concept DOI to `CITATION.cff` (`doi:`) and as a badge in `README.md`, and the version DOI to the
   GitHub release notes.
2. **Next version.** Open the next version's section in `CHANGELOG.md`. A fix to a release goes into a patch version
   (`vX.Y.Z+1`) with the same steps.
