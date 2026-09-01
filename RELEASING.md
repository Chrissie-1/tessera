# Releasing

How to cut a Tessera release.

## What is and is not automated

Pushing a `v*` tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml),
which:

1. Checks the tag matches the version in `Cargo.toml` **and** `worker/pyproject.toml`.
2. Runs the Python and Rust test suites.
3. Builds the Python sdist/wheel and runs `twine check` on them.
4. Builds both container images (to prove they still build).
5. Creates a GitHub release with the matching `CHANGELOG.md` section and
   attaches the Python distribution.

It does **not** publish to PyPI, Docker Hub, or GHCR. That is deliberate:
pushing to a public registry is irreversible, and a PyPI project name cannot be
reclaimed once taken. See [Publishing to a registry](#publishing-to-a-registry)
if you decide you want that.

## Versioning

[Semantic versioning](https://semver.org/). The Rust workspace and the Python
package share one version number, and the release workflow enforces that.

## Cutting a release

1. **Update the version** in both manifests:

   - `Cargo.toml` → `[workspace.package] version`
   - `worker/pyproject.toml` → `[project] version`

2. **Update `CHANGELOG.md`.** Move `[Unreleased]` items under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading and update the link definitions at the
   bottom. The release workflow extracts this section verbatim as the release
   notes, so what you write here is what people read.

3. **Verify locally.**

   ```bash
   make test
   make lint
   ```

4. **Commit and tag.**

   ```bash
   git add Cargo.toml worker/pyproject.toml CHANGELOG.md
   git commit -m "chore: release vX.Y.Z"
   git push origin master

   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```

5. **Watch the workflow** at
   [Actions](https://github.com/Chrissie-1/tessera/actions). If verification
   fails, delete the tag (`git tag -d vX.Y.Z && git push --delete origin vX.Y.Z`),
   fix, and re-tag.

## Publishing to a registry

Neither of these is wired up. Both are one-way doors — do them deliberately.

### PyPI

The package name `tessera-worker` is **not** currently registered to this
project. Check availability first, and publish to TestPyPI before the real
index.

```bash
pip install build twine
./scripts/gen_proto.sh          # stubs are gitignored; the sdist needs them
cd worker
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*   # rehearse
python -m twine upload dist/*                         # for real
```

To automate it: add a job to `release.yml` that uses
[trusted publishing](https://docs.pypi.org/trusted-publishers/) (`id-token:
write` permission plus a configured PyPI publisher) rather than a long-lived
API token.

### Container images

```bash
docker build -f gateway/Dockerfile -t <registry>/tessera-gateway:X.Y.Z .
docker build -f worker/Dockerfile  -t <registry>/tessera-worker:X.Y.Z .
docker push <registry>/tessera-gateway:X.Y.Z
docker push <registry>/tessera-worker:X.Y.Z
```

GHCR is the path of least resistance from Actions, since `GITHUB_TOKEN` can
push to `ghcr.io/<owner>/<repo>` with `packages: write` and no extra secrets.

Note that both images are **CPU-only**. `worker/Dockerfile` documents the base
image and index-URL changes needed for a CUDA build; that variant has not been
built or tested.

## Deployment notes

There is no Kubernetes manifest or Helm chart in this repository, and the
`docker compose` stack is a development convenience, not a production topology.
Before running this anywhere real, read the
[Limitations](README.md#limitations) and [scope](SECURITY.md#scope) sections —
in particular, there is no authentication, no TLS, and
`docker compose --scale worker=N` does not work as written.
