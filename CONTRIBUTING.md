# Contributing

## Branches

- `dev` — where work lands. Branch from `dev` and open pull requests against `dev`.
- `main` — released code only; it moves forward from `dev` when a stable version is released.

## Before a pull request

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

- The packer depends on `numpy` and `opencv-python-headless` and nothing else; `platenest.bambu3mf` on the standard
  library only. Tests pin both.
- No printer or slicer data goes into the package: work zones and settings are the caller's.
- The implementation is original. Do not port code from other packers; several are under licences this project
  cannot take.
- Test fixtures are generated in code. Do not commit model files.
- A behaviour change comes with a test that fails without it and a line in `CHANGELOG.md` under `[Unreleased]`.
- Conventional commits: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.

## Releases

Versions are `X.Y.Z` for stable releases and `X.Y.ZaN` / `X.Y.ZbN` / `X.Y.ZrcN` for pre-releases.

1. Set `version` in `pyproject.toml` and rename `[Unreleased]` in `CHANGELOG.md` to `## [<version>] - <date>`
   (leave a new empty `[Unreleased]` above it).
2. Merge to `dev`, wait for CI to be green there. For a stable version, fast-forward `main` to `dev`.
3. Tag that commit `v<version>` and push the tag.

The tag runs `.github/workflows/release.yml`: the whole CI again, then `tools/release.py` checks that the tag equals
the package version and that the changelog has a section for it, a stable tag must be on `main`, and only then the
wheel and the source archive are attached to a GitHub release and published to PyPI through trusted publishing — no
token is stored anywhere. Tags are never moved or deleted: a broken release is followed by the next version.
