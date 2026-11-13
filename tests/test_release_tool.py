"""tools/release.py: what a tag must agree with before anything is published."""

from pathlib import Path

import pytest
import release

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.0b1] - 2026-12-01

- A beta thing.

## [0.1.0] - 2026-11-13

First release.

- One.
- Two.

## [0.0.9]
"""


def test_the_version_comes_from_pyproject():
    assert (
        release.package_version()
        == release.tomllib.loads(Path("pyproject.toml").read_text("utf-8"))["project"]["version"]
    )


def test_a_tag_must_name_the_package_version():
    release.check_tag("v0.1.0", "0.1.0")
    for tag in ("0.1.0", "v0.1.1", "v0.1.0 ", "V0.1.0"):
        with pytest.raises(release.ReleaseError):
            release.check_tag(tag, "0.1.0")


@pytest.mark.parametrize("version, pre", [("0.1.0", False), ("1.2.3b4", True), ("1.2.3rc1", True), ("1.2.3a1", True)])
def test_prerelease_and_title(version, pre):
    assert release.is_prerelease(version) is pre
    assert release.title(version) == f"platenest v{version}" + (" (pre-release)" if pre else "")


@pytest.mark.parametrize("version", ["1.2", "1.2.3.4", "1.2.3-beta", "v1.2.3", "1.2.3b"])
def test_other_version_shapes_are_refused(version):
    with pytest.raises(release.ReleaseError):
        release.is_prerelease(version)


def test_notes_are_the_versions_own_section():
    assert release.notes(CHANGELOG, "0.1.0") == "First release.\n\n- One.\n- Two."
    assert release.notes(CHANGELOG, "0.2.0b1") == "- A beta thing."


@pytest.mark.parametrize("version", ["0.3.0", "0.0.9", "Unreleased"])
def test_a_missing_or_empty_section_is_an_error(version):
    with pytest.raises(release.ReleaseError):
        release.notes(CHANGELOG, version)


def test_prepare_writes_notes_and_outputs(tmp_path):
    version = release.package_version()
    out = release.prepare(f"v{version}", tmp_path / "notes.md")
    assert out == {"version": version, "title": release.title(version), "prerelease": "false"}
    assert (tmp_path / "notes.md").read_text("utf-8").strip()


def test_the_command_line_fails_on_a_wrong_tag(tmp_path, capsys):
    assert release.main(["prepare", "--tag", "v9.9.9", "--notes", str(tmp_path / "n.md")]) == 1
    assert "::error::" in capsys.readouterr().err and not (tmp_path / "n.md").exists()
