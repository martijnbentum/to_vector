#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import re
import sys


PROJECT_HEADER_RE = re.compile(r"(?m)^\[project\]\s*$")
NEXT_SECTION_RE = re.compile(r"(?m)^\[.*\]\s*$")
VERSION_RE = re.compile(r'(?m)^version\s*=\s*"([^"]+)"\s*$')


def bump_version(version: str) -> str:
    parts = version.split(".")
    if not parts:
        raise ValueError("version is empty")
    if not parts[-1].isdigit():
        raise ValueError(
            f"version '{version}' does not end with a numeric segment that can be bumped"
        )
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def main() -> int:
    pyproject_path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")

    project_header = PROJECT_HEADER_RE.search(content)
    if project_header is None:
        raise ValueError("could not find [project] section in pyproject.toml")

    next_section = NEXT_SECTION_RE.search(content, project_header.end())
    project_end = next_section.start() if next_section else len(content)
    project_section = content[project_header.end() : project_end]

    version_match = VERSION_RE.search(project_section)
    if version_match is None:
        raise ValueError("could not find version in [project] section")

    current_version = version_match.group(1)
    next_version = bump_version(current_version)
    updated_section = VERSION_RE.sub(
        f'version = "{next_version}"',
        project_section,
        count=1,
    )
    pyproject_path.write_text(
        content[: project_header.end()] + updated_section + content[project_end:],
        encoding="utf-8",
    )
    print(f"Bumped pyproject version: {current_version} -> {next_version}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - hook path
        print(f"Version bump failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
