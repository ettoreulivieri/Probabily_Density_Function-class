# Copyright (C) 2026 Ettore
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import ast
import datetime
import pathlib
import subprocess
from enum import Enum

from packaging.version import Version, parse

from pdf_gen import __name__ as __package_name__

_ROOT_DIR = pathlib.Path(__file__).parent.parent
_DOCS_DIR = _ROOT_DIR / "docs"
_SRC_DIR = _ROOT_DIR / "src" / __package_name__
_VERSION_FILE_PATH = _SRC_DIR / "_version.py"
_RELEASE_NOTES_PATH = _DOCS_DIR / "release_notes.rst"
_ENCODING = "utf-8"


class BumpMode(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    MICRO = "micro"


def read_version_file() -> Version:
    print(f"Reading version from {_VERSION_FILE_PATH}...")
    with open(_VERSION_FILE_PATH, encoding=_ENCODING) as input_file:
        line = input_file.readline().rstrip("\n")
    if not line.startswith("__version__"):
        raise ValueError(f"Invalid version file content: {line}")
    _, _, version_string = line.partition("=")
    return parse(ast.literal_eval(version_string.strip()))


def bump_version(version: Version, mode: BumpMode) -> Version:
    print(f'Bumping version (mode = {mode})...')
    major, minor, micro = version.release
    if mode == BumpMode.MAJOR:
        version_string = f"{major + 1}.0.0"
    elif mode == BumpMode.MINOR:
        version_string = f"{major}.{minor + 1}.0"
    elif mode == BumpMode.MICRO:
        version_string = f"{major}.{minor}.{micro + 1}"
    else:
        raise ValueError(f"Invalid bump mode {mode}")
    return Version(version_string)


def write_version_file(version: Version) -> None:
    print(f"Writing version {version} to {_VERSION_FILE_PATH}...")
    with open(_VERSION_FILE_PATH, "w", encoding=_ENCODING) as output_file:
        output_file.write(f'__version__ = "{version}"\n')


def update_release_notes(version: Version, num_header_lines: int = 5) -> None:
    print(f"Updating release notes {_RELEASE_NOTES_PATH}...")
    with open(_RELEASE_NOTES_PATH, encoding=_ENCODING) as input_file:
        lines = input_file.readlines()
    text = f'Version {version} ({datetime.datetime.now().date()})'
    underline = '~' * len(text)
    text = f'\n{text}\n{underline}\n\n'
    lines.insert(num_header_lines, text)
    with open(_RELEASE_NOTES_PATH, "w", encoding=_ENCODING) as output_file:
        output_file.writelines(lines)


def _cmd(*args) -> subprocess.CompletedProcess:
    print(f"Executing command \"{' '.join(args)}\"...")
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    print(result.stdout)
    return result


def release(mode: BumpMode, target_branch: str = "main") -> None:
    current_branch = _cmd("git", "branch", "--show-current").stdout.strip("\n")
    if current_branch != target_branch:
        raise RuntimeError(f"You are on the {current_branch} branch, not {target_branch}")
    version = bump_version(read_version_file(), mode)
    write_version_file(version)
    update_release_notes(version)
    msg = f"Prepare for tag {version}."
    _cmd("git", "commit", "-a", "-m", msg)
    _cmd("git", "push")
    msg = f"Tagging version {version}..."
    _cmd("git", "tag", "-a", str(version), "-m", f'"{msg}"')
    _cmd("git", "push", "--tags")
    _cmd("git", "status")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Release a new version of the package.")
    parser.add_argument("mode", choices=[mode.value for mode in BumpMode],
                        help="The version bump mode.")
    arguments = parser.parse_args()
    release(BumpMode(arguments.mode))
