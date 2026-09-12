"""Locating the data files, portably.

Three modules used to hardcode one developer's directory layout as the
default path to `column_assignment.csv`. That works on exactly one machine and
fails everywhere else with a confusing "file not found" pointing at a home
directory the user has never heard of. This module replaces that with a search
over the places the file plausibly lives, and an error message that says how
to get it when the search fails.

Resolution order for the retinotopic map, first hit wins:

1. `$FLYGUARD_COLUMNS`, so a deployment can pin it explicitly;
2. `column_assignment.csv` or `data/column_assignment.csv` under the current
   working directory;
3. `~/flywire/column_assignment.csv`, the layout the README suggests;
4. `<repo>/data/column_assignment.csv`, next to the packaged subnetwork.
"""

from __future__ import annotations

import os
from pathlib import Path

COLUMNS_FILENAME = "column_assignment.csv"
COLUMNS_ENV_VAR = "FLYGUARD_COLUMNS"

SUBNETWORK_FILENAME = "looming.npz"
SUBNETWORK_ENV_VAR = "FLYGUARD_SUBNETWORK"

DOWNLOAD_HINT = (
    "Download it from https://codex.flywire.ai -> Info -> Download Data -> "
    "snapshot 783, then either pass --columns, set $FLYGUARD_COLUMNS, or put "
    "the file in ./data/."
)


def _package_data_dir() -> Path:
    import flyguard

    return Path(flyguard.__file__).resolve().parent.parent / "data"


def candidate_columns_paths() -> list[Path]:
    """Every place `column_assignment.csv` is looked for, in order."""
    candidates = []
    env = os.environ.get(COLUMNS_ENV_VAR)
    if env:
        candidates.append(Path(env).expanduser())
    cwd = Path.cwd()
    candidates += [
        cwd / COLUMNS_FILENAME,
        cwd / "data" / COLUMNS_FILENAME,
        Path.home() / "flywire" / COLUMNS_FILENAME,
        _package_data_dir() / COLUMNS_FILENAME,
    ]
    # Run from the repo root, the working directory and the package data dir
    # are the same place; listing it twice makes the "searched:" error read
    # like a bug in the search rather than a missing file.
    seen, unique = set(), []
    for path in candidates:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_columns_path() -> Path | None:
    """First existing candidate, or None. Never raises."""
    for path in candidate_columns_paths():
        if path.is_file():
            return path
    return None


def default_columns_path() -> str:
    """The default for a `--columns` argument.

    Returns the first existing candidate as a string, or the most likely
    location if none exists -- argparse defaults are printed in `--help`, so
    this must not raise just because the file is missing.
    """
    found = find_columns_path()
    return str(found) if found else str(Path.home() / "flywire" / COLUMNS_FILENAME)


def require_columns_path(given: Path | str | None = None) -> Path:
    """Resolve and verify, with an error that says what to do about it."""
    if given is not None:
        path = Path(given).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. {DOWNLOAD_HINT}")
        return path
    found = find_columns_path()
    if found is None:
        searched = "\n  ".join(str(p) for p in candidate_columns_paths())
        raise FileNotFoundError(
            f"could not find {COLUMNS_FILENAME}. Searched:\n  {searched}\n\n{DOWNLOAD_HINT}"
        )
    return found


# ---------------------------------------------------------------------------
# The extracted subnetwork
# ---------------------------------------------------------------------------

SUBNETWORK_HINT = (
    "It is committed in the repository at data/looming.npz (~1 MB) but is not "
    "shipped inside the wheel, because it is FlyWire-derived and carries a "
    "non-commercial licence the MIT code does not. Clone the repository, or "
    "regenerate it with:\n"
    "    python -m flyguard.extract --data <codex_csv_dir> --out data/looming.npz\n"
    "then pass the path explicitly or set $FLYGUARD_SUBNETWORK."
)


def candidate_subnetwork_paths() -> list[Path]:
    """Where `looming.npz` is looked for, in order."""
    candidates = []
    env = os.environ.get(SUBNETWORK_ENV_VAR)
    if env:
        candidates.append(Path(env).expanduser())
    cwd = Path.cwd()
    candidates += [
        cwd / "data" / SUBNETWORK_FILENAME,
        cwd / SUBNETWORK_FILENAME,
        _package_data_dir() / SUBNETWORK_FILENAME,
        Path.home() / "flywire" / SUBNETWORK_FILENAME,
    ]
    seen, unique = set(), []
    for path in candidates:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_subnetwork_path() -> Path | None:
    for path in candidate_subnetwork_paths():
        if path.is_file():
            return path
    return None


def require_subnetwork_path(given: Path | str | None = None) -> Path:
    """Resolve and verify the subnetwork `.npz`, with an actionable error.

    A `pip install flyguard` puts the package in site-packages with no `data/`
    beside it, so the naive "next to the package" guess silently points at a
    path that never exists. This searches instead, and explains.
    """
    if given is not None:
        path = Path(given).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found.\n{SUBNETWORK_HINT}")
        return path
    found = find_subnetwork_path()
    if found is None:
        searched = "\n  ".join(str(p) for p in candidate_subnetwork_paths())
        raise FileNotFoundError(
            f"could not find {SUBNETWORK_FILENAME}. Searched:\n  {searched}\n\n"
            f"{SUBNETWORK_HINT}"
        )
    return found


def default_subnetwork_path() -> str:
    """The default for an `--npz` argument; never raises."""
    found = find_subnetwork_path()
    return str(found) if found else str(Path("data") / SUBNETWORK_FILENAME)
