"""Tests for data-file location.

Three modules used to hardcode one developer's home directory as the default
path to `column_assignment.csv`, and the subnetwork was resolved by position
relative to the package, which silently points at a non-existent directory
after `pip install`. Both failures produce a "file not found" naming a path
the user has never seen, so the error messages are as much the subject of
these tests as the search order is.
"""

import pytest

from flyguard import paths


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(paths.COLUMNS_ENV_VAR, raising=False)
    monkeypatch.delenv(paths.SUBNETWORK_ENV_VAR, raising=False)


# --- columns ---------------------------------------------------------------


def test_env_var_takes_precedence(tmp_path, monkeypatch):
    pinned = tmp_path / "pinned.csv"
    pinned.write_text("x")
    monkeypatch.setenv(paths.COLUMNS_ENV_VAR, str(pinned))
    assert paths.find_columns_path() == pinned
    assert paths.candidate_columns_paths()[0] == pinned


def test_working_directory_is_searched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / paths.COLUMNS_FILENAME).write_text("x")
    assert paths.find_columns_path() == tmp_path / paths.COLUMNS_FILENAME


def test_data_subdirectory_is_searched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    target = tmp_path / "data" / paths.COLUMNS_FILENAME
    target.write_text("x")
    assert paths.find_columns_path() == target


def test_candidates_are_deduplicated(tmp_path, monkeypatch):
    """Run from the repo root the working directory and the package data dir
    are the same place; listing it twice makes the error read like a bug in
    the search rather than a missing file."""
    monkeypatch.chdir(tmp_path)
    got = paths.candidate_columns_paths()
    assert len(got) == len(set(got))


def test_default_never_raises_when_the_file_is_absent(tmp_path, monkeypatch):
    """argparse prints defaults in --help, so this must not blow up just
    because the data has not been downloaded yet."""
    monkeypatch.chdir(tmp_path)
    assert isinstance(paths.default_columns_path(), str)


def test_missing_columns_error_says_where_to_get_the_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "_package_data_dir", lambda: tmp_path / "nope")
    monkeypatch.setattr(paths.Path, "home", staticmethod(lambda: tmp_path / "home"))
    with pytest.raises(FileNotFoundError, match="codex.flywire.ai"):
        paths.require_columns_path()


def test_explicit_missing_columns_path_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="codex.flywire.ai"):
        paths.require_columns_path(tmp_path / "absent.csv")


# --- subnetwork ------------------------------------------------------------


def test_subnetwork_env_var_takes_precedence(tmp_path, monkeypatch):
    pinned = tmp_path / "pinned.npz"
    pinned.write_bytes(b"x")
    monkeypatch.setenv(paths.SUBNETWORK_ENV_VAR, str(pinned))
    assert paths.find_subnetwork_path() == pinned


def test_missing_subnetwork_error_explains_the_wheel_case(tmp_path, monkeypatch):
    """A pip-installed package has no data/ beside it. The error has to say so,
    otherwise it names a site-packages path the user has never seen."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, "_package_data_dir", lambda: tmp_path / "nope")
    monkeypatch.setattr(paths.Path, "home", staticmethod(lambda: tmp_path / "home"))
    with pytest.raises(FileNotFoundError) as exc:
        paths.require_subnetwork_path()
    message = str(exc.value)
    assert "not shipped inside the wheel" in message
    assert "flyguard.extract" in message


def test_explicit_subnetwork_path_is_returned_when_it_exists(tmp_path):
    npz = tmp_path / "given.npz"
    npz.write_bytes(b"x")
    assert paths.require_subnetwork_path(npz) == npz
