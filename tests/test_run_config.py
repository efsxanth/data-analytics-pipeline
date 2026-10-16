"""Loading the JSON config that maps conditions to extract files.

The file is written by hand, on Windows, so it is the most likely place for a
malformed input to enter the pipeline. Each rejection here is a mistake that
would otherwise surface much later as a confusing failure.
"""

from __future__ import annotations

import pytest

from da_pipeline.io.run_config import escape_windows_backslashes, load_json_containing_paths


def write(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    return path


class TestWindowsPaths:
    def test_unescaped_backslashes_are_accepted(self, tmp_path):
        # This is invalid JSON as written; a hand-edited config almost always is.
        path = write(tmp_path, r'{"A": ["C:\data\run\file.h5"]}')

        config = load_json_containing_paths(path)

        assert config["A"] == [r"C:\data\run\file.h5"]

    def test_already_escaped_backslashes_round_trip_unchanged(self, tmp_path):
        path = write(tmp_path, r'{"A": ["C:\\data\\run\\file.h5"]}')

        config = load_json_containing_paths(path)

        assert config["A"] == [r"C:\data\run\file.h5"]

    def test_forward_slash_paths_are_untouched(self, tmp_path):
        path = write(tmp_path, '{"A": ["/mnt/data/run/file.h5"]}')

        assert load_json_containing_paths(path)["A"] == ["/mnt/data/run/file.h5"]

    def test_escaping_collapses_any_run_of_backslashes(self):
        assert escape_windows_backslashes(r"a\b") == r"a\\b"
        assert escape_windows_backslashes(r"a\\b") == r"a\\b"


class TestRejections:
    def test_duplicate_condition_is_rejected(self, tmp_path):
        # json.loads would keep the last one, silently dropping a whole condition.
        path = write(tmp_path, '{"A": ["one.h5"], "A": ["two.h5"]}')

        with pytest.raises(ValueError, match="duplicate key name"):
            load_json_containing_paths(path)

    def test_empty_file_list_is_rejected(self, tmp_path):
        path = write(tmp_path, '{"A": []}')

        with pytest.raises(ValueError, match="is empty"):
            load_json_containing_paths(path)

    def test_non_list_value_is_rejected(self, tmp_path):
        path = write(tmp_path, '{"A": "one.h5"}')

        with pytest.raises(ValueError, match="must be a list"):
            load_json_containing_paths(path)

    def test_blank_path_is_rejected(self, tmp_path):
        path = write(tmp_path, '{"A": ["one.h5", "   "]}')

        with pytest.raises(ValueError, match="invalid"):
            load_json_containing_paths(path)

    def test_newline_in_condition_name_is_rejected(self, tmp_path):
        # Condition names become plot labels and file names.
        path = write(tmp_path, '{"A\\nB": ["one.h5"]}')

        with pytest.raises(ValueError, match="newline"):
            load_json_containing_paths(path)

    def test_top_level_array_is_rejected(self, tmp_path):
        path = write(tmp_path, '[{"A": ["one.h5"]}]')

        with pytest.raises(ValueError, match="JSON object"):
            load_json_containing_paths(path)


def test_condition_order_is_preserved(tmp_path):
    # The config's key order is the order every figure puts conditions in.
    path = write(tmp_path, '{"Zeta": ["z.h5"], "Alpha": ["a.h5"], "Mid": ["m.h5"]}')

    assert list(load_json_containing_paths(path)) == ["Zeta", "Alpha", "Mid"]
