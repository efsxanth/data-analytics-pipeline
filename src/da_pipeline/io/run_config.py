
import json
import re
from typing import Any


def escape_windows_backslashes(file_content: str) -> str:
    """
    Makes Windows paths safe to parse with `json.loads()`.

    In JSON a single backslash starts an escape sequence (e.g. "\\n", "\\t", "\\u"),
    so a hand-written path such as "C:\\dir\\subdir" is invalid JSON and
    `json.loads()` raises an error. Every backslash run is therefore replaced with
    an escaped backslash ("\\\\"), which JSON parses back into the single backslash
    of the original path.

    This handles both writing styles found in config files:
        "C:\\dir\\subdir"      (unescaped, invalid JSON)
        "C:\\\\dir\\\\subdir"  (already escaped, valid JSON)
    Both parse to: C:\\dir\\subdir

    Args:
        file_content: raw text read from a .json file.

    Returns:
        The same text with all backslash runs escaped for JSON.
    """
    return re.compile(r'\\+').sub(r'\\\\', file_content)


def load_json(config_fpath: str) -> dict:
    """
    Loads and returns a JSON file as a dictionary.

    Windows paths are escaped before parsing and are returned in their original
    Windows form (e.g. "C:\\dir\\subdir\\file.ext").

    Args:
        config_fpath: path to the .json file.
    """
    with open(config_fpath, encoding='utf-8') as file:
        file_content = file.read()

    return json.loads(escape_windows_backslashes(file_content))


def dict_raise_on_duplicates(parsed_pairs: list[tuple[str, Any]]) -> dict:
    """
    Rejects duplicate keys when constructing a dictionary from JSON data.

    This function is intended to be used as the 'object_pairs_hook' argument of
    `json.loads()` (see `load_json_containing_paths`). It processes a list of
    key-value pairs and raises a ValueError if any duplicate keys are found.

    Args:
        parsed_pairs: A list of tuples, where each tuple is a key-value pair from
        the JSON object. The first element is the key (a string), the second is
        the value associated with that key.

    Example:
        the parsed_pairs would look like:
            [
                ("key_name_1", ["C:\\dir\\subdir_1\\file_1.ext", "C:\\dir\\subdir_2\\file_2.ext"]),
                ("key_name_2", ["C:\\dir\\subdir_3\\file_3.ext", "C:\\dir\\subdir_4\\file_4.ext"])
            ]

    Returns:
        d: A dictionary built from the key-value pairs, provided there were no
        duplicate keys.
    """
    d: dict[str, Any] = {}
    for key_name, paths_list in parsed_pairs:
        if key_name in d:
            raise ValueError(f"duplicate key name: {key_name}")
        d[key_name] = paths_list
    return d


def load_json_containing_paths(config_fpath: str) -> dict[str, list[str]]:
    """
    Loads a config .json file that maps key names (groups) to lists of Windows
    file paths.

    Example of the expected structure:
    {
        "key_name_1": [
            "C:\\dir\\subdir_1\\file_1.ext",
            "C:\\dir\\subdir_2\\file_2.ext"
        ],
        "key_name_2": [
            "C:\\dir\\subdir_3\\file_3.ext",
            "C:\\dir\\subdir_4\\file_4.ext"
        ]
    }

    Each key represents a group of related file paths. Paths keep their Windows
    form; they are returned ready to be passed to `open()` / `pathlib.Path()` on
    Windows.

    Args:
        config_fpath: path to the configuration file.

    Returns:
        A dict mapping each key name to its list of Windows file paths.

    Raises:
        ValueError: on duplicate keys, non-list or empty values, non-string
        paths, or key names containing newline characters.
    """
    with open(config_fpath, encoding='utf-8') as file:
        file_content = file.read()

    config_dict = json.loads(
        escape_windows_backslashes(file_content),
        object_pairs_hook=dict_raise_on_duplicates,
    )

    if not isinstance(config_dict, dict):
        raise ValueError(f"Expected a JSON object at the top level of '{config_fpath}'.")

    for key_name, paths in config_dict.items():
        # Key names may be reused elsewhere (plot labels, filenames, ...), so keep
        # them free of newlines. Both a real newline and a literal "\n" (which the
        # backslash escaping above preserves as two characters) are rejected.
        if '\n' in key_name or '\\n' in key_name:
            raise ValueError(
                f"Invalid key name '{key_name}': it contains newline characters ('\\n'). "
                f"Please remove them, as key names may be used for various purposes.")

        if not isinstance(paths, list):
            raise ValueError(f"The value for '{key_name}' must be a list of file paths.")

        if not paths:
            raise ValueError(f"The list for '{key_name}' is empty.")

        for path in paths:
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"'{key_name}' contains an invalid (non-string or empty) path: {path!r}")

    return config_dict
