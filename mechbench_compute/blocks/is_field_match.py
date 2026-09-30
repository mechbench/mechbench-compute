from __future__ import annotations

import fnmatch


def is_field_match(path: str, pattern: str) -> bool:
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if fnmatch.fnmatchcase(prefix, pattern) or fnmatch.fnmatchcase(prefix, "*." + pattern):
            return True
    return False
