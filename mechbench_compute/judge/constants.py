from __future__ import annotations

import re

SCALES = ("numeric", "categorical", "pairwise")


_FIRST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
