from __future__ import annotations

#: Refuse vector payloads past this many floats — a mistyped layer list
#: must not emit a gigabyte of CBOR. ~16 MB of float64 at the cap.
MAX_VECTOR_FLOATS = 2_000_000
