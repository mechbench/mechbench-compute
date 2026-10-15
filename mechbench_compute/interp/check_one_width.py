from __future__ import annotations

from mechbench_compute.capture_rows import CaptureRows


def check_one_width(sink: CaptureRows, point: str, layers: list[int],
                    widths: list[int]) -> None:
    if sink.writer is None or len(set(widths)) == 1:
        return
    by_width: dict[int, list[int]] = {}
    for layer, w in zip(layers, widths):
        by_width.setdefault(w, []).append(layer)
    said = "; ".join(f"layers {ls} are {w} wide" for w, ls in by_width.items())
    raise ValueError(
        f"`{point}` here is {said}, and tensor storage holds rows of one width: "
        "read the layers of each width in a node of their own")
