from __future__ import annotations

from collections.abc import Sequence

from mechbench_compute._mlx import mx
from mechbench_compute.adapters.read_sft_items import SftItem
from mechbench_compute.spans import add_to_span

ROWS = 256


# external: MLX/Metal — a full [seq, vocab] float32 logits tensor in the gradient graph trips the command-buffer watchdog on long prompts; cast the rows a chunk at a time
def compute_sft_loss(lm, batch: Sequence[SftItem]) -> mx.array:
    total = mx.zeros(())
    count = 0
    for item in batch:
        out = lm(mx.array([item.ids[:-1]]))
        add_to_span(forwards=1)
        logits = (out.logits if hasattr(out, "logits") else out)[0]
        trained = item.trained[1:]
        cuts = list(range(ROWS, len(trained), ROWS))
        chunks = zip(mx.split(logits, cuts), mx.split(mx.array(item.ids[1:]), cuts),
                     range(0, len(trained), ROWS), strict=True)
        for rows, targets, start in chunks:
            weights = trained[start:start + ROWS]
            if not any(weights):
                continue
            rows = rows.astype(mx.float32)
            nll = (mx.logsumexp(rows, axis=-1)
                   - mx.take_along_axis(rows, targets[:, None], axis=-1)[:, 0])
            total = total + mx.sum(nll * mx.array([float(w) for w in weights]))
        count += sum(trained)
    return total / count
