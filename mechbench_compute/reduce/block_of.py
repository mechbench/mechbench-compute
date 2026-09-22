from __future__ import annotations

from collections.abc import Mapping

from mechbench_compute.reduce.monoid_for import monoid_for


def _block_of(name: str):
    def fn(inputs, params):
        from mechbench_compute.lexicon import kinds as K

        m = monoid_for(name, params)
        raw = inputs.get("records") if isinstance(inputs, Mapping) else inputs
        recs = K.items_of(raw if raw is not None else [])
        return m.finalize(m.partial(recs, params), params)
    return fn
