from __future__ import annotations

#: The spec fields a sweep may vary, in the order the product runs them,
#: and the coordinate each one becomes on the result (000602). `strength`
#: is the outermost, because a weight edit is applied once per strength
#: and every cell under it runs against the edited model.
SWEEP_AXES: dict[str, str] = {"strength": "factor", "layers": "layer",
                              "heads": "head", "positions": "position",
                              "neurons": "neuron"}
