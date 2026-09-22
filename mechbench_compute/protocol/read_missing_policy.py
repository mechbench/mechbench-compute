"""What to do about an input its upstream never produced."""

from __future__ import annotations


def read_missing_policy(decl, edges) -> str:
    """What to do about an absent input on this port.

    The OP declares what its port can meaningfully do without the input;
    an EDGE may override, because whether a partial result is worth
    having is a question about the experiment, not about the operation.
    An edge that says nothing inherits the port's declaration, and a
    port that says nothing fails — silence never buys tolerance.
    """
    chosen = [e.get("on_missing") for e in edges if e.get("on_missing")]
    if chosen:
        return str(chosen[0])
    return decl.on_missing if decl is not None else "fail"
