"""Ops that make, combine and read **directions**.

A direction is a unit vector in a model's activation space at one
(layer, point), carrying its own derivation: how it was made, from what,
on which model. One object flows everywhere a direction is used — into
`intervene/apply` (add, project out, clamp, rotate), `direction/project`,
`direction/unembed` — so a direction found one way can be tried every
other way without conversion.

The record is an `activations/vector` with a derivation: `{"kind":
"direction/vector", "space": {"model": "...", "layer": 14, "point":
"resid_post", "d": 2048}, "vector": [...], "norm": 37.2, "unit": true,
"derivation": {"method": "diff_of_means", "sources": [...], "model":
"...", ...}}`. `norm` is the magnitude before normalisation, which some
readings use.
"""

from __future__ import annotations

from mechbench_compute.lexicon._base import WILDCARD, In, Op, P

_DIRECTION = In("direction", "direction/vector", "The direction.")


_NAMED_DIRECTIONS = (
    In("directions", "direction/vector",
       "The directions as one list, when they do not each arrive on a port "
       "of their own; they are named `d0`, `d1`, … in the result.",
       many=True, required=False),
    In(WILDCARD, "direction/vector",
       "One direction per edge, on a port of your naming; the port name is "
       "the direction's name in the result.", required=False),
)


def _point() -> P:
    return P("point", "string",
             "Override the point recorded on the direction — `\"resid_post\"`, "
             "`\"resid_pre\"`, or any point name. By default it is taken from "
             "the vectors' own `space`.",
             None, value="point")


def _source() -> P:
    return P("source", "string",
             "A label for where the vectors came from, recorded in the "
             "direction's derivation for provenance.",
             None)


OPS: tuple[Op, ...] = (
    
    
    
)
