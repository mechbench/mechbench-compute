"""The budget cap (task 000337; mandatory on every remote node per
Benji's decision, 000334).

A cap that is checked after the fact is an epitaph, not a control. So
the order is: count the input tokens, price the worst case (input +
max_tokens of output), refuse BEFORE the call if that would cross the
cap, then settle the actual cost from the response's usage.

A Budget is shared per node (and, with a job-level cap, chained to a
parent), so concurrent remote items cannot each individually fit while
together overrunning: `reserve` holds the worst case until `settle`
replaces it with what was really spent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from mechbench_compute.providers.errors import BudgetExceeded


@dataclass
class Budget:
    """`cap_usd` is the hard ceiling. `spent` is settled cost;
    `reserved` is the worst case of calls in flight."""

    cap_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    calls: int = 0
    parent: Budget | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False,
                                  compare=False)

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self.reserved_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.committed_usd)

    def reserve(self, estimate_usd: float, *, provider: str = "",
                model: str = "") -> float:
        """Hold the worst case; raise before any call is made. Returns
        the reservation, which `settle` takes back."""
        if self.parent is not None:
            self.parent.reserve(estimate_usd, provider=provider, model=model)
        with self._lock:
            if self.spent_usd + self.reserved_usd + estimate_usd > self.cap_usd + 1e-12:
                if self.parent is not None:
                    self.parent.release(estimate_usd)
                raise BudgetExceeded(cap_usd=self.cap_usd, spent_usd=self.spent_usd,
                                     estimate_usd=estimate_usd, provider=provider,
                                     model=model)
            self.reserved_usd += estimate_usd
        return estimate_usd

    def settle(self, reservation_usd: float, actual_usd: float) -> None:
        """Replace a reservation with what the response actually cost."""
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reservation_usd)
            self.spent_usd += actual_usd
            self.calls += 1
        if self.parent is not None:
            self.parent.settle(reservation_usd, actual_usd)

    def release(self, reservation_usd: float) -> None:
        """Give back a reservation whose call never happened."""
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reservation_usd)
        if self.parent is not None:
            self.parent.release(reservation_usd)

    def child(self, cap_usd: float) -> Budget:
        """A node's budget under a job's: the node cannot exceed its own
        cap, and every node together cannot exceed the job's."""
        return Budget(cap_usd=min(cap_usd, self.cap_usd), parent=self)

    def to_wire(self) -> dict[str, float]:
        return {"cap_usd": round(self.cap_usd, 6),
                "spent_usd": round(self.spent_usd, 6),
                "calls": self.calls}


def budget_from(params, *, required: bool = True) -> Budget:
    """A node's Budget from its params. Remote nodes MUST declare one —
    the api refuses the protocol without it (000335), and this is the
    second gate, in the executor, for a spec that got in another way."""
    cap = params.get("budget_usd") if hasattr(params, "get") else None
    if cap is None:
        if required:
            raise ValueError(
                "this node calls a remote provider and declares no "
                "budget_usd. Every remote node carries a cap — an "
                "unbounded loop against a metered API is the one failure "
                "mode that costs money while it fails.")
        return Budget(cap_usd=float("inf"))
    cap = float(cap)
    if cap <= 0:
        raise ValueError(f"budget_usd must be positive, not {cap}")
    return Budget(cap_usd=cap)
