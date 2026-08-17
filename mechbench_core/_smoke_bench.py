"""Smoke test for the bench client (task 000238).

Requires a running mechbench-api with MECHBENCH_API_URL and
MECHBENCH_API_KEY set; SKIPs cleanly when they aren't (offline interp
work must never depend on a server).

Run: MECHBENCH_API_URL=... MECHBENCH_API_KEY=... \
     python -m mechbench_core._smoke_bench
"""

from __future__ import annotations

import os
import sys

PASS, FAIL = "PASS", "FAIL"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}"
          + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def main() -> None:
    if not (os.environ.get("MECHBENCH_API_URL")
            and os.environ.get("MECHBENCH_API_KEY")):
        print("SKIP: MECHBENCH_API_URL / MECHBENCH_API_KEY not set")
        sys.exit(0)

    from mechbench_core import bench

    base = "benji/bench-smoke"
    parent = f"{base}/inputs/seed"
    child = f"{base}/results/derived"

    r1 = bench.emit(parent, {"kind": "smoke-seed", "values": [1, 2, 3]},
                    params={"n": 3})
    check("emit parent", r1.get("path") == parent and
          r1.get("hash", "").startswith("sha256:"), r1.get("hash", "")[:18])

    r2 = bench.emit(child, {"kind": "smoke-derived",
                            "items": [{"text": f"item {i}"} for i in range(7)]},
                    inputs=[parent], fidelity="text")
    check("emit child with input", r2.get("lineageParents") == 1)

    back = bench.fetch(parent)
    check("round-trip", back["payload"]["values"] == [1, 2, 3])

    page = bench.fetch_items(child, offset=2, limit=3)
    check("item pagination", page["total"] == 7 and len(page["items"]) == 3
          and page["items"][0]["text"] == "item 2")

    ls = bench.listing(base)
    check("listing", {o["path"] for o in ls["objects"]} >= {parent, child})

    up = bench.lineage(child, "up")
    check("lineage up", {(e["child"], e["parent"]) for e in up["edges"]}
          == {(child, parent)})
    down = bench.lineage(parent, "down")
    check("lineage down", any(e["child"] == child for e in down["edges"]))

    try:
        bench.emit("someone-else/p/x", {"v": 1})
        check("foreign namespace rejected", False)
    except bench.BenchError as e:
        check("foreign namespace rejected", "FORBIDDEN" in str(e))

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + str(failures)}",
          flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
