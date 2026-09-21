"""What is shared between operations, who uses it, and what copying it
into each operation's file would cost. Analysis only."""
from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plan import analyse  # noqa: E402

a = analyse()
mods = a.mods
shared = [d for m in mods.values() for d in m.defs.values() if len(d.users) >= 2]


def sites_of(d) -> int:
    """Places in the analysed files that refer to `d`, its own definition aside."""
    n = 0
    for m in mods.values():
        for node in ast.walk(m.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == d.name:
                if m.rel == d.file or m.imported.get(node.id) == (d.file, d.name):
                    n += 1
            elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.attr == d.name and m.aliases.get(node.value.id) == d.file):
                n += 1
    return n


rows = sorted(((d, sites_of(d)) for d in shared), key=lambda r: (-r[0].lines, r[0].name))
print(f"{len(shared)} definitions are used by two or more operations — {sum(d.lines for d in shared)} lines\n")

print("BY SIZE")
print(f"  {'lines':>11} {'defs':>5} {'total lines':>12} {'median ops using':>17} {'cost to copy into every user':>30}")
for lo, hi in ((1, 5), (6, 15), (16, 40), (41, 100), (101, 10_000)):
    grp = [d for d in shared if lo <= d.lines <= hi]
    if not grp:
        continue
    users = sorted(len(d.users) for d in grp)
    dup = sum(d.lines * (len(d.users) - 1) for d in grp)
    label = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    print(f"  {label:>11} {len(grp):5} {sum(d.lines for d in grp):12} {users[len(users)//2]:17} {'+' + str(dup):>30}")

print("\nBY WHAT IT IS")
kinds = defaultdict(list)
for d in shared:
    where = "declaration fragment (a shared port or param)" if d.file.startswith("lexicon/") else {
        "function": "helper function", "class": "class", "assign": "constant or table"}[d.kind]
    kinds[where].append(d)
for k, grp in sorted(kinds.items(), key=lambda kv: -sum(d.lines for d in kv[1])):
    print(f"  {k:48} {len(grp):3} defs  {sum(d.lines for d in grp):5} lines")

print("\nTHE LARGEST (where copying stops being cheap)")
print(f"  {'lines':>5} {'ops':>4} {'sites':>6}  definition")
for d, n in rows[:14]:
    print(f"  {d.lines:5} {len(d.users):4} {n:6}  {d.file}:{d.name}  [{d.kind}]")

print("\nTHE MOST WIDELY USED")
for d, n in sorted(rows, key=lambda r: -len(r[0].users))[:12]:
    print(f"  {d.lines:5} {len(d.users):4} {n:6}  {d.file}:{d.name}  [{d.kind}]")


# ---- what one agent reading one operation pays, under each strategy ---------
SUBSYSTEMS = {"intervene.py", "chat.py", "tools.py"}
alldefs = [d for m in mods.values() for d in m.defs.values()]


def copyable(d) -> bool:
    """A class has identity — `isinstance` and `except` stop working
    across copies — so only functions, constants and declaration
    fragments can be copied, and not out of a subsystem."""
    return d.kind != "class" and d.file not in SUBSYSTEMS


def after_move(copy: bool) -> dict[str, int]:
    out = {}
    for f, m in mods.items():
        gone = sum(d.lines for d in m.defs.values() if len(d.users) == 1)
        gone += sum(r.lines for roots in a.pieces.values() for r in roots
                    if r.file == f and r.kind in ("method", "registry"))
        if copy:
            gone += sum(d.lines for d in m.defs.values() if len(d.users) >= 2 and copyable(d))
        out[f] = len(m.text) - gone
    return out


def med(xs):
    return sorted(xs)[len(xs) // 2]


print("\n\nWHAT ONE AGENT READING ONE OPERATION PAYS (after single-use code has moved)")
for label, copy in (("shared stays in modules", False), ("copy what can be copied", True)):
    left = after_move(copy)
    rows = []
    for op, roots in a.pieces.items():
        own = sum(d.lines for d in alldefs if d.users == {op})
        own += sum(r.lines for r in roots if r.kind in ("method", "registry"))
        mine = [d for d in shared if op in d.users]
        if copy:
            own += sum(d.lines for d in mine if copyable(d))
            mine = [d for d in mine if not copyable(d)]
        files = {d.file for d in mine if not d.file.startswith("protocol")}
        rows.append((own, len(files), sum(left[f] for f in files)))
    print(f"  {label:26} median {med([r[0] + r[2] for r in rows]):5} lines in {med([1 + r[1] for r in rows])} file(s)"
          f"   self-contained: {sum(1 for r in rows if r[1] == 0):2} of {len(rows)}"
          f"   worst {max(r[0] + r[2] for r in rows)}")
cp = [d for d in shared if copyable(d)]
print(f"  copying adds {sum(d.lines * (len(d.users) - 1) for d in cp)} lines to the repository "
      f"({len(cp)} definitions, {sum(d.lines for d in cp)} lines, copied to each user)")
