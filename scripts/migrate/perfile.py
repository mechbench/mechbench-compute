"""Could every shared definition have a file of its own, and what would an
agent reading one operation pay if it did? Analysis only."""
import ast, sys
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from plan import analyse
a = analyse(); mods = a.mods
alldefs = [d for m in mods.values() for d in m.defs.values()]
shared = [d for d in alldefs if len(d.users) >= 2]
key = lambda d: (d.file, d.name)
S = {key(d): d for d in shared}

def deps(d):
    """Shared definitions `d` refers to directly."""
    m = mods[d.file]; out = set()
    for r in d.refs:
        if (d.file, r) in S and r != d.name: out.add((d.file, r))
        elif r in m.imported and m.imported[r] in S: out.add(m.imported[r])
    for alias, attr in d.attr_refs:
        rel = m.aliases.get(alias)
        if (rel, attr) in S: out.add((rel, attr))
    return out
G = {k: deps(d) for k, d in S.items()}

# 1. can each definition have a file of its own? mutual references cannot be split
index = {}; low = {}; stack = []; on = set(); sccs = []; counter = [0]
def strong(v):
    index[v] = low[v] = counter[0]; counter[0] += 1; stack.append(v); on.add(v)
    for w in G[v]:
        if w not in index: strong(w); low[v] = min(low[v], low[w])
        elif w in on: low[v] = min(low[v], index[w])
    if low[v] == index[v]:
        comp = []
        while True:
            w = stack.pop(); on.discard(w); comp.append(w)
            if w == v: break
        sccs.append(comp)
sys.setrecursionlimit(10000)
for v in G:
    if v not in index: strong(v)
knots = [c for c in sccs if len(c) > 1]
print(f"1. CAN EACH HAVE ITS OWN FILE?  {len(S)} shared definitions -> {len(sccs)} files")
print(f"   definitions that refer to each other in a cycle, and so must share a file: {sum(len(c) for c in knots)} in {len(knots)} knots")
for c in knots: print("      ", sorted(n for _, n in c), "in", {f for f, _ in c})

names = Counter(n for _, n in S)
print(f"\n2. NAMES: {sum(1 for c in names.values() if c > 1)} names are used by more than one shared definition:",
      {n: sorted(f for f, m in S if m == n) for n, c in names.items() if c > 1})

# 3. how deep do helpers chain?
depth = {}
def dep_depth(v, seen=()):
    if v in depth: return depth[v]
    if v in seen: return 0
    d = 1 + max((dep_depth(w, seen + (v,)) for w in G[v]), default=0)
    depth[v] = d; return d
for v in G: dep_depth(v)
print(f"\n3. CHAINS: a helper leads to another helper, {Counter(depth.values())} (1 = leads nowhere)")

# 4. what does one agent reading one operation pay?
PREAMBLE = 5   # `from __future__`, its own imports, blank lines
def reach(start):
    seen, st = set(), list(start)
    while st:
        v = st.pop()
        if v in seen: continue
        seen.add(v); st.extend(G[v])
    return seen
med = lambda xs: sorted(xs)[len(xs)//2]
rows = []
for op, roots in a.pieces.items():
    mine = [d for d in alldefs if d.users == {op}] + [r for r in roots if r.kind in ("method", "registry")]
    own = sum(d.lines for d in mine)
    direct = set()
    for d in mine:
        direct |= deps(d) if key(d) not in S else set()
        m = mods[d.file]
        for r in d.refs:
            if (d.file, r) in S: direct.add((d.file, r))
            elif r in m.imported and m.imported[r] in S: direct.add(m.imported[r])
        for alias, attr in d.attr_refs:
            if (m.aliases.get(alias), attr) in S: direct.add((m.aliases.get(alias), attr))
    allr = reach(direct)
    rows.append((op, own, len(direct), len(allr), sum(S[k].lines + PREAMBLE for k in allr)))
print(f"\n4. ONE AGENT, ONE OPERATION — every shared definition in a file of its own")
print(f"   the operation's own file:                       median {med([r[1] for r in rows])} lines")
print(f"   helpers it names directly (what the agent sees): median {med([r[2] for r in rows])}   max {max(r[2] for r in rows)}")
print(f"   helpers reachable through those:                 median {med([r[3] for r in rows])}   max {max(r[3] for r in rows)}")
print(f"   if it opens NONE of them (the names suffice):    median {med([r[1] for r in rows])} lines, 1 file")
print(f"   if it opens EVERY one of them:                   median {med([r[1]+r[4] for r in rows])} lines, {med([1+r[3] for r in rows])} files   worst {max(r[1]+r[4] for r in rows)} lines, {max(1+r[3] for r in rows)} files")
sizes = sorted(d.lines for d in shared)
print(f"   one helper costs:  median {med(sizes)+PREAMBLE} lines to open   (body {med(sizes)} + ~{PREAMBLE} of imports)")
print(f"   helpers whose body is shorter than their own import preamble: {sum(1 for s in sizes if s <= PREAMBLE)} of {len(sizes)}")
