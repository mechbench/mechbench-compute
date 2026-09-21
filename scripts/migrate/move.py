"""Move operations, and the helpers they share, into files of their own
(docs/OPS_LAYOUT.md).

    python scripts/migrate/move.py intervene/patch records/summarize
    python scripts/migrate/move.py --dry intervene/patch
    python scripts/migrate/move.py --family intervene

Text-preserving: a definition is lifted by line range, so its comments
and formatting travel with it byte for byte. The only text rewritten is
what the new layout changes the meaning of — an executor method becomes
`run(ctx, inputs, params)`, so `self._model_loaded` becomes `ctx.model`
and a lent keyword becomes `ctx.<name>` — and every rewrite is a splice
at an AST position, never a regex over code.

A helper two or more operations share goes to `<topic>/<name>.py`, and
the module it left becomes that package's `__init__.py`, which imports
the helper back so nothing that names it breaks. What one operation
alone uses goes into that operation's file, and there nothing is
imported back — an operation's file is a leaf — so the places that named
it are rewritten to name its new home.

Afterwards `verify()` proves the move: every definition lifted verbatim
has the same `ast.dump` where it landed as where it came from.
"""
from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plan import PKG, ROOT, Def, Module, analyse, locate, op_path, topic_of  # noqa: E402

LENT = ("on_item", "on_start", "on_checkpoint", "resume_items", "resume_state",
        "secrets", "input_paths", "bindings", "result_base")
HEADER = "from __future__ import annotations\n"

#: The one name a declaration file and a mechanism file bind to different
#: things: `P` is the param constructor in the lexicon, and three
#: mechanism modules alias `points as P`. An operation's file holds both,
#: so the declaration's spelling wins there and the alias is respelled
#: where it is used, at its AST positions. A helper's file holds no
#: declaration and keeps its spelling.
RESPELL = {"P": ("from mechbench_compute import points as P",
                 "hookpoints", "from mechbench_compute import points as hookpoints")}


def dotted(rel: str) -> str:
    stem = rel[:-3]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return "mechbench_compute." + stem.replace("/", ".")


def char_col(line: str, byte_col: int) -> int:
    """AST columns count UTF-8 bytes; a line with a dash in a string
    before the span would be spliced in the wrong place otherwise."""
    return len(line.encode("utf-8")[:byte_col].decode("utf-8"))


def splice(lines: list[str], first: int, reps: list[tuple[int, int, int, str]]) -> list[str]:
    """Apply (lineno, col, end_col, text) replacements to `lines`, whose
    first element is source line `first`. Right to left, so columns hold."""
    out = list(lines)
    for lineno, col, end, text in sorted(reps, key=lambda r: (r[0], r[1]), reverse=True):
        i = lineno - first
        line = out[i]
        a, b = char_col(line, col), char_col(line, end)
        out[i] = line[:a] + text + line[b:]
    return out


def import_table(mod: Module) -> dict[str, str]:
    """name -> the statement that binds it at the top of `mod`."""
    table: dict[str, str] = {}
    for n in mod.tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                bound = a.asname or a.name.split(".")[0]
                stmt = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
                if a.name == "mlx.core" and a.asname == "mx":
                    stmt = "from mechbench_compute._mlx import mx"
                table[bound] = stmt
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module != "__future__":
            for a in n.names:
                table[a.asname or a.name] = (f"from {n.module} import {a.name}"
                                             + (f" as {a.asname}" if a.asname else ""))
    return table


def respelled(mod: Module, d: Def) -> tuple[list[str], set[str]]:
    """`d`'s source lines with any respelling applied, and the import
    statements that respelling needs."""
    table = import_table(mod)
    lines = mod.text[d.start - 1:d.end]
    node = next((n for n in mod.tree.body if (n.end_lineno or 0) == d.end and d.start <= n.lineno <= d.end), None)
    needs: set[str] = set()
    if node is None:
        return lines, needs
    reps = []
    for old_name, (bound_as, new_name, stmt) in RESPELL.items():
        if table.get(old_name) != bound_as:
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id == old_name:
                reps.append((n.lineno, n.col_offset, n.end_col_offset, new_name))
        if reps:
            needs.add(stmt)
    return (splice(lines, d.start, reps) if reps else lines), needs


def signature_end(text: list[str]) -> tuple[int, int]:
    """(line index, column) just past the colon that ends a `def` header."""
    depth = 0
    for tok in tokenize.generate_tokens(io.StringIO("\n".join(text) + "\n").readline):
        if tok.type == tokenize.OP:
            if tok.string in "([{":
                depth += 1
            elif tok.string in ")]}":
                depth -= 1
            elif tok.string == ":" and depth == 0:
                return tok.end[0] - 1, tok.end[1]
    raise ValueError("no end to the signature")


def run_from_method(mod: Module, d: Def, dest: str, dest_of: dict[tuple[str, str], str],
                    notes: list[str]) -> tuple[str, set[tuple[str, str]]]:
    node = next(n for cls in mod.tree.body if isinstance(cls, ast.ClassDef)
                for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == d.name)
    args = [a.arg for a in node.args.args + node.args.kwonlyargs]
    unknown = [a for a in args if a not in ("self", "inputs", "params", *LENT)]
    if unknown or node.args.vararg or node.args.kwarg:
        notes.append(f"{d.name}: signature takes {unknown or 'star-args'} — by hand")
    lent = {a for a in args if a in LENT}
    for sub in ast.walk(node):
        if sub is not node and isinstance(sub, (ast.FunctionDef, ast.Lambda)):
            inner = {a.arg for a in sub.args.args + sub.args.kwonlyargs}
            if inner & lent:
                notes.append(f"{d.name}: a nested function rebinds {sorted(inner & lent)} — check by hand")
    reps: list[tuple[int, int, int, str]] = []
    wanted: set[tuple[str, str]] = set()
    claimed: set[int] = set()
    local_names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            base = n.value.id
            if base == "self":
                text = "ctx.model" if n.attr == "_model_loaded" else f"ctx.executor.{n.attr}"
                reps.append((n.lineno, n.col_offset, n.end_col_offset, text))
                claimed.add(id(n.value))
            elif base in mod.aliases and dest_of.get((mod.aliases[base], n.attr)) == dest:
                # `interp.patch_trace(` names something landing in this
                # very file, where nothing imports it back: name it
                # directly. A shared helper is left as `alias.name` — it
                # is still reachable through its package, and leaving it
                # is verbatim. (`plan = intervene_mod.plan(...)` would
                # become `plan = plan(...)`, which shadows itself.)
                if n.attr in local_names:
                    notes.append(f"{d.name}: `{base}.{n.attr}` lands in this file but `{n.attr}` is also a local name — by hand")
                    continue
                reps.append((n.lineno, n.col_offset, n.end_col_offset, n.attr))
                claimed.add(id(n.value))
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and id(n) not in claimed:
            if n.id in lent and isinstance(n.ctx, ast.Load):
                reps.append((n.lineno, n.col_offset, n.end_col_offset, f"ctx.{n.id}"))
            elif n.id == "self":
                reps.append((n.lineno, n.col_offset, n.end_col_offset, "ctx.executor"))
    body = splice(mod.text[d.start - 1:d.end], d.start, reps)
    lead = node.lineno - d.start          # comment lines above the def
    i, col = signature_end(body[lead:])
    header = body[lead + i][col:]
    indent = len(body[lead]) - len(body[lead].lstrip())
    body = body[:lead] + [" " * indent + "def run(ctx, inputs, params):" + header] + body[lead + i + 1:]
    body = [ln[indent:] if ln[:indent].strip() == "" else ln for ln in body]
    return "\n".join(body) + "\n", wanted


def run_from_registry(mod: Module, d: Def, value: ast.expr, notes: list[str]) -> str:
    if isinstance(value, ast.Lambda):
        if [a.arg for a in value.args.args] != ["inputs", "params"]:
            notes.append(f"{d.name}: registry lambda takes {[a.arg for a in value.args.args]} — by hand")
        body = ast.get_source_segment(mod.src, value.body) or ""
        body = re.sub(r"\n\s+", "\n        ", body)
        return f"def run(ctx, inputs, params):\n    return {body}\n"
    if isinstance(value, ast.Name):
        return f"def run(ctx, inputs, params):\n    return {value.id}(inputs, params)\n"
    notes.append(f"{d.name}: registry entry is a {type(value).__name__} — by hand")
    return ""


def names_used(code: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(code)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    defined: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name):
                    defined.add(t.id)
    return used, defined


def remove_ranges(text: list[str], ranges: list[tuple[int, int]]) -> list[str]:
    drop: set[int] = set()
    for a, b in ranges:
        drop.update(range(a, b + 1))
    kept = [ln for i, ln in enumerate(text, 1) if i not in drop]
    out: list[str] = []
    for ln in kept:                                  # no run of more than two blank lines
        if ln.strip() == "" and len(out) >= 2 and out[-1].strip() == "" and out[-2].strip() == "":
            continue
        out.append(ln)
    return out


def registry_entry_range(mod: Module, op: str) -> tuple[int, int] | None:
    for n in ast.walk(mod.tree):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value == op and isinstance(v, (ast.Lambda, ast.Name)):
                    return mod._lead(k.lineno), v.end_lineno or v.lineno
    return None


def dispatch_branch_range(mod: Module, op: str) -> tuple[int, int] | None:
    for n in ast.walk(mod.tree):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == "block"
                and len(n.test.comparators) == 1 and isinstance(n.test.comparators[0], ast.Constant)
                and n.test.comparators[0].value == op):
            return n.lineno, n.body[-1].end_lineno or n.lineno
    return None


def insert_after_imports(text: list[str], stmts: list[str]) -> list[str]:
    """After the last top-level import, found in the syntax tree: a
    parenthesised import spans lines, and its last line is not one that
    starts with `from`."""
    tops = [n for n in ast.parse("\n".join(text)).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    last = max((n.end_lineno or n.lineno for n in tops), default=0)
    if not tops:                                      # below the module docstring, if any
        body = ast.parse("\n".join(text)).body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            last = body[0].end_lineno or 0
    return text[:last] + stmts + text[last:]


def move(op_names: list[str], dry: bool) -> None:
    a = analyse()
    dest_of = a.dest_of()
    staying = {(d.file, d.name) for d in a.stays}
    notes: list[str] = []
    new_text: dict[str, list[str]] = defaultdict(list)      # dest -> blocks
    tails: dict[str, list[str]] = defaultdict(list)         # dest -> blocks that must come last
    sources: dict[str, set[str]] = defaultdict(set)         # dest -> source files
    wanted: dict[str, set[tuple[str, str]]] = defaultdict(set)
    extra_imports: dict[str, set[str]] = defaultdict(set)
    removed: dict[str, list[tuple[int, int]]] = defaultdict(list)
    moved: dict[str, list[tuple[Def, str]]] = defaultdict(list)   # source file -> (def, dest)
    verbatim: list[tuple[Def, str]] = []
    done: set[tuple[str, str]] = set()

    for op in op_names:
        if op not in a.pieces:
            sys.exit(f"{op}: not an operation still to move")
        dest = op_path(op)
        roots = a.pieces[op]
        travelling = [d for m in a.mods.values() for d in m.defs.values()
                      if op in d.users and (d.file, d.name) in dest_of]
        # In an operation's file: the contract, the entry point, the mechanism.
        ordered = ([d for d in travelling if d.file.startswith("lexicon/")]
                   + [r for r in roots if r.kind in ("method", "registry")]
                   + [d for d in travelling if not d.file.startswith("lexicon/")])
        entry_made = False
        for d in ordered:
            m = a.mods[d.file]
            if d.kind == "method":
                if entry_made:
                    notes.append(f"{op}: a second executor method, {d.name} — by hand")
                    continue
                text, want = run_from_method(m, d, dest, dest_of, notes)
                new_text[dest].append(text)
                wanted[dest] |= want
                sources[dest].add(d.file)
                removed[d.file].append((d.start, d.end))
                entry_made = True
                continue
            if d.kind == "registry":
                table, value = m.registry[op]
                if table == "MONOIDS":
                    # Last in the file: it names a class defined below the entry point.
                    tails[dest].append(f"MONOID = {ast.get_source_segment(m.src, value)}\n")
                elif not entry_made:
                    new_text[dest].append(run_from_registry(m, d, value, notes))
                    entry_made = True
                sources[dest].add(d.file)
                rng = registry_entry_range(m, op)
                if rng:
                    removed[d.file].append(rng)
                continue
            if (d.file, d.name) in done:
                continue
            done.add((d.file, d.name))
            where = dest_of[(d.file, d.name)]
            if where.startswith("ops/"):
                lines, needs = respelled(m, d)      # a declaration shares this file
                extra_imports[where] |= needs
            else:
                lines = m.text[d.start - 1:d.end]
            block = "\n".join(lines) + "\n"
            if d.kind == "declaration":
                block = re.sub(rf"^{re.escape(d.name)}\b", "OP", block, count=1, flags=re.M)
            new_text[where].append(block)
            sources[where].add(d.file)
            removed[d.file].append((d.start, d.end))
            moved[d.file].append((d, where))
            verbatim.append((d, where))
        if not entry_made:
            notes.append(f"{op}: no entry point found to make `run` from — by hand")
        executor = next((f for f, m in a.mods.items() if m.hosts), None)
        rng = dispatch_branch_range(a.mods[executor], op) if executor else None
        if rng:
            removed[executor].append(rng)

    # ---- the new files ------------------------------------------------------
    outputs: dict[str, str] = {}
    for dest, late in tails.items():
        new_text[dest].extend(late)
    for dest, blocks in new_text.items():
        path = PKG / dest
        existing = path.read_text() if path.exists() else ""
        body = "\n\n".join(b.rstrip("\n") + "\n" for b in blocks)
        used, defined = names_used(re.sub(r"^from __future__.*\n", "", existing) + "\n" + body)
        lines: set[str] = set()
        for src in sorted(sources[dest]):
            m = a.mods[src]
            table = import_table(m)
            for name in used - defined:
                if (src, name) in dest_of and dest_of[(src, name)] != dest:
                    lines.add(f"from {dotted(dest_of[(src, name)])} import {name}")
                elif name in table:
                    lines.add(table[name])
                elif (src, name) in staying:
                    if m.hosts:
                        notes.append(f"{dest}: borrows `{name}` from the executor — import it inside the function, by hand")
                    else:
                        lines.add(f"from {dotted(src)} import {name}")
        for src, name in wanted[dest]:
            lines.add(f"from {dotted(dest_of[(src, name)])} import {name}")
        if extra_imports[dest]:
            lines |= extra_imports[dest]
            for _old, (bound_as, _new, _stmt) in RESPELL.items():
                lines.discard(bound_as)
        already = set(re.findall(r"^(?:from|import) .*$", existing, re.M))
        fresh = sorted(ln for ln in lines if ln not in already)
        if existing:
            merged = insert_after_imports(existing.rstrip("\n").split("\n"), fresh)
            text = "\n".join(merged) + "\n\n\n" + body
        else:
            text = HEADER + ("\n" + "\n".join(fresh) + "\n" if fresh else "") + "\n\n" + body
        outputs[dest] = text

    # ---- the files things left ---------------------------------------------
    conversions: list[tuple[str, str]] = []              # module.py -> package/__init__.py
    for src, ranges in removed.items():
        m = a.mods[src]
        text = remove_ranges(m.text, ranges)
        back: dict[str, list[str]] = defaultdict(list)
        for d, where in moved.get(src, []):
            # A helper is imported back, so whatever names it through
            # this module still finds it. An operation's file is a leaf:
            # nothing is imported back from there.
            if not where.startswith("ops/"):
                back[where].append(d.name)
        if src.startswith("lexicon/"):
            gone = {d.name for d, _ in moved.get(src, []) if d.kind == "declaration"}
            joined = "\n".join(text)
            start = joined.index("\nOPS")
            tail = joined[start:]
            for name in gone:
                tail = re.sub(rf"\b{name}\b,?[ \t]*", "", tail, count=1)
            text = (joined[:start] + tail).split("\n")
        if back:
            stmts = [f"from {dotted(where)} import {', '.join(sorted(names))}  # noqa: F401"
                     for where, names in sorted(back.items())]
            text = insert_after_imports(text, stmts)
        target = src
        if back and not src.endswith("/__init__.py"):
            target = f"{topic_of(src)}/__init__.py"
            conversions.append((src, target))
        outputs[target] = "\n".join(text) + "\n"

    leaf_moves = {(d.file, d.name): where for src in moved for d, where in moved[src] if where.startswith("ops/")}
    rewrites = rewrite_references(leaf_moves, a, notes, outputs, conversions)

    print(f"moving {', '.join(op_names)}")
    made = sorted(new_text)
    print(f"   {sum(1 for d in made if d.startswith('ops/'))} operation file(s), "
          f"{sum(1 for d in made if not d.startswith('ops/'))} helper file(s)")
    for src, target in conversions:
        print(f"   {src} becomes {target}")
    if rewrites:
        print(f"   {sum(rewrites.values())} reference(s) rewritten in {len(rewrites)} file(s)")
    for note in sorted(set(notes)):
        print(f"   !! {note}")
    if dry:
        return

    for src, target in conversions:
        (PKG / target).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(ROOT), "mv", f"mechbench_compute/{src}", f"mechbench_compute/{target}"], check=True)
    for rel, text in outputs.items():
        path = (ROOT / rel) if rel.startswith(("tests/", "scripts/")) else (PKG / rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        init = path.parent / "__init__.py"
        if not rel.startswith(("tests/", "scripts/")) and not init.exists():
            init.write_text("")
        path.write_text(text)
    repoint_sites(op_names, conversions)
    verify(verbatim, a)


def rewrite_references(leaf_moves: dict[tuple[str, str], str], a, notes: list[str],
                       outputs: dict[str, str], conversions: list[tuple[str, str]]) -> dict[str, int]:
    """What one operation alone uses has gone into that operation's file,
    and nothing is imported back from there. So every other file that
    named it through its old module is changed to name its new home."""
    if not leaf_moves:
        return {}
    by_module: dict[str, dict[str, str]] = defaultdict(dict)       # old dotted module -> name -> new dotted
    for (src, name), where in leaf_moves.items():
        by_module[dotted(src)][name] = dotted(where)
    converted = dict(conversions)
    counts: dict[str, int] = {}
    files = [p for base in ("mechbench_compute", "tests", "scripts") for p in (ROOT / base).rglob("*.py")]
    # Files this move creates do not exist yet, and a body lifted into
    # one may import, lazily, something that has moved as well.
    files += [PKG / rel for rel in outputs if not (PKG / rel).exists()]
    for path in files:
        rel_root = str(path.relative_to(ROOT))
        rel_pkg = str(path.relative_to(PKG)) if rel_root.startswith("mechbench_compute/") else None
        key = converted.get(rel_pkg, rel_pkg) if rel_pkg else rel_root
        src_text = outputs[key] if (rel_pkg and key in outputs) else path.read_text()
        if not any(name in src_text for names in by_module.values() for name in names):
            continue
        try:
            tree = ast.parse(src_text)
        except SyntaxError:
            continue
        lines = src_text.split("\n")
        reps: list[tuple[int, int, int, str]] = []
        drop_lines: set[int] = set()
        add: set[str] = set()
        aliases: dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                if n.module == "mechbench_compute":
                    for al in n.names:
                        aliases[al.asname or al.name] = f"mechbench_compute.{al.name}"
                elif n.module in by_module:
                    table = by_module[n.module]
                    going = [al for al in n.names if al.name in table]
                    if not going:
                        continue
                    staying_names = [al for al in n.names if al.name not in table]
                    here = dotted(rel_pkg) if rel_pkg else None
                    # What now lives in this same file needs no import at all.
                    going = [al for al in going if table[al.name] != here or al.asname]
                    for al in going:
                        add.add(f"from {table[al.name]} import {al.name}" + (f" as {al.asname}" if al.asname else ""))
                    indent = " " * n.col_offset
                    first, last = n.lineno, n.end_lineno or n.lineno
                    drop_lines.update(range(first, last + 1))
                    if not going and not staying_names and n.col_offset:
                        reps.append((first, -1, -1, f"{indent}pass  # its imports now live in this file"))
                    if staying_names:
                        kept = ", ".join(al.name + (f" as {al.asname}" if al.asname else "") for al in staying_names)
                        add_here = f"{indent}from {n.module} import {kept}"
                        reps.append((first, -1, -1, add_here))
                    if n.col_offset:                       # an import inside a function stays inside it
                        for al in going:
                            stmt = f"{indent}from {table[al.name]} import {al.name}" + (f" as {al.asname}" if al.asname else "")
                            reps.append((first, -2, -2, stmt))
                            add.discard(stmt.strip())
            elif isinstance(n, ast.Import):
                for al in n.names:
                    if al.asname and al.name in by_module:
                        aliases[al.asname] = al.name
        uses = 0
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                module = aliases.get(n.value.id)
                if module in by_module and n.attr in by_module[module]:
                    reps.append((n.lineno, n.col_offset, n.end_col_offset, n.attr))
                    add.add(f"from {by_module[module][n.attr]} import {n.attr}")
                    uses += 1
            elif isinstance(n, ast.Call):
                # A name passed as a string to something that patches or
                # looks up attributes: `monkeypatch.setattr(interp, "x", …)`.
                # A rewrite cannot know what that means now; a person can.
                called = ast.unparse(n.func)
                if not re.search(r"setattr|getattr|patch|delattr", called):
                    continue
                for arg in n.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        for module, table in by_module.items():
                            for name in table:
                                if arg.value == name or arg.value.endswith(f".{name}"):
                                    notes.append(f"{rel_root}:{n.lineno} `{called}(…, {arg.value!r})` names something that moved — by hand")
        if not reps and not drop_lines and not add:
            continue
        # Attribute uses first (they are splices within a line), then imports (whole lines).
        spliced = splice(lines, 1, [r for r in reps if r[1] >= 0])
        inserts: dict[int, list[str]] = defaultdict(list)
        for lineno, col, _e, text in reps:
            if col < 0:
                inserts[lineno].append(text)
        out: list[str] = []
        for i, ln in enumerate(spliced, 1):
            if i in inserts:
                out.extend(inserts[i])
            if i not in drop_lines:
                out.append(ln)
        if add:
            out = insert_after_imports(out, sorted(add))
        result = "\n".join(out)
        if result != src_text:
            outputs[key if rel_pkg else rel_root] = result
            counts[rel_root] = len(drop_lines) and 1 or 0
            counts[rel_root] += uses
    return counts


def repoint_sites(op_names: list[str], conversions: list[tuple[str, str]]) -> None:
    """The params gate reads each operation's code where SITES says it
    is. A moved operation is every function in its own file, and a module
    that became a package is named by its directory."""
    path = ROOT / "tests" / "test_block_params.py"
    text = path.read_text()
    for src, target in conversions:
        text = text.replace(f'"{src}"', f'"{target}"')
    for op in op_names:
        rel = op_path(op)
        tree = ast.parse((PKG / rel).read_text())
        fns = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        fns.sort(key=lambda n: n != "run")
        sites = ",\n        ".join(f'("{rel}", "{fn}")' for fn in fns)
        entry = f'    "{op}": [\n        {sites}],\n'
        text, n = re.subn(rf'^    "{re.escape(op)}": \[.*?\)\],\n', entry, text, count=1, flags=re.M | re.S)
        if n != 1:
            print(f"   !! {op}: no SITES entry found to repoint")
    path.write_text(text)


def verify(verbatim: list[tuple[Def, str]], a) -> None:
    """Every definition moved verbatim is the same code where it landed."""
    def find(tree: ast.Module, name: str):
        for n in tree.body:
            if getattr(n, "name", None) == name:
                return n
            targets = getattr(n, "targets", None) or ([n.target] if isinstance(n, ast.AnnAssign) else [])
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return n
        return None

    class _NoPlumbing(ast.NodeTransformer):
        """Imports from this package, and the `pass` left where one was
        the only statement, are where code lives, not what it does."""
        def visit_ImportFrom(self, n):
            return None if (n.module or "").startswith("mechbench_compute") else n

        def visit_Pass(self, n):
            return None

    def dump(node: ast.AST) -> str:
        return ast.dump(_NoPlumbing().visit(ast.parse(ast.unparse(node))))

    bad = 0
    for d, where in verbatim:
        old = find(a.mods[d.file].tree, d.name)
        new = find(ast.parse((PKG / where).read_text()), "OP" if d.kind == "declaration" else d.name)
        if old is None or new is None:
            print(f"   ✗ {d.name} did not land in {where}")
            bad += 1
            continue
        od, nd = dump(old), dump(new)
        if d.kind == "declaration":
            od = od.replace(f"id='{d.name}'", "id='OP'", 1)
        if where.startswith("ops/"):
            for old_name, (bound_as, new_name, _stmt) in RESPELL.items():
                if import_table(a.mods[d.file]).get(old_name) == bound_as:
                    od = od.replace(f"Name(id='{old_name}'", f"Name(id='{new_name}'")
        if od != nd:
            print(f"   ✗ {d.name} differs after the move")
            bad += 1
    print(f"   verified {len(verbatim) - bad} of {len(verbatim)} definitions identical by AST")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--family" in sys.argv:
        family = args.pop(0)
        args = sorted(op for op in analyse().pieces if op.split("/")[0] == family)
    move(args, dry="--dry" in sys.argv)
