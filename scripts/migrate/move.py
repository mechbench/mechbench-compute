"""Move operations into their own files (docs/OPS_LAYOUT.md).

    python scripts/migrate/move.py intervene/patch records/summarize
    python scripts/migrate/move.py --dry intervene/patch

Text-preserving: a definition is lifted by line range, so its comments
and formatting travel with it byte for byte. The only text rewritten is
what the new layout changes the meaning of — an executor method becomes
`run(ctx, inputs, params)`, so `self._model_loaded` becomes `ctx.model`
and a lent keyword becomes `ctx.<name>` — and every rewrite is a splice
at an AST position, never a regex over code.

Afterwards `verify()` proves it: every definition moved verbatim has the
same `ast.dump` in its new file as in its old one.
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
from plan import PKG, ROOT, Def, Module, analyse, op_path  # noqa: E402

LENT = ("on_item", "on_start", "on_checkpoint", "resume_items", "resume_state",
        "secrets", "input_paths", "bindings", "result_base")
HEADER = "from __future__ import annotations\n"

#: The one name a declaration file and a mechanism file bind to different
#: things: `P` is the param constructor in the lexicon, and three
#: mechanism modules alias `points as P`. An operation's file holds both,
#: so the declaration's spelling wins and the alias is respelled where it
#: is used, at its AST positions.
RESPELL = {"P": ("from mechbench_compute import points as P",
                 "hookpoints", "from mechbench_compute import points as hookpoints")}


def respelled(mod: Module, d: Def) -> tuple[list[str], set[str]]:
    """`d`'s source lines with any respelling applied, and the import
    statements that respelling needs."""
    table = import_table(mod)
    lines = mod.text[d.start - 1:d.end]
    needs: set[str] = set()
    node = next((n for n in mod.tree.body if (n.end_lineno or 0) == d.end and n.lineno <= d.end
                 and n.lineno >= d.start), None)
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


def dotted(rel: str) -> str:
    return "mechbench_compute." + rel[:-3].replace("/", ".")


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


def module_aliases(mod: Module, fn: ast.AST | None) -> dict[str, str]:
    """Names bound to a `mechbench_compute` module: at the top of the
    file, and lazily inside `fn`."""
    found: dict[str, str] = {}
    scopes = [mod.tree.body]
    if fn is not None:
        scopes.append([n for n in ast.walk(fn)])
    for scope in scopes:
        for n in scope:
            if isinstance(n, ast.ImportFrom) and n.module == "mechbench_compute":
                for a in n.names:
                    found[a.asname or a.name] = f"{a.name}.py"
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("mechbench_compute.") and a.asname:
                        found[a.asname] = a.name.split(".", 1)[1].replace(".", "/") + ".py"
    return found


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
    aliases = module_aliases(mod, node)
    reps: list[tuple[int, int, int, str]] = []
    wanted: set[tuple[str, str]] = set()
    claimed: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            base = n.value.id
            if base == "self":
                text = "ctx.model" if n.attr == "_model_loaded" else f"ctx.executor.{n.attr}"
                reps.append((n.lineno, n.col_offset, n.end_col_offset, text))
                claimed.add(id(n.value))
            elif base in aliases and (aliases[base], n.attr) in dest_of:
                reps.append((n.lineno, n.col_offset, n.end_col_offset, n.attr))
                claimed.add(id(n.value))
                if dest_of[(aliases[base], n.attr)] != dest:
                    wanted.add((aliases[base], n.attr))
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


def move(op_names: list[str], dry: bool) -> None:
    a = analyse()
    dest_of = a.dest_of()
    notes: list[str] = []
    new_text: dict[str, list[str]] = defaultdict(list)      # dest -> blocks
    tails: dict[str, list[str]] = defaultdict(list)         # dest -> blocks that must come last
    extra_imports: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)         # dest -> source files
    wanted: dict[str, set[tuple[str, str]]] = defaultdict(set)
    removed: dict[str, list[tuple[int, int]]] = defaultdict(list)
    moved_defs: dict[str, list[tuple[Def, str]]] = defaultdict(list)   # source file -> (def, dest)
    verbatim: list[tuple[Def, str]] = []

    for op in op_names:
        if op not in a.pieces:
            sys.exit(f"{op}: not an operation still to move")
        dest = op_path(op)
        roots = a.pieces[op]
        travelling = [d for m in a.mods.values() for d in m.defs.values() if op in d.users]
        # The contract first, then the entry point, then the mechanism.
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
            where = dest_of[(d.file, d.name)]
            if any(d is x for x, _ in moved_defs[d.file]):
                continue
            lines, needs = respelled(m, d)
            extra_imports[where] |= needs
            block = "\n".join(lines) + "\n"
            if d.kind == "declaration":
                block = re.sub(rf"^{re.escape(d.name)}\b", "OP", block, count=1, flags=re.M)
            new_text[where].append(block)
            sources[where].add(d.file)
            removed[d.file].append((d.start, d.end))
            moved_defs[d.file].append((d, where))
            verbatim.append((d, where))
        if not entry_made:
            notes.append(f"{op}: no entry point found to make `run` from — by hand")
        rng = dispatch_branch_range(a.mods["protocol.py"], op) if "protocol.py" in a.mods else None
        if rng:
            removed["protocol.py"].append(rng)

    # ---- write the new files ------------------------------------------------
    outputs: dict[str, str] = {}
    for dest, late in tails.items():
        new_text[dest].extend(late)
    for dest, blocks in new_text.items():
        path = PKG / dest
        existing = path.read_text() if path.exists() else ""
        body = "\n\n".join(b.rstrip("\n") + "\n" for b in blocks)
        used, defined = names_used((existing and re.sub(r"^from __future__.*\n", "", existing)) + "\n" + body)
        lines: set[str] = set()
        for src in sorted(sources[dest]):
            table = import_table(a.mods[src])
            for name in used - defined:
                if name in table:
                    lines.add(table[name])
                elif (src, name) in dest_of and dest_of[(src, name)] != dest:
                    lines.add(f"from {dotted(dest_of[(src, name)])} import {name}")
        for src, name in wanted[dest]:
            lines.add(f"from {dotted(dest_of[(src, name)])} import {name}")
        lines |= extra_imports[dest]
        for old_name, (bound_as, _new, _stmt) in RESPELL.items():
            if extra_imports[dest]:
                lines.discard(bound_as)
        already = set(re.findall(r"^(?:from|import) .*$", existing, re.M))
        fresh = sorted(ln for ln in lines if ln not in already)
        if existing:
            head, _, tail = existing.partition("\n\n\n")
            text = head + ("\n" + "\n".join(fresh) if fresh else "") + "\n\n\n" + tail.rstrip("\n") + "\n\n\n" + body
        else:
            text = HEADER + "\n" + "\n".join(fresh) + "\n\n\n" + body
        outputs[dest] = text

    # ---- edit the old files -------------------------------------------------
    for src, ranges in removed.items():
        m = a.mods[src]
        text = remove_ranges(m.text, ranges)
        back = defaultdict(list)
        for d, where in moved_defs.get(src, []):
            # A declaration leaves its family file for good — its entry
            # in OPS goes with it. A shared fragment is still needed by
            # the declarations that have not moved yet.
            if d.kind != "declaration":
                back[where].append(d.name)
        if src.startswith("lexicon/"):
            gone = {d.name for d, _ in moved_defs.get(src, []) if d.kind == "declaration"}
            joined = "\n".join(text)
            start = joined.index("\nOPS")
            tail = joined[start:]
            for name in gone:
                tail = re.sub(rf"\b{name}\b,?[ \t]*", "", tail, count=1)
            text = (joined[:start] + tail).split("\n")
        if back:
            # After the last top-level import, found in the syntax tree:
            # a parenthesised import spans lines, and its last line is
            # not one that starts with `from`.
            tops = [n for n in ast.parse("\n".join(text)).body if isinstance(n, (ast.Import, ast.ImportFrom))]
            last = (max(n.end_lineno or n.lineno for n in tops) - 1) if tops else 0
            stmts = [f"from {dotted(where)} import (  # noqa: F401 - moved; see docs/OPS_LAYOUT.md\n    "
                     + ",\n    ".join(sorted(names)) + ",\n)" for where, names in sorted(back.items())]
            text = text[:last + 1] + stmts + text[last + 1:]
        outputs[src] = "\n".join(text) + "\n"

    print(f"moving {', '.join(op_names)}")
    for dest in sorted(new_text):
        print(f"   -> {dest:36} {len(outputs[dest].splitlines()):5} lines")
    for note in notes:
        print(f"   !! {note}")
    if dry:
        return
    for rel, text in outputs.items():
        path = PKG / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        init = path.parent / "__init__.py"
        if rel.startswith("ops/") and not init.exists():
            init.write_text("")
        path.write_text(text)
    repoint_sites(op_names)
    verify(verbatim, a)


def repoint_sites(op_names: list[str]) -> None:
    """The params gate reads each operation's code where SITES says it
    is. A moved operation is every function in its own file."""
    path = ROOT / "tests" / "test_block_params.py"
    text = path.read_text()
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
    bad = 0
    for d, where in verbatim:
        old = next(n for n in a.mods[d.file].tree.body
                   if getattr(n, "name", None) == d.name
                   or any(isinstance(t, ast.Name) and t.id == d.name
                          for t in getattr(n, "targets", [getattr(n, "target", None)]) if t is not None))
        new_tree = ast.parse((PKG / where).read_text())
        want = "OP" if d.kind == "declaration" else d.name
        new = next((n for n in new_tree.body
                    if getattr(n, "name", None) == want
                    or any(isinstance(t, ast.Name) and t.id == want
                           for t in getattr(n, "targets", [getattr(n, "target", None)]) if t is not None)), None)
        if new is None:
            print(f"   ✗ {d.name} did not land in {where}")
            bad += 1
            continue
        od, nd = ast.dump(old), ast.dump(new)
        for old_name, (bound_as, new_name, _stmt) in RESPELL.items():
            if import_table(a.mods[d.file]).get(old_name) == bound_as:
                od = od.replace(f"Name(id='{old_name}'", f"Name(id='{new_name}'")
        if d.kind == "declaration":
            od = od.replace(f"id='{d.name}'", "id='OP'", 1)
        if od != nd:
            print(f"   ✗ {d.name} differs after the move")
            bad += 1
    print(f"   verified {len(verbatim) - bad} of {len(verbatim)} definitions identical by AST")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    move(args, dry="--dry" in sys.argv)
