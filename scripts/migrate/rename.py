"""Apply scripts/migrate/renames.tsv: names become verbs, files follow.

    python scripts/migrate/rename.py --dry      # what it would change
    python scripts/migrate/rename.py            # change it, then verify

Every edit is a splice at an AST position, or an import statement
regenerated from its own AST — never a regex over code. What is rewritten
for one row:

  the definition          `def _last_logp(` -> `def read_last_logp(`
  the import              `from ...interp.last_logp import _last_logp`
  the bare name           every Name node in a file that binds it
  the attribute           `interp._last_logp` through a module alias
  the string              `("ops/x/y.py", "old")` in the params gate,
                          and `patch("mechbench_compute.x.y.old")`
  the file                `interp/last_logp.py` -> `interp/read_last_logp.py`
                          (git mv), which also rewrites the package
                          `__init__.py` line that imports it back

A row whose new name is `(inline)` or `(delete)` is not a rename: dead
indirection is removed instead, by hand, in by_hand.py `names()`.

`verify()` then proves the pass was a renaming and nothing else: for
every file it touched, the AST before and the AST after are identical
once both are normalised through the substitutions the pass recorded —
and every substitution it recorded is a row of the table. A file it did
not touch is byte-identical.
"""
from __future__ import annotations

import ast
import csv
import io
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).parent))
from names import PKG, ROOT, Inventory, attr_chain, rel_of_module

TABLE = Path(__file__).parent / "renames.tsv"
SKIP = ("(inline)", "(delete)")
#: The migration scripts are the record of the move; they name the old
#: names on purpose, so they are read but never rewritten.
UNTOUCHED = ("scripts/migrate/",)


def targets() -> list[Path]:
    out = []
    for base in ("mechbench_compute", "tests", "scripts"):
        for path in sorted((ROOT / base).rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            if "__pycache__" in path.parts or rel.startswith(UNTOUCHED):
                continue
            out.append(path)
    return out


def dotted(rel: str) -> str:
    stem = rel[:-3].removesuffix("/__init__")
    return "mechbench_compute" + (f".{stem.replace('/', '.')}" if stem else "")


def char_col(line: str, byte_col: int) -> int:
    """AST columns count UTF-8 bytes."""
    return len(line.encode("utf-8")[:byte_col].decode("utf-8"))


def splice(lines: list[str], reps: list[tuple[int, int, int, str]]) -> list[str]:
    """(lineno, col, end_col, text), applied right to left so columns hold.
    A col of -1 replaces the whole line range lineno..end_col.

    Within-line splices go first: a whole-line replacement can shorten
    the file (a two-line import becomes one), and every line number here
    is a line number in the ORIGINAL text."""
    out = list(lines)
    for lineno, col, end, text in sorted([r for r in reps if r[1] >= 0],
                                         key=lambda r: (r[0], r[1]), reverse=True):
        line = out[lineno - 1]
        a, b = char_col(line, col), char_col(line, end)
        out[lineno - 1] = line[:a] + text + line[b:]
    for lineno, _c, end, text in sorted([r for r in reps if r[1] < 0], key=lambda r: -r[0]):
        out[lineno - 1:end] = [text]
    return out


def aligned_continuations(src: str) -> dict[int, tuple[int, int]]:
    """Continuation lines that are aligned to an opening bracket with
    something after it on its own line: line -> (bracket line, column).

    A rename changes a name's length, so a call written
    `fit_component(vectors, layer=…` with its later arguments under
    `vectors` no longer lines up. Only lines that lined up BEFORE are
    moved; code written with a hanging indent is left as it is."""
    out: dict[int, tuple[int, int]] = {}
    stack: list[tuple[int, int, bool]] = []
    lines = src.split("\n")
    prev_line = 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return out
    for tok in toks:
        if tok.type == tokenize.OP and tok.string in "([{":
            rest = lines[tok.start[0] - 1][tok.start[1] + 1:].strip()
            stack.append((tok.start[0], tok.start[1], bool(rest)))
        elif tok.type == tokenize.OP and tok.string in ")]}":
            if stack:
                stack.pop()
        elif (stack and tok.start[0] != prev_line
              and tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
                                   tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)):
            open_line, open_col, visual = stack[-1]
            if visual and tok.start[0] != open_line and tok.start[0] not in out \
                    and tok.start[1] == open_col + 1:
                out[tok.start[0]] = (open_line, open_col)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev_line = tok.end[0]
    return out


def realign(src: str, reps: list[tuple[int, int, int, str]]) -> list[tuple[int, int, int, str]]:
    """The extra replacements that keep visually aligned arguments under
    the bracket they were under, given the within-line edits `reps`."""
    aligned = aligned_continuations(src)
    if not aligned:
        return []
    lines = src.split("\n")
    by_line: dict[int, list[tuple[int, int]]] = defaultdict(list)   # line -> [(col, delta)]
    for lineno, col, end, text in reps:
        if col >= 0:
            by_line[lineno].append((col, len(text) - (end - col)))
    shift: dict[int, int] = {}
    out: list[tuple[int, int, int, str]] = []
    for cont in sorted(aligned):
        open_line, open_col = aligned[cont]
        moved = shift.get(open_line, 0) + sum(d for c, d in by_line[open_line] if c < open_col)
        shift[cont] = moved
        if moved:
            indent = len(lines[cont - 1]) - len(lines[cont - 1].lstrip())
            out.append((cont, 0, indent, " " * max(0, indent + moved)))
    return out


def spell_import(node, module: str, names: list[tuple[str, str | None]],
                 keyword: str, lines: list[str]) -> str:
    """The statement written out again. A statement that was one line
    stays one line; one that was wrapped in parentheses stays wrapped,
    one name to a line, because that is how the file had it. The comment
    after it comes too — on a package's re-export that comment is the
    `# noqa: F401` that says the import is deliberate."""
    indent = " " * node.col_offset
    last = lines[(node.end_lineno or node.lineno) - 1]
    tail = last[char_col(last, node.end_col_offset or len(last)):]
    spelled = ", ".join(n + (f" as {b}" if b else "") for n, b in names)
    head = f"{indent}{keyword} {module} import " if keyword == "from" else f"{indent}import "
    if (node.end_lineno or node.lineno) == node.lineno and len(head) + len(spelled) <= 99:
        return head + spelled + tail
    body = "".join(f"{indent}    {n}{f' as {b}' if b else ''},\n" for n, b in names)
    return f"{head}(\n{body}{indent}){tail}"


class Renamer:
    def __init__(self) -> None:
        rows = list(csv.DictReader(TABLE.open(), delimiter="\t"))
        self.rows = [r for r in rows if r["new_name"] not in SKIP]
        self.held = [r for r in rows if r["new_name"] in SKIP]
        self.rename: dict[tuple[str, str], str] = {
            (r["old_file"], r["old_name"]): r["new_name"] for r in self.rows}
        self.moves: dict[str, str] = {
            r["old_file"]: r["new_file"] for r in self.rows if r["new_file"] != r["old_file"]}
        self.module_moves = {dotted(a): dotted(b) for a, b in self.moves.items()}
        self.pairs = {(r["old_name"], r["new_name"]) for r in self.rows}
        self.inv = Inventory()
        self.subs: dict[str, dict[str, str]] = {}      # file -> {old: new}
        self.problems: list[str] = []

    # -- what a file sees ------------------------------------------------
    def resolve(self, module: str, name: str) -> tuple[str, str] | None:
        return self.inv.resolve(module, name)

    def new_of(self, target: tuple[str, str] | None) -> str | None:
        return self.rename.get(target) if target else None

    def plan(self, path: Path) -> tuple[str, dict[str, str]] | None:
        """The file's new text and the substitutions it needed."""
        rel_root = str(path.relative_to(ROOT))
        rel = (str(path.relative_to(PKG)) if rel_root.startswith("mechbench_compute/")
               else rel_root)
        src = path.read_text()
        tree = ast.parse(src)
        lines = src.split("\n")
        reps: list[tuple[int, int, int, str]] = []
        subs: dict[str, str] = {}
        local: dict[str, str] = {}          # bare names this file binds
        aliases: dict[str, str] = {}        # alias -> dotted module

        def note(old: str, new: str) -> None:
            if subs.setdefault(old, new) != new:
                self.problems.append(f"{rel_root}: {old} would become both "
                                     f"{subs[old]} and {new}")

        # 1. definitions in this file
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                new = self.rename.get((rel, node.name))
                if new:
                    kw = "class " if isinstance(node, ast.ClassDef) else (
                        "async def " if isinstance(node, ast.AsyncFunctionDef) else "def ")
                    col = node.col_offset + len(kw)
                    reps.append((node.lineno, col, col + len(node.name), new))
                    local[node.name] = new
                    note(node.name, new)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if isinstance(t, ast.Name) and (rel, t.id) in self.rename:
                        local[t.id] = self.rename[(rel, t.id)]
                        note(t.id, local[t.id])

        # 2. imports: the names, the module paths, the module aliases
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                changed = False
                names: list[tuple[str, str | None]] = []
                for a in node.names:
                    # A helper's module and the name the package imports
                    # back from it are spelled the same, so the name is
                    # tried first — as Python resolves it.
                    new = self.new_of(self.resolve(node.module, a.name))
                    sub = f"{node.module}.{a.name}"
                    if new is None and rel_of_module(sub):
                        aliases[a.asname or a.name] = sub
                        names.append((a.name, a.asname))
                        continue
                    if new:
                        note(a.name, new)
                        changed = True
                        if a.asname is None:
                            local[a.name] = new
                        names.append((new, a.asname))
                    else:
                        names.append((a.name, a.asname))
                module = self.module_moves.get(node.module, node.module)
                if module != node.module:
                    changed = True
                if changed:
                    reps.append((node.lineno, -1, node.end_lineno or node.lineno,
                                 spell_import(node, module, names, "from", lines)))
            elif isinstance(node, ast.Import):
                names = []
                changed = False
                for a in node.names:
                    if a.name.startswith("mechbench_compute"):
                        aliases[a.asname or a.name] = a.name
                    moved = self.module_moves.get(a.name, a.name)
                    changed |= moved != a.name
                    names.append((moved, a.asname))
                if changed:
                    reps.append((node.lineno, -1, node.end_lineno or node.lineno,
                                 spell_import(node, "", names, "import", lines)))

        # 3. what a module exports by name: `__all__` is a list of
        # strings, and a renamed export has to be respelled there too.
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names_of = (node.targets if isinstance(node, ast.Assign) else [node.target])
            if not any(isinstance(x, ast.Name) and x.id == "__all__" for x in names_of):
                continue
            for el in ast.walk(node):
                if (isinstance(el, ast.Constant) and isinstance(el.value, str)
                        and el.value in local):
                    note(el.value, local[el.value])
                    reps.append((el.lineno, el.col_offset + 1,
                                 (el.end_col_offset or 0) - 1, local[el.value]))

        # 4. bare names, attributes through a module alias, strings
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in local:
                reps.append((node.lineno, node.col_offset, node.end_col_offset, local[node.id]))
            elif isinstance(node, ast.Attribute):
                parts = attr_chain(node)
                if not parts or len(parts) < 2 or parts[0] not in aliases:
                    continue
                # `alias.NAME`, and `alias.mid.NAME` where each step is a
                # module: `judge.chat_mod.read_records` reaches through
                # the alias another module binds.
                module = self.inv.walk_modules(aliases[parts[0]], parts[1:-1])
                new = self.new_of(self.resolve(module, parts[-1])) if module else None
                if new:
                    note(node.attr, new)
                    end = node.end_col_offset or 0
                    reps.append((node.lineno, end - len(node.attr), end, new))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                new = self.string_rename(node.value)
                if new:
                    note(node.value.rsplit(".", 1)[-1] if "." in node.value else node.value,
                         new.rsplit(".", 1)[-1] if "." in new else new)
                    reps.append((node.lineno, node.col_offset + 1,
                                 (node.end_col_offset or 0) - 1, new))
            elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
                # The params gate names a site as ("ops/x/y.py", "fn").
                a, b = node.elts
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)
                        and isinstance(b, ast.Constant) and isinstance(b.value, str)):
                    continue
                new = self.rename.get((a.value, b.value))
                if new:
                    note(b.value, new)
                    reps.append((b.lineno, b.col_offset + 1, (b.end_col_offset or 0) - 1, new))
                moved = self.moves.get(a.value)
                if moved:
                    reps.append((a.lineno, a.col_offset + 1, (a.end_col_offset or 0) - 1, moved))

        # 5. a bare name this file also binds locally would be renamed wrongly
        top_targets = [t for st in tree.body
                       for t in (st.targets if isinstance(st, ast.Assign)
                                 else [st.target] if isinstance(st, ast.AnnAssign) else [])]
        for old in local:
            for node in ast.walk(tree):
                bad = (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and node.name == old and node not in tree.body)
                bad |= isinstance(node, ast.arg) and node.arg == old
                bad |= (isinstance(node, ast.Name) and node.id == old
                        and isinstance(node.ctx, (ast.Store, ast.Del))
                        and not any(node is t for t in top_targets))
                if bad:
                    self.problems.append(
                        f"{rel_root}:{node.lineno} also binds {old!r} — rename it by hand")
        if not reps:
            return None
        reps += realign(src, reps)
        return "\n".join(splice(lines, reps)), subs

    def string_rename(self, text: str) -> str | None:
        """`mechbench_compute.x.y.name` as a string, and a module path."""
        if not text.startswith("mechbench_compute"):
            return None
        if text in self.module_moves:
            return self.module_moves[text]
        module, _, name = text.rpartition(".")
        new = self.new_of(self.resolve(module, name))
        if new:
            return f"{self.module_moves.get(module, module)}.{new}"
        return None

    # -- doing it ---------------------------------------------------------
    def apply(self, dry: bool) -> None:
        before: dict[str, str] = {}
        after: dict[str, str] = {}
        for path in targets():
            rel = str(path.relative_to(ROOT))
            before[rel] = path.read_text()
            planned = self.plan(path)
            if planned is None:
                continue
            text, subs = planned
            after[rel] = text
            self.subs[rel] = subs
        if self.problems:
            for p in self.problems:
                print("  !!", p)
            raise SystemExit("refusing to rewrite: the names above need a person")
        print(f"{len(self.rows)} renames, {len(self.moves)} files move, "
              f"{len(self.held)} rows held for by_hand.names()")
        print(f"{len(after)} files change")
        if dry:
            for rel in sorted(after):
                print("  ", rel)
            return
        for rel, text in after.items():
            (ROOT / rel).write_text(text)
        for old, new in sorted(self.moves.items()):
            if not (PKG / old).exists() and (PKG / new).exists():
                continue          # already applied: the table is cumulative
            subprocess.run(["git", "mv", f"mechbench_compute/{old}", f"mechbench_compute/{new}"],
                           cwd=ROOT, check=True)
        if self.verify(before, after):
            raise SystemExit(1)

    # -- proving it -------------------------------------------------------
    def verify(self, before: dict[str, str], after: dict[str, str]) -> int:
        """A renaming and nothing else.

        The two trees are walked in step. Every node must be the same
        node with the same fields, and the only difference allowed is an
        identifier — a def's name, a Name, an attribute, an import alias
        or module path, a string a test patches with — where the pair
        (before, after) is a row of the table. Anything else, including a
        node that moved or a literal that changed, is a failure."""
        bad = 0
        for rel, old_text in sorted(before.items()):
            new_text = after.get(rel)
            if new_text is None:
                if old_text != (ROOT / rel).read_text():
                    print(f"  !! {rel} changed and was not planned")
                    bad += 1
                continue
            why = self.same(ast.parse(old_text), ast.parse(new_text), rel)
            if why:
                print(f"  !! {rel}: {why}")
                bad += 1
        print("verify:", "clean" if not bad else f"{bad} PROBLEMS")
        return bad

    IDENTS: ClassVar[set[str]] = {"name", "id", "attr", "asname", "module", "arg"}

    def allowed(self, old: str | None, new: str | None) -> bool:
        if old == new:
            return True
        if old is None or new is None:
            return False
        return (old, new) in self.pairs or self.module_moves.get(old) == new

    def same(self, a: ast.AST, b: ast.AST, rel: str, where: str = "") -> str | None:
        if type(a) is not type(b):
            return f"{where}: {type(a).__name__} became {type(b).__name__}"
        for field in a._fields:
            va, vb = getattr(a, field, None), getattr(b, field, None)
            spot = f"{where}.{type(a).__name__}.{field}"
            if isinstance(va, list) or isinstance(vb, list):
                if not isinstance(va, list) or not isinstance(vb, list) or len(va) != len(vb):
                    return f"{spot}: the list changed length"
                for i, (x, y) in enumerate(zip(va, vb)):
                    if isinstance(x, ast.AST) or isinstance(y, ast.AST):
                        why = self.same(x, y, rel, f"{spot}[{i}]")
                        if why:
                            return why
                    elif x != y and not (field == "names" and self.allowed(x, y)):
                        return f"{spot}[{i}]: {x!r} became {y!r}"
            elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                why = self.same(va, vb, rel, spot)
                if why:
                    return why
            elif va != vb:
                if field in self.IDENTS and self.allowed(va, vb):
                    continue
                if (field == "value" and isinstance(va, str) and isinstance(vb, str)
                        and (self.string_rename(va) == vb or (va, vb) in self.pairs)):
                    continue
                return f"{spot}: {va!r} became {vb!r}"
        return None


class Canon(ast.NodeTransformer):
    """The tree with a substitution applied to every identifier it names,
    so a before-tree and an after-tree can be compared for equality."""

    def __init__(self, subs: dict[str, str]) -> None:
        self.subs = subs

    def sub(self, name: str | None) -> str | None:
        return self.subs.get(name, name) if name else name

    def visit_Name(self, node):
        node.id = self.sub(node.id)
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node.attr = self.sub(node.attr)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.name = self.sub(node.name)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_alias(self, node):
        node.name = self.sub(node.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = self.sub(node.module)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.subs.get(node.value, node.value)
            head, _, tail = node.value.rpartition(".")
            if head and tail in self.subs:
                node.value = f"{self.subs.get(head, head)}.{self.subs[tail]}"
        return node


def at_head(rel: str) -> str | None:
    out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    return out.stdout if out.returncode == 0 else None


def verify_applied(r: Renamer) -> int:
    """Re-check a pass that has already been applied and moved: the
    before text comes from HEAD, the after text from the working tree."""
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for old, new in r.moves.items():
        old_rel, new_rel = f"mechbench_compute/{old}", f"mechbench_compute/{new}"
        head = at_head(old_rel)
        if head is None:
            print(f"  !! {old_rel} is not in HEAD")
            continue
        before[new_rel] = head
        after[new_rel] = (ROOT / new_rel).read_text()
        r.subs[new_rel] = {}
    for path in targets():
        rel = str(path.relative_to(ROOT))
        if rel in after:
            continue
        head = at_head(rel)
        if head is None:
            continue
        text = path.read_text()
        before[rel] = head
        if text != head:
            after[rel] = text
    return r.verify(before, after)


def main() -> None:
    counts: dict[str, int] = defaultdict(int)
    r = Renamer()
    for row in r.rows:
        counts[row["kind"]] += 1
    print("by kind:", dict(counts))
    if "--verify" in sys.argv:
        raise SystemExit(1 if verify_applied(r) else 0)
    r.apply("--dry" in sys.argv)


if __name__ == "__main__":
    main()
