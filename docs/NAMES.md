# What a name says

Two rules. Both are about the reader of a call site, who has the name
and not the definition.

**A leading underscore means "mine alone."** It is a promise that the
name is private to its file: change it, move it, delete it, and nothing
else notices. On a name another module of the package imports, reads as
an attribute, or patches by string, the promise is false, and the
reader who believed it is the one who gets hurt. Either drop the
underscore, or move the definition to the one file that uses it.

**A function's name starts with a verb.** It says what it does, not
what it returns, so that a call site reads as an action:
`read_last_logp(logits)` is what happens, `last_logp(logits)` is a noun
pretending to be a call. `is_` / `has_` predicates already read as a
question and are counted apart.

The verb rule is held where the layout is — an operation's file under
`ops/`, and a helper file named for the one thing it holds
(OPS_LAYOUT.md). Those are the files whose whole claim is that the path
and the name tell you what is inside. The older topic modules are not
held to it.

## Two judgement calls

Both are about what counts as another file naming something.

- **A package `__init__.py` importing a name back is a re-export, not a
  second file.** Every file that reaches the name through the package
  is already counted at the far end of the chain, so counting the
  re-export as well would make a helper nobody uses look as though it
  crossed a file boundary.
- **A test naming a private does not make the underscore false.**
  Reaching into a module's privates is what a test — and a script —
  is allowed to do. The name is still private to the package, and the
  promise it makes is still kept.

## The gate

`tests/test_names.py`. It reads every top-level definition in
`mechbench_compute/`, and every reference to one from the package, the
suite and `scripts/` — through an import, through an attribute on a
module bound under any alias, and through a string a test patches with.
It fails naming the file, the line and the symbol.

The verb list lives in that file. It is what the codebase already says
well, learned by reading it; a first word that is a verb and is not
there yet belongs in the list, and the gate's message says so.
