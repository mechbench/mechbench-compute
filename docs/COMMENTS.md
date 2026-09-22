# What a comment is for

This repository is read by agents. A comment costs the same attention as
the code beside it, so it earns its place by saying something the code
cannot: a constraint, an invariant, a reason the obvious approach is
wrong. A comment that narrates how the code came to be answers a
question nobody asked, and its references leave the repository — a task
id points into a private tracker, a date and a version number point at a
tree that is no longer here.

The scope is exactly two things: `#` comments, and docstrings (the first
statement of a module, class or function when it is a string). Every
other string is DATA and is never touched — an operation's declared
prose, a param's description, a kind's name, an error message, a
fixture, a version this code compares against.

## Out

- Task ids, dates, version numbers, commit hashes — and any sentence
  carrying one.
- How the code came to be: was / used to / before / since / moved from /
  replaced / no longer / legacy / "the old ..." / the move / the
  migration / the refactor.
- Who asked or decided.
- Restating the code ("increment i", "return the result").
- A docstring that only repeats the function's name, now that names are
  verbs: `def read_items(...): """Read the items."""` has no docstring.
- Commented-out code.

## Kept, and kept short

- A constraint or invariant the code depends on that is not visible in
  the code: an ordering requirement, why a threshold has its value,
  units, what a caller may assume.
- Why the obvious approach is wrong, stated as the constraint rather
  than the incident: "`Y` reads what `X` writes, so `X` runs first",
  never "fixed when `Y` crashed".
- A one-line docstring saying what a function does when the name cannot
  carry it — cross-file helpers and operation mechanisms especially; a
  longer one only when it says what the caller needs (argument shapes,
  what is returned, what raises).
- A module docstring saying what the module is for, in a sentence or
  two.

Plain, present tense, a fact about the code as it is now. No aphorisms.

## In the suite

A test's docstring or name that says what the test guards is signal —
that is the property, and it is the best place for it. The id of the
task that found the bug is not.

`tests/test_no_history.py` holds the gate: it reads every comment and
docstring under `mechbench_compute/` and `tests/` and fails, naming the
file and line, on a six-digit task id, an ISO date, or a three-part
version number. `scripts/` is excluded. The gate catches the spelling;
the rest of this page is the part it cannot check.
