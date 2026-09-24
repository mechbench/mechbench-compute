# Comments

This repository is read mostly by agents, who read the code itself. A
comment costs the same attention as the code beside it and is rarely
checked when the code changes, so the code carries no comments and no
docstrings, except three kinds.

1. **Directives**, which are not really comments: `# noqa: …`,
   `# type: ignore[…]`, `# pragma: no cover`, `# fmt: …`, a shebang.
   The directive alone, with no prose after it.
2. **Text the product renders or publishes**, which is data: an
   operation's declared prose, CLI help, error messages, and a
   docstring that code reads at run time. The docstrings read at run
   time are registered in `READ_AT_RUN_TIME` in
   `tests/test_no_history.py`, with the reader named; today that is the
   public functions of `bench`, whose docstrings the documentation
   site's client reference renders.
3. **A fact about the outside world, or a rule that spans files**, that
   a capable reader could not get from the code: a provider's
   requirement, a library's behaviour, a limit another repository
   enforces. Write a test that fails if the behaviour goes, and no
   comment. Only when no test can hold it, one line:

   ```python
   # external: <the outside thing> — <the fact>
   ```

Everything else goes: what the code does, the reason the code already
shows, history, dates, versions, roadmaps, section banners,
commented-out code.

No file, string or doc names a task in the tracker (CHANGELOG.md
aside): a reader cannot open it. Say what is true instead.

`tests/test_no_history.py` holds all of this over `mechbench_compute/`,
`tests/` and `scripts/`, and the task-id rule over every file in the
repository.
