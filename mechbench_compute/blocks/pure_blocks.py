from __future__ import annotations


# An operation that has its own file (docs/OPS_LAYOUT.md) and runs with
# no executor is callable here by name, as the ones above are. It is
# looked up when asked for, never at import: finding the operations
# imports their files, their files import helpers from packages like
# this one, and a package's `__init__` that went looking for operations
# while one of them was half-imported would find it without its OP.
class _PureBlocks(dict):
    @staticmethod
    def _elsewhere() -> frozenset[str]:
        from mechbench_compute import ops

        return ops.find_standalone()

    def __contains__(self, name: object) -> bool:
        return dict.__contains__(self, name) or name in self._elsewhere()

    def __missing__(self, name: str):
        if name not in self._elsewhere():
            raise KeyError(name)
        from mechbench_compute import ops

        return lambda inputs, params: ops.run_standalone(name, inputs, params)

    def __iter__(self):
        yield from dict.__iter__(self)
        yield from sorted(n for n in self._elsewhere() if not dict.__contains__(self, n))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def keys(self):
        return list(self)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default
