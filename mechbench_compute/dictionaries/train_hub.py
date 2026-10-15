from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from typing import Any

from mechbench_compute.dictionaries.train_corpus import order

FIELDS = ("repo", "config", "split", "revision", "field", "chat", "files")

MISSING_PYARROW = ("dictionary/train reads a Hugging Face dataset's parquet files through pyarrow, which "
                   "is not installed here: install `mechbench-compute[dictionaries]`, or bind a stored "
                   "records object to `records`")


def read_dataset(repo: str, revision: str) -> tuple[str, list[str], list[Any]]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, revision=revision)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return str(info.sha), [s.rfilename for s in info.siblings or []], list(card.get("configs") or [])


def open_parquet(repo: str, commit: str, path: str) -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError(MISSING_PYARROW) from None
    from huggingface_hub import HfFileSystem

    return pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}@{commit}/{path}", "rb"))


def read_patterns(configs: Sequence[Any], config: str, split: str, repo: str) -> list[str]:
    found = next((c for c in configs if isinstance(c, Mapping) and c.get("config_name") == config), None)
    if found is None:
        names = [str(c.get("config_name")) for c in configs if isinstance(c, Mapping)]
        raise ValueError(f"dictionary/train: {repo} has no config {config!r}"
                         + (f"; its configs are {', '.join(names)}" if names else
                            "; its card declares none, so name its parquet files with `files`"))
    data = found.get("data_files")
    if isinstance(data, str) or (isinstance(data, list) and all(isinstance(d, str) for d in data)):
        by_split = {"train": data}
    elif isinstance(data, Mapping):
        by_split = dict(data)
    elif isinstance(data, list):
        by_split = {str(d.get("split")): d.get("path") for d in data if isinstance(d, Mapping)}
    else:
        by_split = {}
    if split not in by_split:
        raise ValueError(f"dictionary/train: {repo}'s config {config!r} has no split {split!r}"
                         + (f"; its splits are {', '.join(sorted(by_split))}" if by_split else ""))
    paths = by_split[split]
    return [paths] if isinstance(paths, str) else [str(p) for p in paths]


class HubSource:
    def __init__(self, spec: Mapping[str, Any], seed: int, index: int) -> None:
        unknown = sorted(set(spec) - set(FIELDS))
        if unknown:
            raise ValueError(f"dictionary/train: a hub source takes {', '.join(FIELDS)}; not {', '.join(unknown)}")
        if not isinstance(spec.get("repo"), str) or not spec["repo"]:
            raise ValueError("dictionary/train: a hub source names its dataset's `repo`")
        if spec.get("field") is not None and spec.get("chat") is not None:
            raise ValueError("dictionary/train: a hub source reads a text `field` or a `chat` field, not both")
        self.repo = spec["repo"]
        self.config = str(spec.get("config") or "default")
        self.split = str(spec.get("split") or "train")
        self.revision = str(spec.get("revision") or "main")
        self.chat = spec.get("chat")
        self.field = str(self.chat or spec.get("field") or "text")
        self.seed, self.index = seed, index
        self.commit, listed, configs = read_dataset(self.repo, self.revision)
        globs = spec.get("files")
        patterns = ([globs] if isinstance(globs, str) else list(globs)) if globs is not None else \
            read_patterns(configs, self.config, self.split, self.repo)
        self.files = sorted(f for f in listed if f.endswith(".parquet")
                            and any(fnmatch.fnmatch(f, p) or fnmatch.fnmatch(f, p + ".parquet") for p in patterns))
        if not self.files:
            raise ValueError(f"dictionary/train: no parquet file of {self.repo} at {self.commit} matches "
                             f"{', '.join(patterns)}; this operation reads datasets stored as parquet")
        self.epoch = self.file_pos = self.group_pos = self.row_pos = 0
        self.file: Any = None
        self.groups: Any = None
        self.rows: list[Any] | None = None
        self.perm: Any = None

    def file_index(self) -> int:
        return int(order(self.seed, self.index, self.epoch, n=len(self.files))[self.file_pos])

    def load(self) -> None:
        while True:
            if self.file_pos >= len(self.files):
                self.epoch, self.file_pos, self.group_pos, self.row_pos = self.epoch + 1, 0, 0, 0
                self.file = None
            f = self.file_index()
            if self.file is None:
                self.file = open_parquet(self.repo, self.commit, self.files[f])
                self.groups = order(self.seed, self.index, self.epoch, f, n=self.file.num_row_groups)
            if len(self.groups):
                break
            self.file, self.file_pos, self.group_pos = None, self.file_pos + 1, 0
        g = int(self.groups[self.group_pos])
        column = self.file.read_row_group(g, columns=[self.field]).column(self.field).to_pylist()
        self.rows = column
        self.perm = order(self.seed, self.index, self.epoch, f, g, n=len(column))

    def advance(self) -> None:
        self.rows, self.row_pos = None, 0
        self.group_pos += 1
        if self.group_pos >= len(self.groups):
            self.file, self.file_pos, self.group_pos = None, self.file_pos + 1, 0

    def next_doc(self) -> Mapping[str, Any]:
        while True:
            if self.rows is None:
                self.load()
            assert self.rows is not None
            if self.row_pos >= len(self.rows):
                self.advance()
                continue
            value = self.rows[int(self.perm[self.row_pos])]
            self.row_pos += 1
            if value:
                return {"messages": value} if self.chat else {"text": str(value)}

    def cursor(self) -> list[int]:
        return [self.epoch, self.file_pos, self.group_pos, self.row_pos]

    def seek(self, cursor: Sequence[int]) -> None:
        self.epoch, self.file_pos, self.group_pos, row = (int(c) for c in cursor)
        self.file, self.rows = None, None
        if self.file_pos < len(self.files):
            self.load()
        self.row_pos = row

    def describe(self) -> dict[str, Any]:
        read = {"chat": self.chat} if self.chat else {"field": self.field}
        return {"hub": {"repo": self.repo, "config": self.config, "split": self.split,
                        "revision": self.revision, "commit": self.commit, "files": len(self.files), **read},
                "epochs": self.epoch}
