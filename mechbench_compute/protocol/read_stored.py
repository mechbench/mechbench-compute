from __future__ import annotations

from typing import Any


def read_stored(payload: Any, label: str, *, on_download=None,
                on_download_bytes=None) -> Any:
    from pathlib import Path

    from mechbench_compute import bench, stored_files
    from mechbench_compute import tensors as tensors_mod

    if tensors_mod.is_tensor(payload):
        if on_download is not None:
            on_download(str(label), None)
        return tensors_mod.materialize(
            payload, str(label), bench.get_file_chunks,
            Path.home() / ".mechbench" / "tensors",
            on_bytes=on_download_bytes)
    if stored_files.is_files(payload):
        if on_download is not None:
            on_download(str(label), None)
        return stored_files.restore(payload, bench.get_file_chunks,
                                    on_bytes=on_download_bytes)
    return payload


def prepare_emit(value: Any, target: str) -> Any:
    from mechbench_compute import bench, stored_files
    from mechbench_compute import tensors as tensors_mod

    if tensors_mod.is_tensor(value):
        return tensors_mod.upload(
            value, target,
            lambda lab, path: bench.put_file(lab, path, kind="tensor_shard"),
            have=bench.list_prefix_hashes(target))
    if stored_files.needs_files(value):
        return stored_files.offload(
            value, target,
            lambda lab, path: bench.put_file(lab, path, kind=stored_files.FILE_KIND),
            have=bench.list_prefix_hashes(target))
    return value
