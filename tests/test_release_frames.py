import gc
import weakref

from mechbench_compute.protocol.release import release_device_memory, release_frames


class Weights:
    pass


def fail_holding(held):
    model = Weights()
    held.append(weakref.ref(model))
    raise ValueError("the node failed")


def fail_with_a_cause(held):
    try:
        fail_holding(held)
    except ValueError as e:
        raise RuntimeError("wrapped") from e


def stored_failure(fn):
    held = []
    try:
        fn(held)
    except Exception as exc:  # noqa: BLE001
        stored = release_frames(exc)
    return stored, held[0]


def test_a_stored_failure_keeps_its_message_and_not_the_nodes_locals():
    gc.disable()
    try:
        stored, model = stored_failure(fail_holding)
        assert model() is None
        assert str(stored) == "the node failed"
        assert stored.__traceback__ is not None
    finally:
        gc.enable()


def test_the_cause_of_a_failure_is_released_too():
    gc.disable()
    try:
        stored, model = stored_failure(fail_with_a_cause)
        assert model() is None
        assert isinstance(stored.__cause__, ValueError)
    finally:
        gc.enable()


def test_without_the_release_the_locals_outlive_the_failure():
    gc.disable()
    try:
        held = []
        try:
            fail_holding(held)
        except ValueError as exc:
            stored = exc
        assert held[0]() is not None
        del stored
    finally:
        gc.enable()


def test_releasing_device_memory_runs_without_a_gpu():
    release_device_memory()
