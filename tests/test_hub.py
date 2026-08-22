"""Hub ref parsing (task 000260): repo@revision splitting."""

from mechbench_compute.hub import parse_model_ref


def test_bare_repo_has_no_revision():
    assert parse_model_ref("mlx-community/gemma-4-e2b-it-bf16") == (
        "mlx-community/gemma-4-e2b-it-bf16", None)


def test_pinned_repo_splits_at_the_at_sign():
    assert parse_model_ref("org/model@22a2753") == ("org/model", "22a2753")


def test_trailing_at_means_no_revision():
    assert parse_model_ref("org/model@") == ("org/model", None)
