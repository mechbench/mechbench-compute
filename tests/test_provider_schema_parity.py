from typing import get_args

from mechbench_schema import EndpointProvider

from mechbench_compute.providers.registry import registry


def test_the_registry_names_exactly_the_providers_schema_accepts():
    assert set(registry()) == set(get_args(EndpointProvider))
