from __future__ import annotations

import pytest

from mechbench_compute.ops.tools.calc import MAX_LENGTH, CalcRefused, calculate


def calc(expression: str):
    return calculate({"arguments": {"expression": expression}}, {})["result"]


class TestCalcIsArithmeticWithBounds:
    @pytest.mark.parametrize("expression", [
        "(3 + 4) * 12 / 7", "2 ** -1", "-2 ** 2", "7 // 2", "7 % 3", "1e308 * 10",
        "(1 + 2j) ** 3", "3 ** 9000", "10 ** 4299", "1, 2", "+5 - -5",
    ])
    def test_it_answers_what_python_answers(self, expression):
        assert calc(expression) == eval(expression)

    @pytest.mark.parametrize("expression", ["9 ** 9 ** 9", "2 ** 10 ** 10", "(-7) ** 99999"])
    def test_a_tower_of_powers_is_refused_before_it_is_computed(self, expression):
        with pytest.raises(CalcRefused, match="4300 digits"):
            calc(expression)

    def test_a_product_past_the_digits_is_refused(self):
        with pytest.raises(CalcRefused, match="product"):
            calc("3 ** 9000 * 3 ** 9000")

    @pytest.mark.parametrize("expression", [
        "'a' * 10 ** 9", "(1, 2) * 10 ** 9", "(1,) + (2,)", "__import__('os')",
        "[1, 2]", "1 << 99999", "x",
    ])
    def test_anything_but_numbers_is_refused(self, expression):
        with pytest.raises(CalcRefused):
            calc(expression)

    def test_an_overlong_expression_is_refused(self):
        with pytest.raises(CalcRefused, match=str(MAX_LENGTH)):
            calc("1+" * MAX_LENGTH + "1")
