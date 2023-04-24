"""Test for minitorch/operator.py."""
from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists
from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that the main operators all return the same value of the python version."""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    """Check that relu is 0.0 when a < 0 and a when a > 0."""
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    """Check that the derivative of relu 0.0 when a < 0 and 1 when a > 0."""
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    """Check that id returns the value equal to input."""
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    """Check that a - 1.0 is always less than a."""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    """Check that max always return bigger value."""
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    """Check that a can only equal to a."""
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function.

    Specifically,
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is strictly increasing.
    """
    assert 0.0 <= sigmoid(a) <= 1.0
    assert_close(1 - sigmoid(a), sigmoid(-a))
    assert_close(sigmoid(0), 0.5)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Test the transitive property of less-than (a < b and b < c implies a < c)."""
    if lt(a, b) and lt(b, c):
        assert lt(a, c)


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a: float, b: float) -> None:
    """Task0_2.

    Write a test that ensures that :func:`minitorch.operators.mul` is symmetric.
    i.e. gives the same value regardless of the order of its input.
    """
    assert_close(mul(a, b), mul(b, a))


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(a: float, b: float, c: float) -> None:
    r"""Task0_2.

    Write a test that ensures that your operators distribute.
    i.e. :math:`z \times (x + y) = z \times x + z \times y`
    """
    assert_close(mul(c, add(a, b)), add(mul(c, a), mul(c, b)))


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_other(a: float, b: float, c: float) -> None:
    """Task0_2.

    Write a test that ensures some other property holds for your functions.
    Test associative of mul, add
    """
    assert_close(mul(a, mul(b, c)), mul(mul(a, b), c))
    assert_close(add(a, add(b, c)), add(add(a, b), c))


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    """Check that zipwith will add the corresponding element."""
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """Task0_3.

    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    assert_close(add(sum(ls1), sum(ls2)), sum(addLists(ls1, ls2)))


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    """Check the sum always have similar result."""
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    """Check that prod return the same value with of the python version."""
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    """Check that add minus sign can reverse negList operation."""
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    """Check that fn can take one argument."""
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    """Check that fn can take two argument."""
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    """Check that all derivative function works."""
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
