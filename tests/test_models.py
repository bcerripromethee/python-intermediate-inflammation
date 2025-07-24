"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise

@pytest.mark.parametrize("test, expected",
                         [
                             ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
                             ([ [1, 2], [3, 4], [5, 6] ], [3, 4])
                         ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max_zeros():
    """Test that max function works for an array of zeros."""

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    
    test_result = np.array([0, 0])

    npt.assert_array_almost_equal(daily_max(test_input), test_result)

@pytest.mark.parametrize("test, expected", [
                             ([ [1, 50, 0], [522, 10, 0], [10, 10, 0] ], [522, 50, 0]),
                             ([[88, 99, 0], [1, -1, 15], [55, 63, 0]], [88, 99, 15])
                         ])
def test_daily_max_integers(test, expected):
    """Test that max function works for an array of integers."""

    npt.assert_array_almost_equal(daily_max(np.array(test)), np.array(expected))


def test_daily_min_zeros():
    """Test that max function works for an array of zeros."""

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    
    test_result = np.array([0, 0])

    npt.assert_array_almost_equal(daily_min(test_input), test_result)

@pytest.mark.parametrize("test, expected", [
                             ([ [1, 50, 0], [522, 10, 0], [10, 10, 0] ], [1, 10, 0]),
                             ([[88, 99, 0], [1, -1, 15], [55, 63, 0]], [1, -1, 0])
                         ])
def test_daily_min_integers(test, expected):
    """Test that max function works for an array of integers."""

    npt.assert_array_almost_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_mean([['Hello', 'there'], ['Comment', 'vas-tu']])


@pytest.mark.parametrize(
    "test, expected, expect_raises, match",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None, None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None, None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None, None),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], None, ValueError, "inflammation values should be non-negative"),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None, None),
        ([4, 5, 6], None, ValueError, "inflammation array should be 2-dimensional"),
        ('hello', None, TypeError, "data input should be ndarray"),
        (3, None, TypeError, "data input should be ndarray"),
    ]
)
def test_patient_normalise(test, expected, expect_raises, match):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises, match=match):
            patient_normalise(test)
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


