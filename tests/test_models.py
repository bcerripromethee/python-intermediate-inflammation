"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

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