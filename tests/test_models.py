"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

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


def test_daily_max_integers():
    """Test that max function works for an array of integers."""

    test_input = np.array([[1, 2, 3],
                           [4, 50, 6],
                           [7, 8, 9]])
    
    test_result = np.array([7, 50, 9])

    npt.assert_array_almost_equal(daily_max(test_input), test_result)


def test_daily_min_zeros():
    """Test that max function works for an array of zeros."""

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    
    test_result = np.array([0, 0])

    npt.assert_array_almost_equal(daily_min(test_input), test_result)


def test_daily_min_integers():
    """Test that max function works for an array of integers."""

    test_input = np.array([[11, 2, 0],
                           [3, 40, 1],
                           [51, 6, 13]])
    
    test_result = np.array([3, 2, 0])

    npt.assert_array_almost_equal(daily_min(test_input), test_result)


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_mean([['Hello', 'there'], ['Comment', 'vas-tu']])