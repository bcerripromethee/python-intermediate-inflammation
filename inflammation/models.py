"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data 
                (each row contains measurements for a single patient across all days).
    :returns: An array of mean values of measurements for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data 
                (each row contains measurements for a single patient across all days).
    :returns: An array of maximal values of measurements for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data 
                (each row contains measurements for a single patient across all days).
    :returns: An array of minimal values of measurements for each day."""
    return np.min(data, axis=0)



def s_dev(data):
    """Computes and returns standard deviation for data."""
    mmm = np.mean(data, axis=0)
    devs = []
    for entry in data:
        devs.append((entry - mmm) * (entry - mmm))

    s_dev2 = sum(devs) / len(data)
    return {'standard deviation': s_dev2}

def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array"""
    if not isinstance(data, np.ndarray):
        raise TypeError("data input should be ndarray")
    if len(data.shape) != 2:
        raise ValueError("inflammation array should be 2-dimensional")
    if np.any(data < 0):
        raise ValueError('inflammation values should be non-negative')
    max_values = np.max(data, axis=1).reshape(-1, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalized = data / max_values
    
    normalized[np.isnan(normalized)] = 0
    normalized[normalized < 0] = 0
    return normalized

