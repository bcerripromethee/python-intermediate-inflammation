"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import glob
import json
import os
import numpy as np


def load_csv(filename):  
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def load_json(filename):
    """Load a numpy array from a JSON document.
    
    Expected format:
    [
      {
        "observations": [0, 1]
      },
      {
        "observations": [0, 2]
      }    
    ]
    :param filename: Filename of CSV to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]

class CSVDataSource:

    def __init__(self, dir_path: str):
        self.dir_path = dir_path


    def load_inflammation_data(self) -> list[np.ndarray]:
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.dir_path}")
        data = map(load_csv, data_file_paths)

        return list(data)

class JSONDataSource:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path


    def load_inflammation_data(self) -> list[np.ndarray]:
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation JSON files found in path {self.dir_path}")
        data = map(load_json, data_file_paths)

        return list(data)


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)

def compute_standard_deviation_by_day(data):
    """Compute daily standard deviation"""
    means_by_day = map(daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return daily_standard_deviation
