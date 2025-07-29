"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views
from inflammation.models import CSVDataSource, JSONDataSource

def analyse_data(data_source: CSVDataSource | JSONDataSource):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    
    data = data_source.load_inflammation_data()
    
    data_std = models.compute_standard_deviation_by_day(data)
    
    graph_data = {
        'standard deviation by day': data_std,
    }
    # views.visualize(graph_data)

    return graph_data