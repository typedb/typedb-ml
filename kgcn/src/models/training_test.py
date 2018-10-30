import unittest

import numpy as np

import kgcn.src.models.training as training


def trial_data():
    num_samples = 30
    neighbourhood_sizes = (4, 3)
    feature_length = 8
    neighbourhood_shape = list(neighbourhood_sizes) + [feature_length]
    shapes = [[num_samples] + neighbourhood_shape[i:] for i in range(len(neighbourhood_shape))]

    raw_neighbourhood_depths = [np.ones(shape) for shape in shapes]

    label_value = [1, 0]
    raw_labels = [label_value for _ in range(num_samples)]
    labels = raw_labels
    return raw_neighbourhood_depths, labels


class TestTraining(unittest.TestCase):
    def test_train(self):
        neighbourhoods_depths, labels = trial_data()
        training.supervised_train(neighbourhoods_depths, labels)