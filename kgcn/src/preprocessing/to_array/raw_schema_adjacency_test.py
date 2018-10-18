import collections
import unittest

import numpy as np

import kgcn.src.preprocessing.to_array.raw_schema_adjacency as adj


class TestBuildAdjacencyMatrix(unittest.TestCase):

    def test_input_output(self):
        d = collections.OrderedDict((('animal', ['animal']),
                                     ('dog', ['animal', 'dog']),
                                     ('collie', ['animal', 'dog', 'collie']),
                                     ('fish', ['animal', 'fish']),
                                     ('border collie', ['animal', 'dog', 'collie', 'border collie']),
                                     ))
        print(d.items())

        output_1, output_2 = adj.build_adjacency_matrix(d)

        with self.subTest('schema types'):
            desired_types = list(d.keys())
            np.testing.assert_array_equal(output_1, desired_types)

        with self.subTest('type adjacency matrix'):
            desired_output = np.identity(len(d), dtype=np.int64)
            desired_output[:, 0] = 1
            desired_output[[2, 4], 1] = 1
            desired_output[4, 2] = 1

            print("output:")
            print(output_2)
            np.testing.assert_array_equal(output_2, desired_output)
