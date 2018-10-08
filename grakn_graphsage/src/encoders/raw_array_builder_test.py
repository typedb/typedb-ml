import unittest

import grakn

import grakn_graphsage.src.encoders.raw_array_building as builders
import grakn_graphsage.src.neighbourhood.traversal as trv
import grakn_graphsage.src.neighbourhood.traversal_mocks as mock


import numpy as np
client = grakn.Grakn(uri="localhost:48555")
session = client.session(keyspace="test_schema")


# def expected_output():
#     """
#
#     :return: A list of length 3, each element is a dict. Each dict holds matrices for the different properties we need.
#     """
#
#     # 'role_direction', np.int), ('role_type', np.int), ('thing_type', np.int), ('data_type', np.int),
#     #          ('value_long', np.int), ('value_double', np.float), ('value_boolean', np.bool),
#     #          ('value_date', np.datetime64), ('value_string', np.str)])
#
#     full_shape = (1, 2, 2)
#
#     o = {'role_type': np.full(full_shape[:1], np.nan, np.int)}


class TestNeighbourTraversalFromEntity(unittest.TestCase):

    def setUp(self):
        self._tx = session.transaction(grakn.TxType.WRITE)
        self._neighbourhood_sizes = (2, 2)
        self._concept_info_with_neighbourhood = mock.mock_traversal_output()

        self._top_level_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(
            [self._concept_info_with_neighbourhood])
        
    def tearDown(self):
        self._tx.close()

    def test_build_raw_arrays(self):
        thing_type_labels = ['name', 'person', '@has-name', 'employment', 'company']

        # TODO Only required while we have a bug on roles as variables in Graql
        role_type_labels = ['employee', 'employer', '@has-name-value', '@has-name-owner']

        n_starting_concepts = len(self._top_level_roles)

        builder = builders.RawArrayBuilder(thing_type_labels, role_type_labels, self._neighbourhood_sizes,
                                           n_starting_concepts)
        depthwise_matrices = builder.build_raw_arrays(self._top_level_roles)
