import unittest

import kgcn.src.neighbourhood.data.strategy as strat
import kgcn.src.neighbourhood.data.traversal as trv
import kgcn.src.neighbourhood.data.traversal_mocks as mock
import kgcn.src.preprocess.raw_array_building as builders


class TestNeighbourTraversalFromEntity(unittest.TestCase):

    def setUp(self):
        self._neighbourhood_sizes = (3, 2)
        self._concept_info_with_neighbourhood = mock.mock_traversal_output()

        self._concept_infos_with_neighbourhoods = trv.concepts_with_neighbourhoods_to_neighbour_roles(
            [self._concept_info_with_neighbourhood, self._concept_info_with_neighbourhood])

        # thing_type_labels = ['name', 'person', '@has-name', 'employment', 'company']
        # role_type_labels = ['employee', 'employer', '@has-name-value', '@has-name-owner']

        self._n_starting_concepts = len(self._concept_infos_with_neighbourhoods)

        self._builder = builders.RawArrayBuilder(self._neighbourhood_sizes, self._n_starting_concepts)

        self._expected_dims = [self._n_starting_concepts] + list(self._neighbourhood_sizes) + [1]

    def _check_dims(self, arrays):
        # We expect dimensions:
        # (2, 3, 2, 1)
        # (2, 2, 1)
        # (2, 1)
        exp = [[self._expected_dims[0]] + list(self._expected_dims[i+1:]) for i in range(len(self._expected_dims)-1)]
        for i in range(len(self._expected_dims) - 1):
            with self.subTest(exp[i]):
                self.assertEqual(arrays[i]['neighbour_type'].shape, tuple(exp[i]))

    def test_build_raw_arrays(self):

        depthwise_arrays = self._builder.build_raw_arrays(self._concept_infos_with_neighbourhoods)
        self._check_dims(depthwise_arrays)
        with self.subTest('spot-check thing type'):
            self.assertEqual(depthwise_arrays[-1]['neighbour_type'][0, 0], 'person')
        with self.subTest('spot-check role type'):
            self.assertEqual(depthwise_arrays[0]['role_type'][0, 0, 0, 0], 'employer')
        with self.subTest('check role_type absent in final arrays'):
            self.assertFalse('role_type' in list(depthwise_arrays[-1].keys()))
        with self.subTest('check role_direction absent in final arrays'):
            self.assertFalse('role_direction' in list(depthwise_arrays[-1].keys()))

    def test_initialised_array_sizes(self):

        initialised_arrays = self._builder._initialise_arrays()
        self._check_dims(initialised_arrays)

    def test__determine_values_to_put_with_entity(self):
        role_label = 'employer'
        role_direction = strat.TARGET_PLAYS
        neighbour_type_label = 'company'
        neighbour_data_type = None
        neighbour_value = None
        values_dict = self._builder._determine_values_to_put(role_label, role_direction, neighbour_type_label,
                                                             neighbour_data_type, neighbour_value)
        expected_result = {"role_type": 'employer',
                           'role_direction': role_direction,
                           'neighbour_type': 'company'
                           }
        self.assertEqual(values_dict, expected_result)

    def test__determine_values_to_put_with_string_attribute(self):
        role_label = '@has-name-value'
        role_direction = strat.NEIGHBOUR_PLAYS
        neighbour_type_label = 'name'
        neighbour_data_type = 'string'
        neighbour_value = 'Person\'s Name'
        values_dict = self._builder._determine_values_to_put(role_label, role_direction, neighbour_type_label,
                                                             neighbour_data_type, neighbour_value)
        expected_result = {"role_type": '@has-name-value',
                           'role_direction': role_direction,
                           'neighbour_type': 'name',
                           'neighbour_data_type': 'string',
                           'neighbour_value_string': neighbour_value}
        self.assertEqual(expected_result, values_dict)
