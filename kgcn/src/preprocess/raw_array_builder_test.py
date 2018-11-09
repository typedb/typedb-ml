import unittest

import numpy as np

import kgcn.src.neighbourhood.data.executor as data_ex
import kgcn.src.neighbourhood.data.sampling.ordered as ordered
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.data.traversal as trv
import kgcn.src.neighbourhood.data.traversal_mocks as mock
import kgcn.src.preprocess.raw_array_builder as builders
import kgcn.src.preprocess.raw_array_builder as raw


class TestDetermineValuesToPut(unittest.TestCase):

    def test__determine_values_to_put_with_entity(self):
        role_label = 'employer'
        role_direction = data_ex.TARGET_PLAYS
        neighbour_type_label = 'company'
        neighbour_data_type = None
        neighbour_value = None
        values_dict = builders.determine_values_to_put(role_label, role_direction, neighbour_type_label,
                                                       neighbour_data_type, neighbour_value)
        expected_result = {"role_type": 'employer',
                           'role_direction': role_direction,
                           'neighbour_type': 'company'
                           }
        self.assertEqual(values_dict, expected_result)

    def test__determine_values_to_put_with_string_attribute(self):
        role_label = '@has-name-value'
        role_direction = data_ex.NEIGHBOUR_PLAYS
        neighbour_type_label = 'name'
        neighbour_data_type = 'string'
        neighbour_value = 'Person\'s Name'
        values_dict = builders.determine_values_to_put(role_label, role_direction, neighbour_type_label,
                                                       neighbour_data_type, neighbour_value)
        expected_result = {"role_type": '@has-name-value',
                           'role_direction': role_direction,
                           'neighbour_type': 'name',
                           'neighbour_data_type': 'string',
                           'neighbour_value_string': neighbour_value}
        self.assertEqual(expected_result, values_dict)


class TestNeighbourTraversalFromMockEntity(unittest.TestCase):

    def setUp(self):
        self._neighbourhood_sizes = (2, 3)
        self._n_starting_concepts = 2

        self._builder = builders.RawArrayBuilder(self._neighbourhood_sizes, self._n_starting_concepts)

        self._expected_dims = [self._n_starting_concepts] + list(reversed(self._neighbourhood_sizes)) + [1]

    def _check_dims(self, arrays):
        # We expect dimensions:
        # (2, 3, 2, 1)
        # (2, 2, 1)
        # (2, 1)
        exp = [[self._expected_dims[0]] + list(self._expected_dims[i+1:]) for i in range(len(self._expected_dims)-1)]
        for i in range(len(self._expected_dims) - 1):
            with self.subTest(exp[i]):
                self.assertEqual(arrays[i]['neighbour_type'].shape, tuple(exp[i]))

    def _concept_infos_with_neighbourhoods_factory(self):
        return trv.concepts_with_neighbourhoods_to_neighbour_roles(
            [mock.mock_traversal_output(), mock.mock_traversal_output()])

    def test_build_raw_arrays(self):

        depthwise_arrays = self._builder.build_raw_arrays(self._concept_infos_with_neighbourhoods_factory())
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

    def test_array_values(self):
        depthwise_arrays = self._builder.build_raw_arrays(self._concept_infos_with_neighbourhoods_factory())
        with self.subTest('role_type not empty'):
            self.assertFalse('' in depthwise_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in depthwise_arrays[0]['neighbour_type'])


class TestIntegrationsNeighbourTraversalFromEntity(unittest.TestCase):
    def setUp(self):
        import grakn
        entity_query = "match $x isa company, has name 'Google'; get;"
        uri = "localhost:48555"
        keyspace = "test_schema"
        client = grakn.Grakn(uri=uri)
        session = client.session(keyspace=keyspace)
        self._tx = session.transaction(grakn.TxType.WRITE)

        neighbour_sample_sizes = (6, 4, 4)
        sampling_method = ordered.ordered_sample

        samplers = []
        for sample_size in neighbour_sample_sizes:
            samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size * 2))

        concepts = [concept.get('x') for concept in list(self._tx.query(entity_query))]

        concept_infos = [data_ex.build_concept_info(concept) for concept in concepts]

        data_executor = data_ex.TraversalExecutor(self._tx)

        neighourhood_traverser = trv.NeighbourhoodTraverser(data_executor, samplers)

        neighbourhood_depths = [neighourhood_traverser(concept_info) for concept_info in concept_infos]

        neighbour_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(neighbourhood_depths)

        # flat = trv.flatten_tree(neighbour_roles)
        # [trv.collect_to_tree(neighbourhood_depth) for neighbourhood_depth in neighbourhood_depths]

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################

        self._raw_builder = raw.RawArrayBuilder(neighbour_sample_sizes, len(concepts))
        self._raw_arrays = self._raw_builder.build_raw_arrays(neighbour_roles)

    def test_array_values(self):
        with self.subTest('role_type not empty'):
            self.assertFalse('' in self._raw_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in self._raw_arrays[0]['neighbour_type'])


class TestIntegrationsNeighbourTraversalWithMock(unittest.TestCase):
    def setUp(self):

        self._neighbour_sample_sizes = (2, 3)

        neighbourhood_depths = [mock.mock_traversal_output()]

        neighbour_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(neighbourhood_depths)

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################

        self._raw_builder = raw.RawArrayBuilder(self._neighbour_sample_sizes, 1)
        self._raw_arrays = self._raw_builder.build_raw_arrays(neighbour_roles)

    def test_role_type_not_empty(self):
        self.assertFalse('' in self._raw_arrays[0]['role_type'])

    def test_neighbour_type_not_empty(self):
        self.assertFalse('' in self._raw_arrays[0]['neighbour_type'])

    def test_string_values_not_empty(self):
        self.assertFalse('' in self._raw_arrays[0]['neighbour_value_string'][0, :, 1, 0])

    def test_data_type_values_not_empty(self):
        self.assertFalse('' in self._raw_arrays[0]['neighbour_data_type'][0, :, 1, 0])

    def test_shapes_as_expected(self):
        self.assertTupleEqual(self._raw_arrays[0]['neighbour_type'].shape, (1, 3, 2, 1))
        self.assertTupleEqual(self._raw_arrays[1]['neighbour_type'].shape, (1, 2, 1))
        self.assertTupleEqual(self._raw_arrays[2]['neighbour_type'].shape, (1, 1))

    # def test_all_indices_visited(self):
    #     print(self._raw_builder.indices_visited)
    #     self.assertEqual(len(self._raw_builder.indices_visited), 6+2+1)

    def test_array_values(self):
        with self.subTest('role_type not empty'):
            self.assertFalse('' in self._raw_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in self._raw_arrays[0]['neighbour_type'])


class TestFillArrayWithRepeats(unittest.TestCase):
    def test_repeat_top_neighbour(self):
        shape = (2, 3, 2, 1)
        arr = np.array([[[[1],
                          [2]],
                         [[3],
                          [4]],
                         [[5],
                          [6]]],
                        [[[7],
                          [8]],
                         [[9],
                          [10]],
                         [[11],
                          [12]]]])
        arr[1, :, 1:, 0] = 0

        builders.fill_array_with_repeats(arr, (1, ..., slice(1), 0), (1, ..., slice(1, None), 0))
        print(arr)

        expected_output = np.array([[[[1],
                                      [2]],
                                     [[3],
                                      [4]],
                                     [[5],
                                      [6]]],
                                    [[[7],
                                      [7]],
                                     [[9],
                                      [9]],
                                     [[11],
                                      [11]]]])
        np.testing.assert_array_equal(expected_output, arr)

    def test_repeat_top_neighbour_deep(self):
        shape = (2, 2, 2, 2, 1)
        arr = np.array([[[[[1],
                           [2]],
                          [[3],
                           [4]]],
                         [[[5],
                           [6]],
                          [[7],
                           [8]]]],
                        [[[[9],
                           [10]],
                          [[11],
                           [12]]],
                         [[[13],
                           [14]],
                          [[15],
                           [16]]]]])
        arr[1, :, :, 1:, 0] = 0

        builders.fill_array_with_repeats(arr, (1, ..., slice(1), 0), (1, ..., slice(1, None), 0))
        print(arr)

        expected_output = np.array([[[[[1],
                                       [2]],
                                      [[3],
                                       [4]]],
                                     [[[5],
                                       [6]],
                                      [[7],
                                       [8]]]],
                                    [[[[9],
                                       [9]],
                                      [[11],
                                       [11]]],
                                     [[[13],
                                       [13]],
                                      [[15],
                                       [15]]]]])
        np.testing.assert_array_equal(expected_output, arr)

    def test_repeat_child_neighbour(self):
        shape = (2, 3, 2, 1)
        arr = np.array([[[[1],
                          [2]],
                         [[3],
                          [4]],
                         [[5],
                          [6]]],
                        [[[7],
                          [8]],
                         [[9],
                          [10]],
                         [[11],
                          [12]]]])
        arr[1, 1:, 1, 0] = 0

        builders.fill_array_with_repeats(arr, ([1], ..., slice(1), [1], [0]), ([1], ..., slice(1, None), [1], [0]))
        print(arr)

        expected_output = np.array([[[[1],
                                      [2]],
                                     [[3],
                                      [4]],
                                     [[5],
                                      [6]]],
                                    [[[7],
                                      [8]],
                                     [[9],
                                      [8]],
                                     [[11],
                                      [8]]]])
        np.testing.assert_array_equal(expected_output, arr)

    def test_repeat_less_than_available(self):
        shape = (2, 3, 2, 1)
        arr = np.array([[[[1],
                          [2]],
                         [[3],
                          [4]],
                         [[5],
                          [6]]],
                        [[[7],
                          [8]],
                         [[9],
                          [10]],
                         [[11],
                          [12]]]])
        arr[0, 2:, 0, 0] = 0

        builders.fill_array_with_repeats(arr, (0, ..., slice(2), 0, 0), (0, ..., slice(2, None), 0, 0))
        print(arr)

        expected_output = np.array([[[[1],
                                      [2]],
                                     [[3],
                                      [4]],
                                     [[1],
                                      [6]]],
                                    [[[7],
                                      [8]],
                                     [[9],
                                      [10]],
                                     [[11],
                                      [12]]]])
        np.testing.assert_array_equal(expected_output, arr)
