import unittest

import kgcn.src.neighbourhood.data.executor as data_ex
import kgcn.src.neighbourhood.data.sampling.ordered as ordered
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.data.traversal as trv
import kgcn.src.neighbourhood.data.traversal_mocks as mock
import kgcn.src.preprocess.raw_array_builder as builders
import kgcn.src.preprocess.raw_array_builder as raw


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
        role_direction = data_ex.TARGET_PLAYS
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
        role_direction = data_ex.NEIGHBOUR_PLAYS
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


class TestIntegrationsNeighbourTraversalFromEntity(unittest.TestCase):
    def setUp(self):
        import grakn
        entity_query = "match $x isa company, has name 'Google'; get;"
        uri = "localhost:48555"
        keyspace = "test_schema"
        client = grakn.Grakn(uri=uri)
        session = client.session(keyspace=keyspace)
        self._tx = session.transaction(grakn.TxType.WRITE)

        neighbour_sample_sizes = (4, 3)
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

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################

        raw_builder = raw.RawArrayBuilder(neighbour_sample_sizes, len(concepts))
        self._raw_arrays = raw_builder.build_raw_arrays(neighbour_roles)

    def test_array_values(self):
        with self.subTest('role_type not empty'):
            self.assertFalse('' in self._raw_arrays[0]['role_type'])
        with self.subTest('neighbour_type not empty'):
            self.assertFalse('' in self._raw_arrays[0]['neighbour_type'])


class TestIntegrationsNeighbourTraversal(unittest.TestCase):
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

    def test_all_indices_visited(self):
        print(self._raw_builder.indices_visited)
        self.assertEqual(len(self._raw_builder.indices_visited), 6+2+1)


