import unittest

import grakn

import grakn_graphsage.src.encoders.raw_array_building as builders
import grakn_graphsage.src.neighbourhood.traversal as trv

client = grakn.Grakn(uri="localhost:48555")
session = client.session(keyspace="test_schema")


class TestNeighbourTraversalFromEntity(unittest.TestCase):
    def setUp(self):
        self.tx = session.transaction(grakn.TxType.WRITE)

        # identifier = "Jacob J. Niesz"
        # entity_query = "match $x isa person, has identifier '{}'; get $x;".format(identifier)
        entity_query = "match $x isa person, has name 'Sundar Pichai'; get;"

        self._concept = list(self.tx.query(entity_query))[0].get('x')
        self._neighbourhood_sizes = (2, 3)
        self._concept_with_neighbourhood = trv.build_neighbourhood_generator(self.tx, self._concept, self._neighbourhood_sizes)

    def tearDown(self):
        self.tx.close()

    def test_build_raw_arrays(self):
        thing_type_labels = ['ownership', 'affiliation', 'name', 'organisation', 'person', '@has-name', 'employment',
                             'membership', 'job-title', 'company', '@has-job-title']
        # role_type_labels = ['property', 'owner', 'party', '@has-name-value', 'group', 'member', '@has-name-owner',
        #                     '@has-job-title-value', 'employer', 'employee', '@has-job-title-owner']

        # TODO Only required while we have a bug on roles as variables in Graql
        role_type_labels = [trv.UNKNOWN_ROLE_NEIGHBOUR_PLAYS_LABEL, trv.UNKNOWN_ROLE_TARGET_PLAYS_LABEL]

        starting_concepts = [self._concept_with_neighbourhood]
        n_starting_concepts = len(starting_concepts)

        builder = builders.RawArrayBuilder(thing_type_labels, role_type_labels, self._neighbourhood_sizes,
                                           n_starting_concepts)
        depthwise_matrices = builder.build_raw_arrays(starting_concepts)
