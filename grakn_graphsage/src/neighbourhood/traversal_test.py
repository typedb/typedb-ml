
import unittest
import grakn
from grakn.service.Session.Concept.Concept import Concept, Role

from grakn_graphsage.src.neighbourhood.neighbour_traversal import build_neighbourhood_generator, NeighbourRole, \
    ConceptWithNeighbourhood, NEIGHBOUR_PLAYS, TARGET_PLAYS, collect_to_tree, UNKNOWN_ROLE_TARGET_PLAYS, \
    UNKNOWN_ROLE_NEIGHBOUR_PLAYS, get_max_depth

client = grakn.Grakn(uri="localhost:48555")
session = client.session(keyspace="test_schema")


class TestNeighbourTraversalFromEntity(unittest.TestCase):
    def setUp(self):
        self.tx = session.transaction(grakn.TxType.WRITE)

        # identifier = "Jacob J. Niesz"
        # entity_query = "match $x isa person, has identifier '{}'; get $x;".format(identifier)
        entity_query = "match $x isa person, has name 'Sundar Pichai'; get;"

        self._concept = list(self.tx.query(entity_query))[0].get('x')

    def tearDown(self):
        self.tx.close()

    def _assert_types_correct(self, concept_with_neighbourhood):
        """
        Check that all of the types in the structure are as expected
        :param concept_with_neighbourhood:
        :return:
        """
        self.assertIsInstance(concept_with_neighbourhood, ConceptWithNeighbourhood)
        self.assertIsInstance(concept_with_neighbourhood.concept, Concept)
        self.assertIn(type(concept_with_neighbourhood.neighbourhood).__name__, ('generator', 'chain'))

        try:
            neighbour_role = next(concept_with_neighbourhood.neighbourhood)

            self.assertIsInstance(neighbour_role, NeighbourRole)

            self.assertTrue(isinstance(neighbour_role.role, Role) or neighbour_role.role in [UNKNOWN_ROLE_TARGET_PLAYS,
                                                                                             UNKNOWN_ROLE_NEIGHBOUR_PLAYS])
            self.assertIn(neighbour_role.role_direction, [TARGET_PLAYS, NEIGHBOUR_PLAYS])
            self.assertTrue(self._assert_types_correct(neighbour_role.neighbour_with_neighbourhood))
        except StopIteration:
            pass

        return True

    def _assert_depth_correct(self, concept_with_neighbourhood):
        neighbour_role = next(concept_with_neighbourhood.neighbourhood, None)
        if neighbour_role is not None:
            self._assert_depth_correct(neighbour_role.neighbour)

    def test_neighbour_traversal_structure_types(self):
        data = ((1,), (2, 3), (2, 3, 4))
        for sample_sizes in data:
            with self.subTest(sample_sizes=str(data)):
                self._concept_with_neighbourhood = build_neighbourhood_generator(self.tx, self._concept, sample_sizes)
                self._assert_types_correct(self._concept_with_neighbourhood)

    def test_neighbour_traversal_check_depth(self):
        data = ((1,), (2, 3), (2, 3, 4))
        for sample_sizes in data:
            with self.subTest(sample_sizes=str(data)):
                self._concept_with_neighbourhood = build_neighbourhood_generator(self.tx, self._concept, sample_sizes)

                collected_tree = collect_to_tree(self._concept_with_neighbourhood)

                with self.subTest("Check number of immediate neighbours"):
                    self.assertEqual(len(collected_tree.neighbourhood), sample_sizes[0])
                with self.subTest("Check max depth of tree"):
                    self.assertEqual(len(sample_sizes), get_max_depth(self._concept_with_neighbourhood))
